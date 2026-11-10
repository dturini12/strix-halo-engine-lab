// DeepSeek V4.1 image preprocessing equals the release's image_processor.load_image (Pillow 11.3): the grid plan and
// the BF16 patch bytes of 31 synthetic pictures (tools/ds41/vision_prep_fixtures.py), the span layout, BF16 rounding.
//   strix-ds41-vision-prep-tests tests/data/ds41_vision/prep_cases.json [FILES_DIR]   (FILES_DIR/prep_cases.json: --files)
#include "strix/ds41_vision.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/serve/json.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

std::string hex(std::span<const std::uint8_t> b) {
    std::string out;
    char t[3];
    for (const auto v : b) {
        std::snprintf(t, sizeof t, "%02x", v);
        out += t;
    }
    return out;
}

strix::image::Rgb8 picture(std::uint32_t w, std::uint32_t h) {
    strix::image::Rgb8 im;
    im.width = w;
    im.height = h;
    im.pixels.resize(std::size_t{w} * h * 3U);
    for (std::uint32_t y = 0; y < h; ++y)
        for (std::uint32_t x = 0; x < w; ++x) {
            auto* p = im.pixels.data() + (std::size_t{y} * w + x) * 3U;
            p[0] = static_cast<std::uint8_t>((7U * x + 3U * y) & 255U);
            p[1] = static_cast<std::uint8_t>(((x * y + 17U * x) >> 3U) & 255U);
            p[2] = static_cast<std::uint8_t>(((x ^ y) * 5U) & 255U);
        }
    return im;
}

}  // namespace

int main(int argc, char** argv) try {
    using namespace strix::deepseek41::vision;
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: strix-ds41-vision-prep-tests prep_cases.json\n";
        return 2;
    }
    const VisionConfig cfg;
    std::size_t n = 0;
    for (int arg = 1; arg < argc; ++arg) {
    const std::string dir = arg == 2 ? std::string(argv[2]) + "/" : "";
    std::ifstream in(arg == 2 ? dir + "prep_cases.json" : std::string(argv[1]), std::ios::binary);
    const auto doc = strix::serve::parse_json(std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()));
    for (const auto& c : doc.find("cases")->as_array()) {
        const auto w = static_cast<std::uint32_t>(c.find("width")->as_int()), h = static_cast<std::uint32_t>(c.find("height")->as_int());
        const std::string name = std::to_string(w) + "x" + std::to_string(h);
        const auto u = [&](const char* k) { return static_cast<std::uint32_t>(c.find(k)->as_int()); };
        const auto plan = plan_grid(w, h, cfg);
        check(plan.best_w == u("best_w") && plan.best_h == u("best_h") && plan.vit_w == u("vit_w") && plan.vit_h == u("vit_h") &&
                  plan.llm_w == u("llm_w") && plan.llm_h == u("llm_h"),
              name + ": grid plan");
        check(plan.span_tokens() == u("span_tokens"), name + ": span tokens");
        PreparedImage prep;
        if (const auto* f = c.find("file")) {
            std::ifstream pf(dir + f->as_string(), std::ios::binary);
            const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
            prep = prepare_bytes(bytes, cfg);
        } else {
            prep = prepare(picture(w, h), cfg);
        }
        std::vector<std::uint8_t> le(prep.patches.size() * 2U);
        for (std::size_t i = 0; i < prep.patches.size(); ++i) {
            le[2U * i] = static_cast<std::uint8_t>(prep.patches[i]);
            le[2U * i + 1U] = static_cast<std::uint8_t>(prep.patches[i] >> 8U);
        }
        const auto digest = strix::sha256_bytes_fast(std::as_bytes(std::span(le)));
        check(hex(digest.bytes) == c.find("patches_sha256")->as_string(), name + ": patches differ from the reference");
        check(prep.digest == prepared_digest(prep.plan, prep.patches), name + ": digest");
        ++n;
    }
    }
    // span layout: START, (CELL * w, NEWLINE) * h, END
    GridPlan g;
    g.llm_w = 3;
    g.llm_h = 2;
    const SpanKind want[] = {SpanKind::start, SpanKind::cell, SpanKind::cell, SpanKind::cell, SpanKind::newline,
                             SpanKind::cell,  SpanKind::cell, SpanKind::cell, SpanKind::newline, SpanKind::end};
    check(g.span_tokens() == 10U, "span tokens 3x2");
    for (std::uint32_t i = 0; i < 10U; ++i) {
        std::uint32_t cell = 99;
        const auto k = span_kind(g, i, &cell);
        check(k == want[i], "span kind " + std::to_string(i));
        if (k == SpanKind::cell) check(cell == (i < 5U ? i - 1U : i - 2U), "span cell " + std::to_string(i));
    }
    // bf16 rounding: ties to even, NaN stays NaN
    check(bf16_bits(1.0f) == 0x3F80U && bf16_bits(-2.0f) == 0xC000U, "bf16 exact");
    check(bf16_bits(bf16_value(0x3F80U) + std::ldexp(1.0f, -8)) == 0x3F80U, "bf16 tie to even (down)");
    check(bf16_bits(bf16_value(0x3F81U) + std::ldexp(1.0f, -8)) == 0x3F82U, "bf16 tie to even (up)");
    check((bf16_bits(std::numeric_limits<float>::quiet_NaN()) & 0x7FC0U) == 0x7FC0U, "bf16 NaN");
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << n << " pictures; vision prep tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}
