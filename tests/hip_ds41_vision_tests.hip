// DeepSeek V4.1 vision tower: the device tower equals the host reference bit for bit at every step (embedding, each
// block, final norm, aligner rows, span rows) on pictures whose grids exercise every GEMM tail; then the device time
// and memory of a full-size picture. Needs the vision GGUF and a GPU (explicit run).
//   strix-hip-ds41-vision-tests DeepSeek-V4.1-Flash-1017-vision-BF16.gguf [--quick]
#include "strix/hip_ds41_vision.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>

using namespace strix::deepseek41::vision;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

strix::image::Rgb8 picture(std::uint32_t w, std::uint32_t h, std::uint32_t seed) {
    strix::image::Rgb8 im;
    im.width = w;
    im.height = h;
    im.pixels.resize(std::size_t{w} * h * 3U);
    std::uint32_t s = seed * 2654435761U;
    for (std::uint32_t y = 0; y < h; ++y)
        for (std::uint32_t x = 0; x < w; ++x) {
            s = s * 1103515245U + 12345U;
            auto* p = im.pixels.data() + (std::size_t{y} * w + x) * 3U;
            p[0] = static_cast<std::uint8_t>((7U * x + 3U * y) ^ (s >> 24U));
            p[1] = static_cast<std::uint8_t>(((x * y + 17U * x) >> 3U) & 255U);
            p[2] = static_cast<std::uint8_t>((((x - w / 2U) * (x - w / 2U) + (y - h / 2U) * (y - h / 2U)) < (w * h / 9U)) ? 250U : (x ^ y) * 5U);
        }
    return im;
}

std::string key(const char* step, std::uint32_t layer) { return std::string(step) + "." + std::to_string(layer); }

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2) {
        std::cerr << "usage: strix-hip-ds41-vision-tests VISION.gguf [--quick]\n";
        return 2;
    }
    const bool quick = argc > 2 && std::string(argv[2]) == "--quick";
    if (!host_has_fma()) throw std::runtime_error("the host reference needs FMA/AVX2");
    const auto t0 = std::chrono::steady_clock::now();
    const auto w = load_vision_file(argv[1]);
    hipStream_t stream;
    if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) throw std::runtime_error("stream");
    std::size_t free0 = 0, total = 0, free1 = 0;
    (void)hipMemGetInfo(&free0, &total);
    DeviceTower tower(w, stream);
    (void)hipMemGetInfo(&free1, &total);
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("tower: weights %.1f MiB, work %.1f MiB (device free %.1f -> %.1f MiB), max %u patches / %u cells, load %.0f ms\n",
                static_cast<double>(tower.weight_bytes()) / 1048576.0, static_cast<double>(tower.work_bytes()) / 1048576.0,
                static_cast<double>(free0) / 1048576.0, static_cast<double>(free1) / 1048576.0, max_patches(w.config), max_cells(w.config),
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    check(max_patches(w.config) == 9189U && max_cells(w.config) == 1021U, "max grid");

    // bitwise comparison with the host reference; sizes chosen so that rows, outputs and K hit GEMM tails
    struct Case {
        std::uint32_t w, h;
    };
    const Case cases[] = {{64, 48}, {700, 250}};
    for (const auto& cs : cases) {
        const auto img = prepare(picture(cs.w, cs.h, cs.w), w.config);
        const auto& g = img.plan;
        std::printf("picture %ux%u: patches %ux%u (%u), cells %ux%u, span %u\n", cs.w, cs.h, g.vit_w, g.vit_h, g.patches(), g.llm_w, g.llm_h,
                    g.span_tokens());
        std::map<std::string, std::vector<std::uint16_t>> host_steps;
        const auto th0 = std::chrono::steady_clock::now();
        const auto host_rows = encode_reference(w, img, [&](const char* s, std::uint32_t l, std::span<const std::uint16_t> v) {
            if (!quick || l < 2U || std::string(s) != "block") host_steps[key(s, l)].assign(v.begin(), v.end());
        });
        const auto th1 = std::chrono::steady_clock::now();
        std::string first_bad;
        const auto dev_span = tower.encode_to_host(img, [&](const char* s, std::uint32_t l, std::span<const std::uint16_t> v) {
            const auto it = host_steps.find(key(s, l));
            if (it == host_steps.end()) return;
            std::size_t bad = 0, first = v.size();
            for (std::size_t i = 0; i < v.size(); ++i)
                if (v[i] != it->second[i]) {
                    if (first == v.size()) first = i;
                    ++bad;
                }
            if (bad && first_bad.empty()) {
                first_bad = key(s, l);
                std::printf("  first mismatch at step %s: %zu of %zu values differ, first at %zu (device %04x host %04x)\n", first_bad.c_str(), bad,
                            v.size(), first, v[first], it->second[first]);
            }
        });
        const auto th2 = std::chrono::steady_clock::now();
        check(first_bad.empty(), "device equals host at every step (" + std::to_string(cs.w) + "x" + std::to_string(cs.h) + ")");
        const auto host_span = span_rows(w, g, host_rows);
        check(dev_span == host_span, "span rows equal");
        std::printf("  host %.0f ms, device (with step downloads) %.0f ms, %s\n", std::chrono::duration<double, std::milli>(th1 - th0).count(),
                    std::chrono::duration<double, std::milli>(th2 - th1).count(), first_bad.empty() ? "bitwise equal" : "DIFFERENT");
    }

    // full-size picture: device time and determinism (two runs equal)
    const auto big = prepare(picture(1920, 1080, 7), w.config);
    std::printf("picture 1920x1080: patches %u, cells %u, span %u\n", big.plan.patches(), big.plan.cells(), big.plan.span_tokens());
    (void)tower.encode_to_host(big);  // warm-up
    const auto tb0 = std::chrono::steady_clock::now();
    const auto r1 = tower.encode_to_host(big);
    const auto tb1 = std::chrono::steady_clock::now();
    const auto r2 = tower.encode_to_host(big);
    check(r1 == r2, "device tower is deterministic");
    std::printf("  device encode %.0f ms\n", std::chrono::duration<double, std::milli>(tb1 - tb0).count());
    (void)hipStreamDestroy(stream);
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "hip ds41 vision tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}
