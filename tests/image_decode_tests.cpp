// The in-house PNG / JPEG decoders equal Pillow 11.3's Image.open(...).convert("RGB") byte for byte on the recorded
// pictures (tools/image/make_decode_fixtures.py), plus base64, data URLs and the format refusals.
//   strix-image-decode-tests tests/data/image [EXTRA.json]
#include "strix/image/image_decode.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/serve/json.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string hex(const strix::ModelContentIdentity& id) {
    std::string out;
    char b[3];
    for (const auto v : id.bytes) {
        std::snprintf(b, sizeof b, "%02x", v);
        out += b;
    }
    return out;
}

void run_cases(const std::filesystem::path& dir, const std::filesystem::path& json) {
    const auto text = read_file(json);
    const auto doc = strix::serve::parse_json(std::string(text.begin(), text.end()));
    std::size_t n = 0;
    for (const auto& c : doc.find("cases")->as_array()) {
        const auto name = c.find("file")->as_string();
        const auto bytes = read_file(dir / name);
        try {
            const auto img = strix::image::decode_image(bytes);
            const auto digest = hex(strix::sha256_bytes_fast(std::as_bytes(std::span(img.pixels))));
            check(img.width == static_cast<std::uint32_t>(c.find("width")->as_int()) &&
                      img.height == static_cast<std::uint32_t>(c.find("height")->as_int()),
                  name + ": size");
            check(digest == c.find("rgb_sha256")->as_string(), name + ": RGB differs from Pillow");
        } catch (const std::exception& e) {
            check(false, name + ": " + e.what());
        }
        ++n;
    }
    std::cout << json.filename().string() << ": " << n << " pictures\n";
}

}  // namespace

int main(int argc, char** argv) try {
    using namespace strix::image;
    if (argc < 2) {
        std::cerr << "usage: strix-image-decode-tests DIR [EXTRA_DIR]\n";
        return 2;
    }
    run_cases(argv[1], std::filesystem::path(argv[1]) / "decode_cases.json");
    if (argc > 2) run_cases(argv[2], std::filesystem::path(argv[2]) / "decode_cases.json");

    // base64 and data URLs
    const auto b = base64_decode("aGVs\nbG8h");
    check(std::string(b.begin(), b.end()) == "hello!", "base64 with a newline");
    const auto p1 = base64_decode("aGk="), p2 = base64_decode("aGk");
    check(std::string(p1.begin(), p1.end()) == "hi" && p1 == p2, "base64 padding optional");
    bool threw = false;
    try {
        (void)base64_decode("aGVsb");
    } catch (const ImageError&) {
        threw = true;
    }
    check(threw, "base64 dangling character");
    const auto d = data_url_bytes("data:image/png;base64,aGk=");
    check(std::string(d.begin(), d.end()) == "hi", "data URL");
    for (const char* bad : {"https://example.com/x.png", "data:image/png,hi", "data:image/png;base64"}) {
        threw = false;
        try {
            (void)data_url_bytes(bad);
        } catch (const ImageError&) {
            threw = true;
        }
        check(threw, std::string("refused URL ") + bad);
    }
    // format refusals name the format
    const std::uint8_t gif[] = {'G', 'I', 'F', '8', '9', 'a', 0, 0};
    try {
        (void)decode_image(gif);
        check(false, "GIF accepted");
    } catch (const ImageError& e) {
        check(std::string(e.what()).find("GIF") != std::string::npos, "GIF refusal names the format");
    }
    const std::uint8_t junk[] = {1, 2, 3, 4};
    try {
        (void)decode_image(junk);
        check(false, "junk accepted");
    } catch (const ImageError&) {
    }
    // a decompression bomb is refused before decoding
    {
        auto png = read_file(std::filesystem::path(argv[1]) / "tiny_1x1.png");
        DecodeLimits small;
        small.max_pixels = 0;
        try {
            (void)decode_png(png, small);
            check(false, "bomb limit");
        } catch (const ImageError& e) {
            check(std::string(e.what()).find("decompression bomb") != std::string::npos, "bomb message");
        }
    }
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "image decode tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}
