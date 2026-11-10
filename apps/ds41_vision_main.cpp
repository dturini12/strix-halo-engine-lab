// strix-ds41-vision: DeepSeek V4.1 image input tools.
//   strix-ds41-vision plan WIDTH HEIGHT                      grid plan and span size of a picture size
//   strix-ds41-vision encode --vision FILE --picture P [--out ROWS] [--compare ROWS ...] [--threads N]
//       host reference of the tower on one picture; writes the BF16 aligner rows and compares them with other
//       rows files (e.g. tools/ds41/vision_torch_reference.py output): max/mean |diff|, mean cosine per row.
#include "strix/ds41_vision.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using namespace strix::deepseek41::vision;

namespace {

std::vector<std::uint8_t> read_bytes(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + p);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void compare(const std::vector<std::uint16_t>& a, const std::string& path, std::size_t width) {
    const auto raw = read_bytes(path);
    if (raw.size() != a.size() * 2U) {
        std::cout << path << ": size differs (" << raw.size() / 2U << " vs " << a.size() << " values)\n";
        return;
    }
    double max_abs = 0, sum_abs = 0, cos_sum = 0, cos_min = 1;
    std::size_t equal = 0;
    const std::size_t rows = a.size() / width;
    for (std::size_t r = 0; r < rows; ++r) {
        double dot = 0, na = 0, nb = 0;
        for (std::size_t k = 0; k < width; ++k) {
            const std::size_t i = r * width + k;
            const auto bb = static_cast<std::uint16_t>(raw[2U * i] | (raw[2U * i + 1U] << 8U));
            const double x = bf16_float(a[i]), y = bf16_float(bb);
            if (a[i] == bb) ++equal;
            const double d = std::fabs(x - y);
            max_abs = std::max(max_abs, d);
            sum_abs += d;
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        const double c = dot / std::sqrt(na * nb);
        cos_sum += c;
        cos_min = std::min(cos_min, c);
    }
    std::printf("%s: max|d| %.4g  mean|d| %.4g  cosine mean %.6f min %.6f  equal bf16 %.2f%%\n", path.c_str(), max_abs,
                sum_abs / static_cast<double>(a.size()), cos_sum / static_cast<double>(rows), cos_min, 100.0 * static_cast<double>(equal) / static_cast<double>(a.size()));
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2) {
        std::cerr << "usage: strix-ds41-vision plan W H | encode --vision FILE --picture P [--out ROWS] [--compare ROWS ...]\n";
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "plan" && argc == 4) {
        const auto g = plan_grid(static_cast<std::uint32_t>(std::stoul(argv[2])), static_cast<std::uint32_t>(std::stoul(argv[3])), VisionConfig{});
        std::printf("best %ux%u  patches %ux%u (%u)  cells %ux%u (%u)  span %u tokens\n", g.best_w, g.best_h, g.vit_w, g.vit_h, g.patches(), g.llm_w,
                    g.llm_h, g.cells(), g.span_tokens());
        return 0;
    }
    if (cmd != "encode") throw std::runtime_error("unknown command " + cmd);
    std::string vision, picture, out;
    std::vector<std::string> cmp;
    unsigned threads = 0;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--vision") vision = next();
        else if (a == "--picture") picture = next();
        else if (a == "--out") out = next();
        else if (a == "--compare") cmp.push_back(next());
        else if (a == "--threads") threads = static_cast<unsigned>(std::stoul(next()));
        else throw std::runtime_error("unknown option " + a);
    }
    if (!host_has_fma()) throw std::runtime_error("the host reference needs a CPU with FMA and AVX2");
    const auto t0 = std::chrono::steady_clock::now();
    const auto w = load_vision_file(vision);
    const auto t1 = std::chrono::steady_clock::now();
    const auto img = prepare_bytes(read_bytes(picture), w.config);
    const auto& g = img.plan;
    std::printf("picture %s: patches %ux%u, cells %ux%u, span %u tokens\n", picture.c_str(), g.vit_w, g.vit_h, g.llm_w, g.llm_h, g.span_tokens());
    const auto t2 = std::chrono::steady_clock::now();
    const auto rows = encode_reference(w, img, {}, threads);
    const auto t3 = std::chrono::steady_clock::now();
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::printf("load %.0f ms, prepare %.0f ms, host reference %.0f ms\n", ms(t0, t1), ms(t1, t2), ms(t2, t3));
    if (!out.empty()) {
        std::ofstream o(out, std::ios::binary);
        for (const auto v : rows) {
            const char b[2] = {static_cast<char>(v & 0xFFU), static_cast<char>(v >> 8U)};
            o.write(b, 2);
        }
    }
    for (const auto& c : cmp) compare(rows, c, w.config.out_dim);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}
