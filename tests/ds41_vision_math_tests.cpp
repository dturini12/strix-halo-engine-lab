// DeepSeek V4.1 vision tower element-wise arithmetic: declared_erf against double-precision erf, GELU / SiLU forms,
// the rotary tables and the span rows layout.
#include "strix/ds41_vision.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}
}  // namespace

int main() {
    using namespace strix::deepseek41::vision;
    // erf: relative error against the double-precision erf on a dense grid
    double worst = 0, worst_x = 0;
    for (int i = -800000; i <= 800000; ++i) {
        const float x = static_cast<float>(i) * 1e-5F;
        const double want = std::erf(static_cast<double>(x));
        const double got = declared_erf(x);
        const double err = std::fabs(got - want) / std::max(std::fabs(want), 1e-30);
        if (err > worst) {
            worst = err;
            worst_x = x;
        }
    }
    std::cout << "declared_erf worst relative error " << worst << " at " << worst_x << '\n';
    check(worst < 4e-7, "declared_erf accuracy");
    check(declared_erf(0.0F) == 0.0F && declared_erf(10.0F) == 1.0F && declared_erf(-10.0F) == -1.0F, "erf limits");
    check(std::isnan(declared_erf(std::nanf(""))), "erf NaN");
    check(declared_erf(-0.3F) == -declared_erf(0.3F), "erf odd");
    // GELU and SiLU against double-precision forms
    double gw = 0, sw = 0;
    for (int i = -60000; i <= 60000; ++i) {
        const float x = static_cast<float>(i) * 1e-4F;
        const double xd = x;
        const double g = 0.5 * xd * (1.0 + std::erf(xd / std::sqrt(2.0))), s = xd / (1.0 + std::exp(-xd));
        gw = std::max(gw, std::fabs(vision_gelu(x) - g) / std::max(std::fabs(xd), 1.0));  // 1 + erf(x) cancels for x << 0, as in torch
        sw = std::max(sw, std::fabs(vision_silu(x) - s) / std::max(std::fabs(s), 1e-6));
    }
    std::cout << "gelu worst error / max(|x|, 1) " << gw << ", silu " << sw << '\n';
    check(gw < 2e-6 && sw < 2e-6, "gelu/silu accuracy");
    // rotary tables: row-major patches, first quarter by row, second by column
    GridPlan g;
    g.vit_w = 5;
    g.vit_h = 3;
    std::vector<float> c, s;
    rope_tables(g, VisionConfig{}, c, s);
    check(c.size() == 15U * 32U, "rope size");
    const std::size_t r = 2U * 5U + 4U;  // row 2, column 4
    check(c[r * 32U + 0U] == static_cast<float>(std::cos(2.0)) && c[r * 32U + 16U] == static_cast<float>(std::cos(4.0)), "rope j=0 uses row, j=16 column");
    check(s[r * 32U + 1U] == static_cast<float>(std::sin(static_cast<double>(2.0F * static_cast<float>(1.0 / std::pow(10000.0, 2.0 / 32.0))))), "rope f_1");
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "vision math tests passed\n";
    return 0;
}
