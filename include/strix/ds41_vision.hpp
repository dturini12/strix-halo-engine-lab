#pragma once
// DeepSeek V4.1 image input, host side: the vision file's configuration and weights, the reference image
// preprocessing (inference/image_processor.py: grid plan, Pillow BICUBIC resize + ImageOps.pad, normalised BF16
// patches), the image span layout in the token stream, and the host reference of the ViT + aligner with its
// declared arithmetic. docs/models/deepseek-v4.1/vision.md.
#include "strix/image/image_decode.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace strix::deepseek41::vision {

// deepseek4-vision GGUF metadata (vision_config of the release's config.json).
struct VisionConfig {
    std::uint32_t layers = 32, dim = 1024, heads = 16, inter = 2816, patch = 14, merge = 3, out_dim = 5120;
    std::uint32_t max_tokens = 1024, min_pixels = 295936, image_token_id = 129264;
    float theta = 10000.0f, eps = 1e-6f;
    std::optional<double> max_wh_ratio;  // null in the release
    [[nodiscard]] std::uint32_t head_dim() const noexcept { return dim / heads; }
    [[nodiscard]] std::uint32_t patch_in() const noexcept { return 3U * patch * patch; }
    [[nodiscard]] std::uint32_t merged_in() const noexcept { return dim * merge * merge; }
};

// plan_image_grid: the pixel size the picture is padded to, its ViT patch grid and the aligner's token grid.
struct GridPlan {
    std::uint32_t best_w = 0, best_h = 0, vit_w = 0, vit_h = 0, llm_w = 0, llm_h = 0;
    [[nodiscard]] std::uint32_t patches() const noexcept { return vit_w * vit_h; }
    [[nodiscard]] std::uint32_t cells() const noexcept { return llm_w * llm_h; }
    // [IMAGE_START] + ([IMAGE] * llm_w + [IMAGE_NEW_LINE]) * llm_h + [IMAGE_END]
    [[nodiscard]] std::uint32_t span_tokens() const noexcept { return llm_h * (llm_w + 1U) + 2U; }
    friend bool operator==(const GridPlan&, const GridPlan&) = default;
};

[[nodiscard]] GridPlan plan_grid(std::uint32_t width, std::uint32_t height, const VisionConfig& config);

// Pillow 11.3 Image.resize(size, BICUBIC) on an RGB image (8-bit fixed-point two-pass resample).
[[nodiscard]] image::Rgb8 resize_bicubic(const image::Rgb8& in, std::uint32_t width, std::uint32_t height);
// ImageOps.pad(image, (width, height), color=(fill, fill, fill)): contain() then centre on the fill colour.
[[nodiscard]] image::Rgb8 pad(const image::Rgb8& in, std::uint32_t width, std::uint32_t height, std::uint8_t fill = 127);

// One picture ready for the tower: its plan, the BF16 patches [vit_h * vit_w][3][patch][patch] in reading order
// ((v / 255 - 0.5) / 0.5 in FP32, rounded to BF16), and the SHA-256 of both (the picture's identity).
struct PreparedImage {
    GridPlan plan;
    std::vector<std::uint16_t> patches;
    std::array<std::uint8_t, 32> digest{};
};

[[nodiscard]] PreparedImage prepare(const image::Rgb8& picture, const VisionConfig& config);
// Decode (PNG/JPEG) and prepare; throws image::ImageError with a client-facing message.
[[nodiscard]] PreparedImage prepare_bytes(std::span<const std::uint8_t> encoded, const VisionConfig& config,
                                          const image::DecodeLimits& limits = {});
[[nodiscard]] std::array<std::uint8_t, 32> prepared_digest(const GridPlan& plan, std::span<const std::uint16_t> patches);

// Span position kinds (image_processor.py: IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END).
enum class SpanKind : std::uint8_t { start = 0, cell = 1, newline = 2, end = 3 };
[[nodiscard]] SpanKind span_kind(const GridPlan& plan, std::uint32_t index, std::uint32_t* cell = nullptr) noexcept;

// --- bf16 helpers (round to nearest even, as torch's .to(torch.bfloat16)) ---
[[nodiscard]] std::uint16_t bf16_bits(float v) noexcept;
[[nodiscard]] float bf16_value(std::uint16_t bits) noexcept;

}  // namespace strix::deepseek41::vision
