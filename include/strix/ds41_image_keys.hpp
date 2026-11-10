#pragma once
// DeepSeek V4.1 image input inside the engine: every position of an image span carries an image key instead of the
// image token id (129264). The reference model reads that id only for the embedding (replaced by the span's rows) and
// as an Engram dead position, so a key changes nothing there; but keys differ between pictures, so token-keyed
// structures (prefix cache chains and block comparison, session histories, the pair's prompt transfer) tell pictures
// apart. A key has the top bit set (no vocabulary id does): key = 0x80000000 | (FNV-1a of the picture's prepared
// digest and the position's index in its span, low 31 bits). docs/models/deepseek-v4.1/vision.md, "Image keys".
#include <array>
#include <cstdint>
#include <span>

namespace strix::ds41 {

inline constexpr std::uint32_t kImageKeyBit = 0x80000000U;
[[nodiscard]] constexpr bool is_image_key(std::uint32_t token) noexcept { return (token & kImageKeyBit) != 0U; }

[[nodiscard]] inline std::uint32_t image_key(const std::array<std::uint8_t, 32>& digest, std::uint32_t index) noexcept {
    std::uint64_t h = 14695981039346656037ULL;
    for (const auto b : digest) h = (h ^ b) * 1099511628211ULL;
    for (int k = 0; k < 4; ++k) h = (h ^ ((index >> (8 * k)) & 0xFFU)) * 1099511628211ULL;
    return kImageKeyBit | static_cast<std::uint32_t>((h ^ (h >> 31U)) & 0x7FFFFFFFU);
}

// The prompt's image spans as keys, in place.
inline void key_image_span(std::span<std::uint32_t> prompt, std::uint32_t start, std::uint32_t count, const std::array<std::uint8_t, 32>& digest) {
    for (std::uint32_t i = 0; i < count; ++i) prompt[start + i] = image_key(digest, i);
}

}  // namespace strix::ds41
