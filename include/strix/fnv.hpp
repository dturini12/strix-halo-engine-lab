#pragma once

#include <cstdint>
#include <string_view>

// Host identity hashing. Existing formats must pass their historical seed.
namespace strix::host_fnv {
inline constexpr std::uint64_t offset64 = 14695981039346656037ULL;
inline constexpr std::uint64_t prime64 = 1099511628211ULL;
inline constexpr std::uint64_t byte(std::uint64_t hash, unsigned char value) {
    return (hash ^ value) * prime64;
}
inline constexpr std::uint64_t text(std::string_view value, std::uint64_t seed = offset64) {
    for (const char c : value) seed = byte(seed, static_cast<unsigned char>(c));
    return seed;
}
// Length framing makes compositional fields unambiguous, including embedded NULs.
inline constexpr std::uint64_t field(std::uint64_t hash, std::string_view value) {
    auto size = static_cast<std::uint64_t>(value.size());
    for (unsigned i = 0; i < 8; ++i) {
        hash = byte(hash, static_cast<unsigned char>(size & 255U));
        size >>= 8;
    }
    return text(value, hash);
}
}  // namespace strix::host_fnv
