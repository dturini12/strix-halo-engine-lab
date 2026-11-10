#pragma once
// Picture decoding for image input: PNG and baseline/progressive JPEG to 8-bit RGB, written for this engine
// (no third-party decoder). Each decoder equals what the model release's reference preprocessing gets from
// Pillow 11.3 / libjpeg-turbo 3.1 with `Image.open(...).convert("RGB")`: PNG alpha is dropped (not composited),
// 16-bit colour samples keep their high byte (16-bit grey is clipped at 255), low-depth grey is scaled to 0..255, palette entries past PLTE are black;
// JPEG uses libjpeg's integer "islow" IDCT, its fancy (triangle) upsampling for 2x1, 1x2 and 2x2 chroma, and its
// fixed-point YCbCr -> RGB tables. No EXIF orientation is applied (the reference does not apply one).
// docs/models/deepseek-v4.1/vision.md, "Pictures".
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::image {

class ImageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// 8-bit RGB, rows top to bottom, pixels interleaved R, G, B.
struct Rgb8 {
    std::uint32_t width = 0, height = 0;
    std::vector<std::uint8_t> pixels;
};

struct DecodeLimits {
    // Pillow's decompression-bomb error threshold (2 x MAX_IMAGE_PIXELS).
    std::uint64_t max_pixels = 178956970ULL;
    std::uint64_t max_input_bytes = 64ULL << 20U;
};

enum class Format : std::uint8_t { unknown, png, jpeg, gif, webp, bmp };

[[nodiscard]] Format sniff(std::span<const std::uint8_t> bytes) noexcept;
[[nodiscard]] const char* format_name(Format format) noexcept;

[[nodiscard]] Rgb8 decode_png(std::span<const std::uint8_t> bytes, const DecodeLimits& limits = {});
[[nodiscard]] Rgb8 decode_jpeg(std::span<const std::uint8_t> bytes, const DecodeLimits& limits = {});
// PNG or JPEG by signature; any other format throws ImageError naming it.
[[nodiscard]] Rgb8 decode_image(std::span<const std::uint8_t> bytes, const DecodeLimits& limits = {});

// RFC 1950 zlib stream (header, deflate data, Adler-32 checked); output capped at max_output bytes.
[[nodiscard]] std::vector<std::uint8_t> zlib_inflate(std::span<const std::uint8_t> stream, std::size_t max_output);

// Standard base64 (RFC 4648 alphabet). Like Python's base64.b64decode(validate=False): characters outside the
// alphabet are skipped; padding may be omitted. Throws ImageError on a dangling single character.
[[nodiscard]] std::vector<std::uint8_t> base64_decode(std::string_view text);

// The payload of a `data:[<media type>][;base64],<data>` URL. Only base64 data URLs are accepted.
[[nodiscard]] std::vector<std::uint8_t> data_url_bytes(std::string_view url);

}  // namespace strix::image
