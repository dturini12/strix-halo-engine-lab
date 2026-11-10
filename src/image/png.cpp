// PNG to RGB8 as Pillow 11.3's PngImagePlugin + convert("RGB"): every colour type and bit depth, Adam7, CRCs
// checked; alpha and tRNS are ignored (convert("RGB") drops them), 16-bit colour samples keep the high byte, grey of
// depth 1/2/4 is scaled by 255/85/17, 16-bit grey (Pillow "I;16") is clipped at 255, palette indices past PLTE
// read black.
#include "strix/image/image_decode.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strix::image {
namespace {

[[noreturn]] void fail(const std::string& what) { throw ImageError("PNG: " + what); }

std::uint32_t be32(const std::uint8_t* p) {
    return (std::uint32_t{p[0]} << 24U) | (std::uint32_t{p[1]} << 16U) | (std::uint32_t{p[2]} << 8U) | p[3];
}

std::uint32_t crc32(const std::uint8_t* p, std::size_t n, std::uint32_t crc = 0xFFFFFFFFU) {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256U; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1U) != 0U ? 0xEDB88320U ^ (c >> 1U) : c >> 1U;
            t[i] = c;
        }
        return t;
    }();
    for (std::size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xFFU] ^ (crc >> 8U);
    return crc;
}

std::uint8_t paeth(int a, int b, int c) {
    const int p = a + b - c;
    const int pa = p > a ? p - a : a - p, pb = p > b ? p - b : b - p, pc = p > c ? p - c : c - p;
    if (pa <= pb && pa <= pc) return static_cast<std::uint8_t>(a);
    if (pb <= pc) return static_cast<std::uint8_t>(b);
    return static_cast<std::uint8_t>(c);
}

struct Header {
    std::uint32_t width = 0, height = 0;
    unsigned depth = 0, type = 0, interlace = 0;
    unsigned channels() const { return type == 0 ? 1U : type == 2 ? 3U : type == 3 ? 1U : type == 4 ? 2U : 4U; }
    std::size_t bits_per_pixel() const { return std::size_t{channels()} * depth; }
    std::size_t row_bytes(std::uint32_t w) const { return (static_cast<std::size_t>(w) * bits_per_pixel() + 7U) / 8U; }
};

// Unfilter one pass in place: `data` holds height rows of (1 + row_bytes) bytes; returns the filtered-out rows.
void unfilter(std::uint8_t* data, std::size_t rows, std::size_t row_bytes, std::size_t bpp_bytes, std::vector<std::uint8_t>& out) {
    out.assign(rows * row_bytes, 0);
    const std::uint8_t* prev = nullptr;
    for (std::size_t y = 0; y < rows; ++y) {
        const std::uint8_t* src = data + y * (row_bytes + 1U);
        const unsigned filter = src[0];
        ++src;
        std::uint8_t* dst = out.data() + y * row_bytes;
        for (std::size_t x = 0; x < row_bytes; ++x) {
            const int a = x >= bpp_bytes ? dst[x - bpp_bytes] : 0;
            const int b = prev ? prev[x] : 0;
            const int c = prev && x >= bpp_bytes ? prev[x - bpp_bytes] : 0;
            int v = src[x];
            switch (filter) {
                case 0: break;
                case 1: v += a; break;
                case 2: v += b; break;
                case 3: v += (a + b) >> 1; break;
                case 4: v += paeth(a, b, c); break;
                default: fail("unknown filter type " + std::to_string(filter));
            }
            dst[x] = static_cast<std::uint8_t>(v);
        }
        prev = dst;
    }
}

// Sample k of a row of `depth`-bit samples, as Pillow scales it to 8 bits for grey ("1", "L;2", "L;4", "L", I;16 -> RGB).
std::uint8_t sample8(const std::uint8_t* row, std::size_t k, unsigned depth, bool grey) {
    switch (depth) {
        case 1: return ((row[k / 8U] >> (7U - k % 8U)) & 1U) != 0U ? (grey ? 255 : 1) : 0;
        case 2: {
            const auto v = static_cast<unsigned>((row[k / 4U] >> (6U - 2U * (k % 4U))) & 3U);
            return static_cast<std::uint8_t>(grey ? v * 85U : v);
        }
        case 4: {
            const auto v = static_cast<unsigned>((row[k / 2U] >> (4U - 4U * (k % 2U))) & 15U);
            return static_cast<std::uint8_t>(grey ? v * 17U : v);
        }
        case 8: return row[k];
        default: return row[2U * k];  // 16-bit: high byte
    }
}

}  // namespace

Rgb8 decode_png(std::span<const std::uint8_t> bytes, const DecodeLimits& limits) {
    static constexpr std::uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (bytes.size() < 8U || std::memcmp(bytes.data(), sig, 8) != 0) fail("not a PNG file");
    Header h;
    bool have_header = false, have_end = false;
    std::vector<std::uint8_t> idat;
    std::array<std::uint8_t, 768> palette{};  // past PLTE: black
    std::size_t at = 8;
    while (at < bytes.size()) {
        if (bytes.size() - at < 12U) fail("truncated chunk");
        const std::uint32_t len = be32(bytes.data() + at);
        if (len > bytes.size() - at - 12U) fail("truncated chunk");
        const std::uint8_t* type = bytes.data() + at + 4U;
        const std::uint8_t* data = type + 4;
        const std::uint32_t crc = be32(data + len);
        if ((crc32(type, std::size_t{len} + 4U) ^ 0xFFFFFFFFU) != crc) fail("broken PNG file (CRC mismatch)");
        const std::string name(reinterpret_cast<const char*>(type), 4);
        at += std::size_t{len} + 12U;
        if (name == "IHDR") {
            if (len != 13U || have_header) fail("bad IHDR");
            h.width = be32(data);
            h.height = be32(data + 4);
            h.depth = data[8];
            h.type = data[9];
            h.interlace = data[12];
            if (data[10] != 0U || data[11] != 0U || h.interlace > 1U) fail("unsupported compression, filter or interlace method");
            const bool ok = (h.type == 0 && (h.depth == 1 || h.depth == 2 || h.depth == 4 || h.depth == 8 || h.depth == 16)) ||
                            (h.type == 3 && (h.depth == 1 || h.depth == 2 || h.depth == 4 || h.depth == 8)) ||
                            ((h.type == 2 || h.type == 4 || h.type == 6) && (h.depth == 8 || h.depth == 16));
            if (!ok) fail("unsupported colour type " + std::to_string(h.type) + " at bit depth " + std::to_string(h.depth));
            if (h.width == 0U || h.height == 0U) fail("zero image size");
            if (std::uint64_t{h.width} * h.height > limits.max_pixels)
                throw ImageError("image input: Image size (" + std::to_string(std::uint64_t{h.width} * h.height) +
                                 " pixels) exceeds limit of " + std::to_string(limits.max_pixels) +
                                 " pixels, could be decompression bomb DOS attack.");
            have_header = true;
        } else if (!have_header) {
            fail("first chunk is not IHDR");
        } else if (name == "PLTE") {
            if (len % 3U != 0U || len > 768U) fail("bad PLTE");
            std::memcpy(palette.data(), data, len);
        } else if (name == "IDAT") {
            idat.insert(idat.end(), data, data + len);
        } else if (name == "IEND") {
            have_end = true;
            break;
        } else if ((type[0] & 0x20U) == 0U) {
            fail("unknown critical chunk " + name);
        }
    }
    if (!have_header || idat.empty()) fail("no image data");
    (void)have_end;  // Pillow decodes a file without IEND once the data is complete

    // pass geometry (Adam7 or one pass)
    struct Pass {
        unsigned x0, y0, dx, dy;
    };
    static constexpr Pass adam7[7] = {{0, 0, 8, 8}, {4, 0, 8, 8}, {0, 4, 4, 8}, {2, 0, 4, 4}, {0, 2, 2, 4}, {1, 0, 2, 2}, {0, 1, 1, 2}};
    static constexpr Pass whole[1] = {{0, 0, 1, 1}};
    const Pass* passes = h.interlace ? adam7 : whole;
    const int npass = h.interlace ? 7 : 1;
    const std::size_t bpp_bytes = std::max<std::size_t>(1U, h.bits_per_pixel() / 8U);
    std::size_t total = 0;
    for (int p = 0; p < npass; ++p) {
        const auto& ps = passes[p];
        const std::uint32_t pw = h.width > ps.x0 ? (h.width - ps.x0 + ps.dx - 1U) / ps.dx : 0U;
        const std::uint32_t ph = h.height > ps.y0 ? (h.height - ps.y0 + ps.dy - 1U) / ps.dy : 0U;
        if (pw && ph) total += std::size_t{ph} * (h.row_bytes(pw) + 1U);
    }
    auto raw = zlib_inflate(idat, total);
    if (raw.size() < total) fail("image data is truncated");

    Rgb8 img;
    img.width = h.width;
    img.height = h.height;
    img.pixels.assign(std::size_t{h.width} * h.height * 3U, 0);
    std::vector<std::uint8_t> rows;
    std::size_t offset = 0;
    const bool grey = h.type == 0 || h.type == 4;
    for (int p = 0; p < npass; ++p) {
        const auto& ps = passes[p];
        const std::uint32_t pw = h.width > ps.x0 ? (h.width - ps.x0 + ps.dx - 1U) / ps.dx : 0U;
        const std::uint32_t ph = h.height > ps.y0 ? (h.height - ps.y0 + ps.dy - 1U) / ps.dy : 0U;
        if (!pw || !ph) continue;
        const std::size_t rb = h.row_bytes(pw);
        unfilter(raw.data() + offset, ph, rb, bpp_bytes, rows);
        offset += std::size_t{ph} * (rb + 1U);
        for (std::uint32_t y = 0; y < ph; ++y) {
            const std::uint8_t* row = rows.data() + std::size_t{y} * rb;
            std::uint8_t* out = img.pixels.data() + (std::size_t{ps.y0 + y * ps.dy} * h.width) * 3U;
            for (std::uint32_t x = 0; x < pw; ++x) {
                std::uint8_t* px = out + std::size_t{ps.x0 + x * ps.dx} * 3U;
                const std::size_t ch = h.channels();
                if (h.type == 3) {
                    const unsigned idx = sample8(row, x, h.depth, false);
                    px[0] = palette[idx * 3U];
                    px[1] = palette[idx * 3U + 1U];
                    px[2] = palette[idx * 3U + 2U];
                } else if (h.type == 0 && h.depth == 16) {  // Pillow "I;16" -> RGB clips at 255
                    const unsigned v16 = (unsigned{row[2U * x]} << 8U) | row[2U * x + 1U];
                    px[0] = px[1] = px[2] = static_cast<std::uint8_t>(v16 > 255U ? 255U : v16);
                } else if (grey) {
                    const auto v = sample8(row, std::size_t{x} * ch, h.depth, true);
                    px[0] = px[1] = px[2] = v;
                } else {
                    for (unsigned c = 0; c < 3U; ++c) px[c] = sample8(row, std::size_t{x} * ch + c, h.depth, false);
                }
            }
        }
    }
    return img;
}

}  // namespace strix::image
