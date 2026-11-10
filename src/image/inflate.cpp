// RFC 1950 / RFC 1951 decoder (zlib wrapper around deflate) for PNG image data.
#include "strix/image/image_decode.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace strix::image {
namespace {

[[noreturn]] void fail(const char* what) { throw ImageError(std::string("PNG image data: ") + what); }

struct Bits {
    std::span<const std::uint8_t> in;
    std::size_t at = 0;
    std::uint64_t buffer = 0;
    unsigned count = 0;

    void need(unsigned n) {
        while (count < n) {
            if (at >= in.size()) fail("truncated deflate stream");
            buffer |= std::uint64_t{in[at++]} << count;
            count += 8U;
        }
    }
    std::uint32_t take(unsigned n) {
        if (n == 0U) return 0U;
        need(n);
        const auto v = static_cast<std::uint32_t>(buffer & ((std::uint64_t{1} << n) - 1U));
        buffer >>= n;
        count -= n;
        return v;
    }
    void align() {
        const unsigned drop = count % 8U;
        buffer >>= drop;
        count -= drop;
    }
};

// Canonical Huffman decoder (counts per length, symbols sorted by code), as in zlib's puff.c.
struct Huffman {
    std::array<std::uint16_t, 16> count{};
    std::array<std::uint16_t, 320> symbol{};

    void build(const std::uint8_t* lengths, unsigned n) {
        count.fill(0);
        for (unsigned s = 0; s < n; ++s) ++count[lengths[s]];
        if (count[0] == n) return;  // no codes: a decode attempt fails
        int left = 1;
        for (unsigned len = 1; len < 16U; ++len) {
            left <<= 1;
            left -= count[len];
            if (left < 0) fail("over-subscribed Huffman code");
        }
        std::array<std::uint16_t, 16> offs{};
        for (unsigned len = 1; len < 15U; ++len) offs[len + 1U] = static_cast<std::uint16_t>(offs[len] + count[len]);
        for (unsigned s = 0; s < n; ++s)
            if (lengths[s] != 0U) symbol[offs[lengths[s]]++] = static_cast<std::uint16_t>(s);
    }
    int decode(Bits& b) const {
        int code = 0, first = 0, index = 0;
        for (unsigned len = 1; len < 16U; ++len) {
            code |= static_cast<int>(b.take(1));
            const int c = count[len];
            if (code - c < first) return symbol[static_cast<std::size_t>(index + (code - first))];
            index += c;
            first += c;
            first <<= 1;
            code <<= 1;
        }
        fail("invalid Huffman code");
    }
};

constexpr std::array<std::uint16_t, 29> kLenBase{3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr std::array<std::uint8_t, 29> kLenExtra{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr std::array<std::uint16_t, 30> kDistBase{1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr std::array<std::uint8_t, 30> kDistExtra{0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

void codes(Bits& b, std::vector<std::uint8_t>& out, const Huffman& lit, const Huffman& dist, std::size_t max_output) {
    for (;;) {
        const int sym = lit.decode(b);
        if (sym < 256) {
            if (out.size() >= max_output) fail("decompressed data larger than the image");
            out.push_back(static_cast<std::uint8_t>(sym));
            continue;
        }
        if (sym == 256) return;
        const auto li = static_cast<std::size_t>(sym - 257);
        if (li >= kLenBase.size()) fail("invalid length symbol");
        const std::size_t len = kLenBase[li] + b.take(kLenExtra[li]);
        const int ds = dist.decode(b);
        if (ds < 0 || static_cast<std::size_t>(ds) >= kDistBase.size()) fail("invalid distance symbol");
        const std::size_t d = kDistBase[static_cast<std::size_t>(ds)] + b.take(kDistExtra[static_cast<std::size_t>(ds)]);
        if (d > out.size()) fail("distance too far back");
        if (out.size() + len > max_output) fail("decompressed data larger than the image");
        const std::size_t from = out.size() - d;
        for (std::size_t k = 0; k < len; ++k) out.push_back(out[from + k]);
    }
}

}  // namespace

std::vector<std::uint8_t> zlib_inflate(std::span<const std::uint8_t> stream, std::size_t max_output) {
    if (stream.size() < 6U) fail("zlib stream too short");
    const unsigned cmf = stream[0], flg = stream[1];
    if ((cmf & 0x0FU) != 8U || (cmf >> 4U) > 7U) fail("not a deflate zlib stream");
    if (((cmf << 8U) | flg) % 31U != 0U) fail("bad zlib header check");
    if ((flg & 0x20U) != 0U) fail("zlib preset dictionary");
    Bits b{stream.subspan(2), 0, 0, 0};
    std::vector<std::uint8_t> out;
    out.reserve(std::min<std::size_t>(max_output, stream.size() * 4U));
    Huffman fixed_lit, fixed_dist;
    {
        std::array<std::uint8_t, 288> l{};
        for (unsigned s = 0; s < 288U; ++s) l[s] = s < 144U ? 8 : s < 256U ? 9 : s < 280U ? 7 : 8;
        fixed_lit.build(l.data(), 288);
        std::array<std::uint8_t, 30> d{};
        d.fill(5);
        fixed_dist.build(d.data(), 30);
    }
    for (bool last = false; !last;) {
        last = b.take(1) != 0U;
        const auto type = b.take(2);
        if (type == 0U) {
            b.align();
            const auto len = b.take(16), nlen = b.take(16);
            if ((len ^ 0xFFFFU) != nlen) fail("stored block length check");
            // bytes still in the bit buffer come first (count is a multiple of 8 here)
            for (std::uint32_t k = 0; k < len; ++k) {
                if (out.size() >= max_output) fail("decompressed data larger than the image");
                out.push_back(static_cast<std::uint8_t>(b.take(8)));
            }
        } else if (type == 1U) {
            codes(b, out, fixed_lit, fixed_dist, max_output);
        } else if (type == 2U) {
            const unsigned nlen = b.take(5) + 257U, ndist = b.take(5) + 1U, ncode = b.take(4) + 4U;
            if (nlen > 286U || ndist > 30U) fail("bad dynamic block counts");
            static constexpr std::array<std::uint8_t, 19> order{16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            std::array<std::uint8_t, 320> lengths{};
            for (unsigned k = 0; k < ncode; ++k) lengths[order[k]] = static_cast<std::uint8_t>(b.take(3));
            Huffman lencode;
            lencode.build(lengths.data(), 19);
            lengths.fill(0);
            for (unsigned k = 0; k < nlen + ndist;) {
                const int sym = lencode.decode(b);
                if (sym < 16) {
                    lengths[k++] = static_cast<std::uint8_t>(sym);
                    continue;
                }
                std::uint8_t value = 0;
                unsigned rep = 0;
                if (sym == 16) {
                    if (k == 0U) fail("repeat with no previous length");
                    value = lengths[k - 1U];
                    rep = 3U + b.take(2);
                } else if (sym == 17) {
                    rep = 3U + b.take(3);
                } else {
                    rep = 11U + b.take(7);
                }
                if (k + rep > nlen + ndist) fail("too many code lengths");
                while (rep-- != 0U) lengths[k++] = value;
            }
            if (lengths[256] == 0U) fail("no end-of-block code");
            Huffman lit, dist;
            lit.build(lengths.data(), nlen);
            dist.build(lengths.data() + nlen, ndist);
            codes(b, out, lit, dist, max_output);
        } else {
            fail("invalid deflate block type");
        }
    }
    b.align();
    std::uint32_t adler = 0;
    for (int k = 0; k < 4; ++k) adler = (adler << 8U) | b.take(8);
    std::uint32_t a = 1, s = 0;
    for (const auto v : out) {
        a = (a + v) % 65521U;
        s = (s + a) % 65521U;
    }
    if (((s << 16U) | a) != adler) fail("zlib Adler-32 mismatch");
    return out;
}

std::vector<std::uint8_t> base64_decode(std::string_view text) {
    std::vector<std::uint8_t> out;
    out.reserve(text.size() / 4U * 3U + 3U);
    std::uint32_t acc = 0;
    unsigned n = 0;
    for (const char ch : text) {
        int v = -1;
        if (ch >= 'A' && ch <= 'Z') v = ch - 'A';
        else if (ch >= 'a' && ch <= 'z') v = ch - 'a' + 26;
        else if (ch >= '0' && ch <= '9') v = ch - '0' + 52;
        else if (ch == '+') v = 62;
        else if (ch == '/') v = 63;
        else if (ch == '=') break;  // padding ends the data
        if (v < 0) continue;
        acc = (acc << 6U) | static_cast<std::uint32_t>(v);
        if (++n == 4U) {
            out.push_back(static_cast<std::uint8_t>(acc >> 16U));
            out.push_back(static_cast<std::uint8_t>(acc >> 8U));
            out.push_back(static_cast<std::uint8_t>(acc));
            acc = 0;
            n = 0;
        }
    }
    if (n == 1U) throw ImageError("Invalid base64-encoded string: number of data characters cannot be 1 more than a multiple of 4");
    if (n == 2U) out.push_back(static_cast<std::uint8_t>(acc >> 4U));
    if (n == 3U) {
        out.push_back(static_cast<std::uint8_t>(acc >> 10U));
        out.push_back(static_cast<std::uint8_t>(acc >> 2U));
    }
    return out;
}

std::vector<std::uint8_t> data_url_bytes(std::string_view url) {
    if (url.substr(0, 5) != "data:") throw ImageError("image input: image_url must be a base64 data: URL (data:image/png;base64,...)");
    const auto comma = url.find(',');
    if (comma == std::string_view::npos) throw ImageError("image input: malformed data URL");
    const auto header = url.substr(5, comma - 5U);
    if (header.find(";base64") == std::string_view::npos) throw ImageError("image input: image data URLs must be base64");
    return base64_decode(url.substr(comma + 1U));
}

Format sniff(std::span<const std::uint8_t> b) noexcept {
    const auto starts = [&](std::initializer_list<int> sig) {
        if (b.size() < sig.size()) return false;
        std::size_t k = 0;
        for (const int v : sig)
            if (v >= 0 && b[k++] != static_cast<std::uint8_t>(v)) return false;
            else if (v < 0) ++k;
        return true;
    };
    if (starts({0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A})) return Format::png;
    if (starts({0xFF, 0xD8, 0xFF})) return Format::jpeg;
    if (starts({'G', 'I', 'F', '8'})) return Format::gif;
    if (starts({'R', 'I', 'F', 'F', -1, -1, -1, -1, 'W', 'E', 'B', 'P'})) return Format::webp;
    if (starts({'B', 'M'})) return Format::bmp;
    return Format::unknown;
}

const char* format_name(Format f) noexcept {
    switch (f) {
        case Format::png: return "PNG";
        case Format::jpeg: return "JPEG";
        case Format::gif: return "GIF";
        case Format::webp: return "WebP";
        case Format::bmp: return "BMP";
        default: return "unknown";
    }
}

Rgb8 decode_image(std::span<const std::uint8_t> bytes, const DecodeLimits& limits) {
    if (bytes.size() > limits.max_input_bytes) throw ImageError("image input: the picture is larger than the input limit");
    const auto f = sniff(bytes);
    if (f == Format::png) return decode_png(bytes, limits);
    if (f == Format::jpeg) return decode_jpeg(bytes, limits);
    if (f == Format::unknown) throw ImageError("image input: cannot identify image file (PNG and JPEG are supported)");
    throw ImageError(std::string("image input: ") + format_name(f) + " pictures are not supported (send PNG or JPEG)");
}

}  // namespace strix::image
