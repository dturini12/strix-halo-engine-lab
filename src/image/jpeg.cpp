// JPEG to RGB8 as Pillow 11.3 / libjpeg-turbo 3.1 decode it with their defaults: Huffman baseline, extended
// (8-bit) and progressive; the integer "islow" IDCT (jidctint.c), fancy upsampling for 2x1, 1x2 and 2x2 chroma
// (jdsample.c, including its "downsampled width > 2" rule), plain replication for other integral factors, and the
// fixed-point YCbCr -> RGB tables of jdcolor.c. Colour space as jdapimin.c guesses it (JFIF, Adobe transform,
// component ids). Grey JPEGs read as grey replicated to RGB. CMYK/YCCK, arithmetic coding, lossless,
// hierarchical and 12-bit files are refused.
#include "strix/image/image_decode.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strix::image {
namespace {

[[noreturn]] void fail(const std::string& what) { throw ImageError("JPEG: " + what); }

constexpr std::array<std::uint8_t, 64 + 16> kNatural{
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,
    6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31,
    39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    // extra entries for safety in decoders that overrun (libjpeg's jpeg_natural_order has 16 of 63)
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63};

struct HuffTable {
    bool defined = false;
    std::array<std::int32_t, 18> maxcode{};
    std::array<std::int32_t, 17> valoffset{};
    std::array<std::uint8_t, 256> values{};
    // fast lookup: 9 bits -> (length << 8 | value), 0 when longer
    std::array<std::uint16_t, 512> fast{};

    void build(const std::uint8_t* bits /*[17], bits[0] unused*/, const std::uint8_t* vals, int nvals) {
        std::memcpy(values.data(), vals, static_cast<std::size_t>(nvals));
        std::array<std::uint16_t, 257> huffcode{};
        std::array<std::uint8_t, 257> huffsize{};
        int p = 0;
        for (int l = 1; l <= 16; ++l)
            for (int i = 0; i < bits[l]; ++i) huffsize[static_cast<std::size_t>(p++)] = static_cast<std::uint8_t>(l);
        huffsize[static_cast<std::size_t>(p)] = 0;
        int code = 0, si = huffsize[0];
        p = 0;
        while (huffsize[static_cast<std::size_t>(p)] != 0U) {
            while (huffsize[static_cast<std::size_t>(p)] == si) huffcode[static_cast<std::size_t>(p++)] = static_cast<std::uint16_t>(code++);
            if (code >= (1 << si)) fail("bad Huffman table");
            code <<= 1;
            ++si;
        }
        p = 0;
        for (int l = 1; l <= 16; ++l) {
            if (bits[l] != 0U) {
                valoffset[static_cast<std::size_t>(l)] = p - huffcode[static_cast<std::size_t>(p)];
                p += bits[l];
                maxcode[static_cast<std::size_t>(l)] = huffcode[static_cast<std::size_t>(p - 1)];
            } else {
                maxcode[static_cast<std::size_t>(l)] = -1;
            }
        }
        maxcode[17] = 0x7FFFFFFF;
        fast.fill(0);
        p = 0;
        for (int l = 1; l <= 9; ++l)
            for (int i = 0; i < bits[l]; ++i, ++p) {
                const int lookbits = huffcode[static_cast<std::size_t>(p)] << (9 - l);
                for (int c = 0; c < (1 << (9 - l)); ++c)
                    fast[static_cast<std::size_t>(lookbits + c)] = static_cast<std::uint16_t>((l << 8) | values[static_cast<std::size_t>(p)]);
            }
        defined = true;
    }
};

struct Component {
    int id = 0, h = 1, v = 1, tq = 0;
    int bw = 0, bh = 0;          // blocks covering the component (non-interleaved extent)
    int bw_pad = 0, bh_pad = 0;  // blocks covering all MCUs (interleaved extent)
    int dw = 0, dh = 0;          // downsampled width/height in samples
    std::vector<std::int16_t> coef;  // bw_pad * bh_pad blocks of 64, natural order
    std::array<std::uint16_t, 64> quant{};
    bool quant_latched = false;
    int dc_pred = 0;
    int dc_tbl = 0, ac_tbl = 0;
    std::int16_t* block(int bx, int by) { return coef.data() + (static_cast<std::size_t>(by) * static_cast<std::size_t>(bw_pad) + static_cast<std::size_t>(bx)) * 64U; }
};

class BitReader {
public:
    BitReader(const std::uint8_t* data, std::size_t size, std::size_t at) : d_(data), n_(size), at_(at) {}
    std::size_t position() const { return at_; }
    void reset() { acc_ = 0; bits_ = 0; marker_ = false; }

    void fill() {
        while (bits_ <= 56) {
            unsigned byte = 0;
            if (!marker_ && at_ < n_) {
                byte = d_[at_];
                if (byte == 0xFFU) {
                    const unsigned next = at_ + 1U < n_ ? d_[at_ + 1U] : 0xD9U;
                    if (next == 0x00U) {
                        at_ += 2U;
                    } else {
                        marker_ = true;  // a marker: feed zeros, as libjpeg does
                        byte = 0;
                    }
                } else {
                    ++at_;
                }
            }
            acc_ |= std::uint64_t{byte} << (56U - static_cast<unsigned>(bits_));
            bits_ += 8;
        }
    }
    unsigned peek(int n) {
        if (bits_ < n) fill();
        return static_cast<unsigned>(acc_ >> (64U - static_cast<unsigned>(n)));
    }
    void skip(int n) {
        acc_ <<= static_cast<unsigned>(n);
        bits_ -= n;
    }
    int bit() {
        const auto b = peek(1);
        skip(1);
        return static_cast<int>(b);
    }
    int receive(int s) {
        if (s == 0) return 0;
        const auto v = peek(s);
        skip(s);
        return static_cast<int>(v);
    }
    static int extend(int v, int s) { return s == 0 ? 0 : (v < (1 << (s - 1)) ? v - (1 << s) + 1 : v); }
    int decode(const HuffTable& t) {
        if (!t.defined) fail("undefined Huffman table");
        const auto look = peek(9);
        if (const auto f = t.fast[look]; f != 0U) {
            skip(f >> 8U);
            return f & 0xFFU;
        }
        const auto look16 = peek(16);
        for (int l = 10; l <= 16; ++l) {
            const auto code = static_cast<std::int32_t>(look16 >> (16 - l));
            if (code <= t.maxcode[static_cast<std::size_t>(l)]) {
                skip(l);
                const int idx = code + t.valoffset[static_cast<std::size_t>(l)];
                if (idx < 0 || idx > 255) fail("corrupt Huffman data");
                return t.values[static_cast<std::size_t>(idx)];
            }
        }
        skip(16);
        return 0;  // corrupt data: libjpeg warns and returns 0
    }
    // Skip to the restart marker (and past it); bits left in the buffer are discarded.
    void restart(int expected) {
        reset();
        // find the marker
        while (at_ + 1U < n_ && !(d_[at_] == 0xFFU && d_[at_ + 1U] != 0x00U && d_[at_ + 1U] != 0xFFU)) ++at_;
        while (at_ + 1U < n_ && d_[at_] == 0xFFU && d_[at_ + 1U] == 0xFFU) ++at_;
        if (at_ + 1U < n_ && d_[at_ + 1U] == 0xD0U + static_cast<unsigned>(expected)) at_ += 2U;
    }
    // After a scan: position of the next marker.
    std::size_t next_marker() {
        std::size_t p = at_;
        while (p + 1U < n_ && !(d_[p] == 0xFFU && d_[p + 1U] != 0x00U && !(d_[p + 1U] >= 0xD0U && d_[p + 1U] <= 0xD7U))) ++p;
        return p;
    }

private:
    const std::uint8_t* d_;
    std::size_t n_, at_;
    std::uint64_t acc_ = 0;
    int bits_ = 0;
    bool marker_ = false;
};

// jidctint.c jpeg_idct_islow (libjpeg-turbo), into a plane.
constexpr int kConstBits = 13, kPass1Bits = 2;
constexpr std::int64_t FIX_0_298631336 = 2446, FIX_0_390180644 = 3196, FIX_0_541196100 = 4433, FIX_0_765366865 = 6270,
                       FIX_0_899976223 = 7373, FIX_1_175875602 = 9633, FIX_1_501321110 = 12299, FIX_1_847759065 = 15137,
                       FIX_1_961570560 = 16069, FIX_2_053119869 = 16819, FIX_2_562915447 = 20995, FIX_3_072711026 = 25172;

inline std::int64_t descale(std::int64_t x, int n) { return (x + (std::int64_t{1} << (n - 1))) >> n; }

// libjpeg's sample_range_limit + CENTERJSAMPLE, indexed by (x & 1023): post-IDCT clamp with its wraparound.
const std::array<std::uint8_t, 1024>& idct_limit() {
    static const auto table = [] {
        std::array<std::uint8_t, 1024> t{};
        for (int i = 0; i < 1024; ++i) {
            if (i < 128) t[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i + 128);
            else if (i < 512) t[static_cast<std::size_t>(i)] = 255;
            else if (i < 896) t[static_cast<std::size_t>(i)] = 0;
            else t[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i - 896);
        }
        return t;
    }();
    return table;
}

void idct_islow(const std::int16_t* in, const std::uint16_t* q, std::uint8_t* out, std::size_t stride) {
    const auto& limit = idct_limit();
    std::array<int, 64> ws{};
    for (int c = 0; c < 8; ++c) {
        const std::int16_t* ip = in + c;
        const std::uint16_t* qp = q + c;
        int* wp = ws.data() + c;
        if (ip[8] == 0 && ip[16] == 0 && ip[24] == 0 && ip[32] == 0 && ip[40] == 0 && ip[48] == 0 && ip[56] == 0) {
            const int dc = (ip[0] * qp[0]) * (1 << kPass1Bits);
            for (int r = 0; r < 8; ++r) wp[r * 8] = dc;
            continue;
        }
        std::int64_t z2 = ip[16] * qp[16], z3 = ip[48] * qp[48];
        std::int64_t z1 = (z2 + z3) * FIX_0_541196100;
        std::int64_t tmp2 = z1 + z3 * -FIX_1_847759065;
        std::int64_t tmp3 = z1 + z2 * FIX_0_765366865;
        z2 = ip[0] * qp[0];
        z3 = ip[32] * qp[32];
        std::int64_t tmp0 = (z2 + z3) * (1 << kConstBits);
        std::int64_t tmp1 = (z2 - z3) * (1 << kConstBits);
        const std::int64_t tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
        tmp0 = ip[56] * qp[56];
        tmp1 = ip[40] * qp[40];
        tmp2 = ip[24] * qp[24];
        tmp3 = ip[8] * qp[8];
        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        std::int64_t z4 = tmp1 + tmp3;
        const std::int64_t z5 = (z3 + z4) * FIX_1_175875602;
        tmp0 *= FIX_0_298631336;
        tmp1 *= FIX_2_053119869;
        tmp2 *= FIX_3_072711026;
        tmp3 *= FIX_1_501321110;
        z1 *= -FIX_0_899976223;
        z2 *= -FIX_2_562915447;
        z3 *= -FIX_1_961570560;
        z4 *= -FIX_0_390180644;
        z3 += z5;
        z4 += z5;
        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;
        constexpr int s = kConstBits - kPass1Bits;
        wp[0] = static_cast<int>(descale(tmp10 + tmp3, s));
        wp[56] = static_cast<int>(descale(tmp10 - tmp3, s));
        wp[8] = static_cast<int>(descale(tmp11 + tmp2, s));
        wp[48] = static_cast<int>(descale(tmp11 - tmp2, s));
        wp[16] = static_cast<int>(descale(tmp12 + tmp1, s));
        wp[40] = static_cast<int>(descale(tmp12 - tmp1, s));
        wp[24] = static_cast<int>(descale(tmp13 + tmp0, s));
        wp[32] = static_cast<int>(descale(tmp13 - tmp0, s));
    }
    for (int r = 0; r < 8; ++r) {
        const int* wp = ws.data() + r * 8;
        std::uint8_t* op = out + static_cast<std::size_t>(r) * stride;
        if (wp[1] == 0 && wp[2] == 0 && wp[3] == 0 && wp[4] == 0 && wp[5] == 0 && wp[6] == 0 && wp[7] == 0) {
            const auto v = limit[static_cast<std::size_t>(descale(wp[0], kPass1Bits + 3) & 1023)];
            for (int c = 0; c < 8; ++c) op[c] = v;
            continue;
        }
        std::int64_t z2 = wp[2], z3 = wp[6];
        std::int64_t z1 = (z2 + z3) * FIX_0_541196100;
        std::int64_t tmp2 = z1 + z3 * -FIX_1_847759065;
        std::int64_t tmp3 = z1 + z2 * FIX_0_765366865;
        std::int64_t tmp0 = (std::int64_t{wp[0]} + wp[4]) * (1 << kConstBits);
        std::int64_t tmp1 = (std::int64_t{wp[0]} - wp[4]) * (1 << kConstBits);
        const std::int64_t tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
        tmp0 = wp[7];
        tmp1 = wp[5];
        tmp2 = wp[3];
        tmp3 = wp[1];
        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        std::int64_t z4 = tmp1 + tmp3;
        const std::int64_t z5 = (z3 + z4) * FIX_1_175875602;
        tmp0 *= FIX_0_298631336;
        tmp1 *= FIX_2_053119869;
        tmp2 *= FIX_3_072711026;
        tmp3 *= FIX_1_501321110;
        z1 *= -FIX_0_899976223;
        z2 *= -FIX_2_562915447;
        z3 *= -FIX_1_961570560;
        z4 *= -FIX_0_390180644;
        z3 += z5;
        z4 += z5;
        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;
        constexpr int s = kConstBits + kPass1Bits + 3;
        const auto put = [&](int c, std::int64_t v) { op[c] = limit[static_cast<std::size_t>(descale(v, s) & 1023)]; };
        put(0, tmp10 + tmp3);
        put(7, tmp10 - tmp3);
        put(1, tmp11 + tmp2);
        put(6, tmp11 - tmp2);
        put(2, tmp12 + tmp1);
        put(5, tmp12 - tmp1);
        put(3, tmp13 + tmp0);
        put(4, tmp13 - tmp0);
    }
}

std::uint16_t be16(const std::uint8_t* p) { return static_cast<std::uint16_t>((p[0] << 8) | p[1]); }

struct Decoder {
    std::span<const std::uint8_t> in;
    DecodeLimits limits;
    int width = 0, height = 0;
    bool progressive = false, have_frame = false;
    std::vector<Component> comps;
    int hmax = 1, vmax = 1, mcus_x = 0, mcus_y = 0;
    std::array<std::array<std::uint16_t, 64>, 4> qt{};
    std::array<bool, 4> qt_defined{};
    std::array<HuffTable, 4> dc{}, ac{};
    int restart_interval = 0;
    bool jfif = false, adobe = false;
    int adobe_transform = 0;
    int eobrun = 0;

    void frame(const std::uint8_t* p, std::size_t len) {
        if (have_frame) fail("more than one frame");
        if (len < 6U) fail("bad SOF");
        if (p[0] != 8U) fail("only 8-bit JPEGs are supported");
        height = be16(p + 1);
        width = be16(p + 3);
        const int n = p[5];
        if (height == 0) fail("DNL-defined height is not supported");
        if (width == 0) fail("zero width");
        if (n != 1 && n != 3) fail(n == 4 ? "CMYK JPEGs are not supported" : "unsupported number of components");
        if (len < 6U + 3U * static_cast<std::size_t>(n)) fail("bad SOF");
        if (static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) > limits.max_pixels)
            throw ImageError("image input: Image size (" + std::to_string(static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height)) +
                             " pixels) exceeds limit of " + std::to_string(limits.max_pixels) + " pixels, could be decompression bomb DOS attack.");
        comps.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            auto& c = comps[static_cast<std::size_t>(i)];
            c.id = p[6 + 3 * i];
            c.h = p[7 + 3 * i] >> 4;
            c.v = p[7 + 3 * i] & 15;
            c.tq = p[8 + 3 * i];
            if (c.h < 1 || c.h > 4 || c.v < 1 || c.v > 4 || c.tq > 3) fail("bad component parameters");
            hmax = std::max(hmax, c.h);
            vmax = std::max(vmax, c.v);
        }
        mcus_x = (width + 8 * hmax - 1) / (8 * hmax);
        mcus_y = (height + 8 * vmax - 1) / (8 * vmax);
        for (auto& c : comps) {
            c.dw = (width * c.h + hmax - 1) / hmax;
            c.dh = (height * c.v + vmax - 1) / vmax;
            c.bw = (c.dw + 7) / 8;
            c.bh = (c.dh + 7) / 8;
            c.bw_pad = mcus_x * c.h;
            c.bh_pad = mcus_y * c.v;
            c.coef.assign(static_cast<std::size_t>(c.bw_pad) * static_cast<std::size_t>(c.bh_pad) * 64U, 0);
        }
        have_frame = true;
    }

    void decode_block_baseline(BitReader& br, Component& c, std::int16_t* blk) {
        const int s = br.decode(dc[static_cast<std::size_t>(c.dc_tbl)]);
        const int diff = s ? BitReader::extend(br.receive(s), s) : 0;
        c.dc_pred += diff;
        blk[0] = static_cast<std::int16_t>(c.dc_pred);
        const auto& at = ac[static_cast<std::size_t>(c.ac_tbl)];
        for (int k = 1; k < 64;) {
            const int rs = br.decode(at);
            const int r = rs >> 4, ss = rs & 15;
            if (ss) {
                k += r;
                if (k > 63) break;
                blk[kNatural[static_cast<std::size_t>(k)]] = static_cast<std::int16_t>(BitReader::extend(br.receive(ss), ss));
                ++k;
            } else {
                if (r != 15) break;
                k += 16;
            }
        }
    }

    void scan(const std::uint8_t* p, std::size_t len, std::size_t& at) {
        if (!have_frame) fail("SOS before SOF");
        const int ns = p[0];
        if (ns < 1 || ns > 4 || len < 4U + 2U * static_cast<std::size_t>(ns)) fail("bad SOS");
        std::vector<Component*> sc;
        for (int i = 0; i < ns; ++i) {
            const int id = p[1 + 2 * i];
            Component* found = nullptr;
            for (auto& c : comps)
                if (c.id == id) found = &c;
            if (!found) fail("SOS names an unknown component");
            found->dc_tbl = p[2 + 2 * i] >> 4;
            found->ac_tbl = p[2 + 2 * i] & 15;
            if (found->dc_tbl > 3 || found->ac_tbl > 3) fail("bad table selector");
            if (!found->quant_latched) {
                if (!qt_defined[static_cast<std::size_t>(found->tq)]) fail("quantization table not defined");
                found->quant = qt[static_cast<std::size_t>(found->tq)];
                found->quant_latched = true;
            }
            sc.push_back(found);
        }
        const int ss = p[1 + 2 * ns], se = p[2 + 2 * ns], ah = p[3 + 2 * ns] >> 4, al = p[3 + 2 * ns] & 15;
        if (progressive) {
            const bool dc_scan = ss == 0;
            if (dc_scan ? se != 0 : (ss > se || se > 63 || ns != 1)) fail("bad progressive scan parameters");
            if (al > 13 || (ah != 0 && ah - 1 != al)) fail("bad successive approximation");
        }
        for (auto* c : sc) c->dc_pred = 0;
        eobrun = 0;
        BitReader br(in.data(), in.size(), at);
        const bool interleaved = ns > 1;
        const int mx = interleaved ? mcus_x : sc[0]->bw;
        const int my = interleaved ? mcus_y : sc[0]->bh;
        int restarts_left = restart_interval, next_rst = 0;
        for (int y = 0; y < my; ++y) {
            for (int x = 0; x < mx; ++x) {
                if (restart_interval) {
                    if (restarts_left == 0) {
                        br.restart(next_rst);
                        next_rst = (next_rst + 1) & 7;
                        restarts_left = restart_interval;
                        for (auto* c : sc) c->dc_pred = 0;
                        eobrun = 0;
                    }
                    --restarts_left;
                }
                if (interleaved) {
                    for (auto* c : sc)
                        for (int by = 0; by < c->v; ++by)
                            for (int bx = 0; bx < c->h; ++bx) block(br, *c, c->block(x * c->h + bx, y * c->v + by), ss, se, ah, al);
                } else {
                    block(br, *sc[0], sc[0]->block(x, y), ss, se, ah, al);
                }
            }
        }
        at = br.next_marker();
    }

    void block(BitReader& br, Component& c, std::int16_t* blk, int ss, int se, int ah, int al) {
        if (!progressive) {
            decode_block_baseline(br, c, blk);
            return;
        }
        if (ss == 0) {  // DC
            if (ah == 0) {
                const int s = br.decode(dc[static_cast<std::size_t>(c.dc_tbl)]);
                const int diff = s ? BitReader::extend(br.receive(s), s) : 0;
                c.dc_pred += diff;
                blk[0] = static_cast<std::int16_t>(static_cast<unsigned>(c.dc_pred) << static_cast<unsigned>(al));
            } else if (br.bit()) {
                blk[0] = static_cast<std::int16_t>(blk[0] | (1 << al));
            }
            return;
        }
        const auto& at = ac[static_cast<std::size_t>(c.ac_tbl)];
        if (ah == 0) {  // AC first
            if (eobrun > 0) {
                --eobrun;
                return;
            }
            for (int k = ss; k <= se; ++k) {
                const int rs = br.decode(at);
                const int r = rs >> 4, s = rs & 15;
                if (s) {
                    k += r;
                    if (k > 63) break;
                    const int v = BitReader::extend(br.receive(s), s);
                    blk[kNatural[static_cast<std::size_t>(k)]] = static_cast<std::int16_t>(static_cast<int>(static_cast<unsigned>(v) << static_cast<unsigned>(al)));
                } else if (r == 15) {
                    k += 15;
                } else {
                    eobrun = 1 << r;
                    if (r) eobrun += br.receive(r);
                    --eobrun;
                    break;
                }
            }
            return;
        }
        // AC refinement (jdphuff.c decode_mcu_AC_refine)
        const int p1 = 1 << al, m1 = -1 * (1 << al);
        int k = ss;
        if (eobrun == 0) {
            for (; k <= se; ++k) {
                const int rs = br.decode(at);
                int r = rs >> 4, s = rs & 15;
                if (s) {
                    s = br.bit() ? p1 : m1;
                } else if (r != 15) {
                    eobrun = 1 << r;
                    if (r) eobrun += br.receive(r);
                    break;
                }
                do {
                    std::int16_t* coef = blk + kNatural[static_cast<std::size_t>(k)];
                    if (*coef != 0) {
                        if (br.bit() && (*coef & p1) == 0) *coef = static_cast<std::int16_t>(*coef >= 0 ? *coef + p1 : *coef + m1);
                    } else {
                        if (--r < 0) break;
                    }
                    ++k;
                } while (k <= se);
                if (s && k <= 63) blk[kNatural[static_cast<std::size_t>(k)]] = static_cast<std::int16_t>(s);
            }
        }
        if (eobrun > 0) {
            for (; k <= se; ++k) {
                std::int16_t* coef = blk + kNatural[static_cast<std::size_t>(k)];
                if (*coef != 0 && br.bit() && (*coef & p1) == 0) *coef = static_cast<std::int16_t>(*coef >= 0 ? *coef + p1 : *coef + m1);
            }
            --eobrun;
        }
    }

    Rgb8 finish() {
        // IDCT every block of every component into a padded plane
        std::vector<std::vector<std::uint8_t>> planes(comps.size());
        for (std::size_t ci = 0; ci < comps.size(); ++ci) {
            auto& c = comps[ci];
            if (!c.quant_latched) fail("component never appears in a scan");
            const std::size_t pw = static_cast<std::size_t>(c.bw_pad) * 8U;
            planes[ci].assign(pw * static_cast<std::size_t>(c.bh_pad) * 8U, 0);
            for (int by = 0; by < c.bh_pad; ++by)
                for (int bx = 0; bx < c.bw_pad; ++bx)
                    idct_islow(c.block(bx, by), c.quant.data(), planes[ci].data() + static_cast<std::size_t>(by) * 8U * pw + static_cast<std::size_t>(bx) * 8U, pw);
        }
        // upsample each component to width x height
        const auto W = static_cast<std::size_t>(width), H = static_cast<std::size_t>(height);
        std::vector<std::vector<std::uint8_t>> full(comps.size());
        for (std::size_t ci = 0; ci < comps.size(); ++ci) full[ci] = upsample(comps[ci], planes[ci]);
        Rgb8 img;
        img.width = static_cast<std::uint32_t>(width);
        img.height = static_cast<std::uint32_t>(height);
        img.pixels.resize(W * H * 3U);
        if (comps.size() == 1U) {
            for (std::size_t i = 0; i < W * H; ++i) img.pixels[3U * i] = img.pixels[3U * i + 1U] = img.pixels[3U * i + 2U] = full[0][i];
            return img;
        }
        bool rgb = false;
        if (jfif) rgb = false;
        else if (adobe) rgb = adobe_transform == 0;
        else rgb = comps[0].id == 82 && comps[1].id == 71 && comps[2].id == 66;
        if (rgb) {
            for (std::size_t i = 0; i < W * H; ++i)
                for (std::size_t c = 0; c < 3U; ++c) img.pixels[3U * i + c] = full[c][i];
            return img;
        }
        // jdcolor.c build_ycc_rgb_table / ycc_rgb_convert
        static const auto tables = [] {
            struct T {
                std::array<int, 256> cr_r, cb_b;
                std::array<std::int64_t, 256> cr_g, cb_g;
            } t{};
            constexpr int SCALEBITS = 16;
            constexpr std::int64_t ONE_HALF = std::int64_t{1} << (SCALEBITS - 1);
            const auto FIX = [](double x) { return static_cast<std::int64_t>(x * (1L << SCALEBITS) + 0.5); };
            for (int i = 0, x = -128; i < 256; ++i, ++x) {
                t.cr_r[static_cast<std::size_t>(i)] = static_cast<int>((FIX(1.40200) * x + ONE_HALF) >> SCALEBITS);
                t.cb_b[static_cast<std::size_t>(i)] = static_cast<int>((FIX(1.77200) * x + ONE_HALF) >> SCALEBITS);
                t.cr_g[static_cast<std::size_t>(i)] = (-FIX(0.71414)) * x;
                t.cb_g[static_cast<std::size_t>(i)] = (-FIX(0.34414)) * x + ONE_HALF;
            }
            return t;
        }();
        const auto clamp = [](int v) { return static_cast<std::uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v); };
        for (std::size_t i = 0; i < W * H; ++i) {
            const int y = full[0][i], cb = full[1][i], cr = full[2][i];
            img.pixels[3U * i] = clamp(y + tables.cr_r[static_cast<std::size_t>(cr)]);
            img.pixels[3U * i + 1U] = clamp(y + static_cast<int>((tables.cb_g[static_cast<std::size_t>(cb)] + tables.cr_g[static_cast<std::size_t>(cr)]) >> 16));
            img.pixels[3U * i + 2U] = clamp(y + tables.cb_b[static_cast<std::size_t>(cb)]);
        }
        return img;
    }

    // jdsample.c for one component: plane (padded, stride bw_pad*8) -> width x height samples.
    std::vector<std::uint8_t> upsample(const Component& c, const std::vector<std::uint8_t>& plane) const {
        const auto W = static_cast<std::size_t>(width), H = static_cast<std::size_t>(height);
        const std::size_t stride = static_cast<std::size_t>(c.bw_pad) * 8U;
        const int hx = hmax / c.h, vx = vmax / c.v;
        if (hmax % c.h != 0 || vmax % c.v != 0) fail("fractional sampling factors are not supported");
        std::vector<std::uint8_t> out(W * H);
        const auto at_row = [&](long row, std::size_t col) -> int {  // context rows: replicate first/last real row
            if (row < 0) row = 0;
            if (row >= c.dh) row = c.dh - 1;
            return plane[static_cast<std::size_t>(row) * stride + col];
        };
        const std::size_t dw = static_cast<std::size_t>(c.dw);
        std::vector<std::uint8_t> line(dw * 2U + 2U);
        if (hx == 1 && vx == 1) {
            for (std::size_t y = 0; y < H; ++y) std::memcpy(out.data() + y * W, plane.data() + y * stride, W);
        } else if (hx == 2 && vx == 1) {
            if (dw > 2U) {
                for (std::size_t y = 0; y < H; ++y) {
                    const std::uint8_t* ip = plane.data() + y * stride;
                    std::uint8_t* op = line.data();
                    int v = ip[0];
                    *op++ = static_cast<std::uint8_t>(v);
                    *op++ = static_cast<std::uint8_t>((v * 3 + ip[1] + 2) >> 2);
                    for (std::size_t k = 1; k + 1U < dw; ++k) {
                        v = ip[k] * 3;
                        *op++ = static_cast<std::uint8_t>((v + ip[k - 1U] + 1) >> 2);
                        *op++ = static_cast<std::uint8_t>((v + ip[k + 1U] + 2) >> 2);
                    }
                    v = ip[dw - 1U];
                    *op++ = static_cast<std::uint8_t>((v * 3 + ip[dw - 2U] + 1) >> 2);
                    *op++ = static_cast<std::uint8_t>(v);
                    std::memcpy(out.data() + y * W, line.data(), W);
                }
            } else {
                for (std::size_t y = 0; y < H; ++y)
                    for (std::size_t x = 0; x < W; ++x) out[y * W + x] = plane[y * stride + x / 2U];
            }
        } else if (hx == 1 && vx == 2) {
            for (std::size_t y = 0; y < H; ++y) {
                const long row = static_cast<long>(y / 2U);
                const bool upper = y % 2U == 0U;
                const long other = upper ? row - 1 : row + 1;
                const int bias = upper ? 1 : 2;
                for (std::size_t x = 0; x < W; ++x)
                    out[y * W + x] = static_cast<std::uint8_t>((at_row(row, x) * 3 + at_row(other, x) + bias) >> 2);
            }
        } else if (hx == 2 && vx == 2) {
            if (dw > 2U) {
                for (std::size_t y = 0; y < H; ++y) {
                    const long row = static_cast<long>(y / 2U);
                    const long other = y % 2U == 0U ? row - 1 : row + 1;
                    std::uint8_t* op = line.data();
                    const auto colsum = [&](std::size_t k) { return at_row(row, k) * 3 + at_row(other, k); };
                    int thiscol = colsum(0), nextcol = colsum(1), lastcol;
                    *op++ = static_cast<std::uint8_t>((thiscol * 4 + 8) >> 4);
                    *op++ = static_cast<std::uint8_t>((thiscol * 3 + nextcol + 7) >> 4);
                    lastcol = thiscol;
                    thiscol = nextcol;
                    for (std::size_t k = 2; k < dw; ++k) {
                        nextcol = colsum(k);
                        *op++ = static_cast<std::uint8_t>((thiscol * 3 + lastcol + 8) >> 4);
                        *op++ = static_cast<std::uint8_t>((thiscol * 3 + nextcol + 7) >> 4);
                        lastcol = thiscol;
                        thiscol = nextcol;
                    }
                    *op++ = static_cast<std::uint8_t>((thiscol * 3 + lastcol + 8) >> 4);
                    *op++ = static_cast<std::uint8_t>((thiscol * 4 + 7) >> 4);
                    std::memcpy(out.data() + y * W, line.data(), W);
                }
            } else {
                for (std::size_t y = 0; y < H; ++y)
                    for (std::size_t x = 0; x < W; ++x) out[y * W + x] = plane[(y / 2U) * stride + x / 2U];
            }
        } else {  // int_upsample: replicate
            for (std::size_t y = 0; y < H; ++y)
                for (std::size_t x = 0; x < W; ++x)
                    out[y * W + x] = plane[(y / static_cast<std::size_t>(vx)) * stride + x / static_cast<std::size_t>(hx)];
        }
        return out;
    }
};

}  // namespace

Rgb8 decode_jpeg(std::span<const std::uint8_t> bytes, const DecodeLimits& limits) {
    if (bytes.size() < 4U || bytes[0] != 0xFFU || bytes[1] != 0xD8U) fail("not a JPEG file");
    Decoder d;
    d.in = bytes;
    d.limits = limits;
    std::size_t at = 2;
    bool any_scan = false;
    for (;;) {
        while (at < bytes.size() && bytes[at] != 0xFFU) ++at;  // garbage between markers
        while (at < bytes.size() && bytes[at] == 0xFFU) ++at;  // fill bytes
        if (at >= bytes.size()) break;                          // missing EOI: decode what we have
        const unsigned m = bytes[at++];
        if (m == 0xD9U) break;
        if (m == 0xD8U || (m >= 0xD0U && m <= 0xD7U) || m == 0x01U) continue;
        if (at + 2U > bytes.size()) fail("truncated marker");
        const std::size_t len = be16(bytes.data() + at);
        if (len < 2U || at + len > bytes.size()) fail("truncated marker segment");
        const std::uint8_t* p = bytes.data() + at + 2U;
        const std::size_t plen = len - 2U;
        at += len;
        switch (m) {
            case 0xC0:
            case 0xC1:
                d.frame(p, plen);
                break;
            case 0xC2:
                d.progressive = true;
                d.frame(p, plen);
                break;
            case 0xC3: case 0xC5: case 0xC6: case 0xC7: case 0xC9: case 0xCA: case 0xCB: case 0xCD: case 0xCE: case 0xCF:
                fail("unsupported JPEG process (lossless, hierarchical or arithmetic coding)");
            case 0xC4: {
                std::size_t k = 0;
                while (k < plen) {
                    if (plen - k < 17U) fail("bad DHT");
                    const int tc = p[k] >> 4, th = p[k] & 15;
                    if (tc > 1 || th > 3) fail("bad DHT");
                    std::uint8_t bits[17] = {0};
                    int total = 0;
                    for (int l = 1; l <= 16; ++l) total += bits[l] = p[k + static_cast<std::size_t>(l)];
                    if (total > 256 || plen - k - 17U < static_cast<std::size_t>(total)) fail("bad DHT");
                    (tc == 0 ? d.dc : d.ac)[static_cast<std::size_t>(th)].build(bits, p + k + 17U, total);
                    k += 17U + static_cast<std::size_t>(total);
                }
                break;
            }
            case 0xDB: {
                std::size_t k = 0;
                while (k < plen) {
                    const int pq = p[k] >> 4, tq = p[k] & 15;
                    if (tq > 3 || pq > 1) fail("bad DQT");
                    const std::size_t need = pq ? 129U : 65U;
                    if (plen - k < need) fail("bad DQT");
                    for (int i = 0; i < 64; ++i)
                        d.qt[static_cast<std::size_t>(tq)][kNatural[static_cast<std::size_t>(i)]] =
                            pq ? be16(p + k + 1U + 2U * static_cast<std::size_t>(i)) : p[k + 1U + static_cast<std::size_t>(i)];
                    d.qt_defined[static_cast<std::size_t>(tq)] = true;
                    k += need;
                }
                break;
            }
            case 0xDD:
                if (plen < 2U) fail("bad DRI");
                d.restart_interval = be16(p);
                break;
            case 0xDA:
                d.scan(p, plen, at);
                any_scan = true;
                break;
            case 0xE0:
                if (plen >= 5U && std::memcmp(p, "JFIF\0", 5) == 0) d.jfif = true;
                break;
            case 0xEE:
                if (plen >= 12U && std::memcmp(p, "Adobe", 5) == 0) {
                    d.adobe = true;
                    d.adobe_transform = p[11];
                }
                break;
            case 0xDC:
                fail("DNL marker is not supported");
            default:
                break;  // APPn, COM, ...
        }
    }
    if (!d.have_frame || !any_scan) fail("no image data");
    return d.finish();
}

}  // namespace strix::image
