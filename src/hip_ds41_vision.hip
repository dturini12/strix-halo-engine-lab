#pragma clang fp contract(off)
// DeepSeek V4.1 vision tower on the device: the declared arithmetic of ds41_vision_reference.cpp, bit for bit.
//  - linear / attention products: a tiled GEMM in which every output keeps one FP32 accumulator and adds
//    fma(x[k], w[k]) for k ascending from 0 (tiles of 16 k in order); tails pad x and w with 0, which leaves the
//    accumulator unchanged (it starts at +0 and can never become -0);
//  - RMSNorm: one wave per row, wave-order sum (ds41_vision_math.hpp);
//  - attention: one fused kernel per (64 query rows, head), online softmax over key tiles in the declared order;
//  - element-wise steps share their code with the host (ds41_vision_math.hpp).
// docs/models/deepseek-v4.1/vision.md, "Device tower".
#include "strix/hip_ds41_vision.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strix::deepseek41::vision {
namespace {

void check(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string("ds41 vision: ") + what + ": " + hipGetErrorString(status));
}

constexpr int kBM = 128, kBK = 16, kThreads = 256;

struct GemmArgs {
    const void* x;          // BF16 (uint16) or FP32 rows
    long long sxm, sxk;     // element strides of x(m, k)
    const std::uint16_t* w; // BF16
    long long swo, swk;     // element strides of w(o, k)
    const std::uint16_t* bias;  // BF16 [O] or null (BF16 output only)
    void* y;
    long long sym, syo;     // element strides of y(m, o)
    int M, O, K;
    float scale;            // FP32 output: y = acc * scale
    long long bzx, bzw, bzy;  // per-blockIdx.z element offsets
};

template <typename XT>
__device__ __forceinline__ float load_x(const XT* p);
template <>
__device__ __forceinline__ float load_x<std::uint16_t>(const std::uint16_t* p) { return bf16_float(*p); }
template <>
__device__ __forceinline__ float load_x<float>(const float* p) { return *p; }

// y(m, o) = acc over k ascending of fma(x(m, k), w(o, k), acc); then BF16(acc + bias) or FP32 acc * scale.
template <typename XT, bool kF32Out, int kTO>
__global__ __launch_bounds__(kThreads) void gemm_kernel(GemmArgs a) {
    constexpr int BO = 16 * kTO;
    __shared__ float As[kBK][kBM + 4];
    __shared__ float Bs[kBK][BO + 4];
    const int tx = static_cast<int>(threadIdx.x % 16U), ty = static_cast<int>(threadIdx.x / 16U);
    const int o0 = static_cast<int>(blockIdx.x) * BO, m0 = static_cast<int>(blockIdx.y) * kBM;
    const XT* x = static_cast<const XT*>(a.x) + a.bzx * blockIdx.z;
    const std::uint16_t* w = a.w + a.bzw * blockIdx.z;
    float acc[8][kTO];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < kTO; ++j) acc[i][j] = 0.0F;
    const bool w_k_fast = a.swk == 1;
    for (int k0 = 0; k0 < a.K; k0 += kBK) {
        for (int e = static_cast<int>(threadIdx.x); e < kBM * kBK; e += kThreads) {
            const int mi = e / kBK, kk = e % kBK, m = m0 + mi, k = k0 + kk;
            As[kk][mi] = (m < a.M && k < a.K) ? load_x<XT>(x + m * a.sxm + k * a.sxk) : 0.0F;
        }
        for (int e = static_cast<int>(threadIdx.x); e < BO * kBK; e += kThreads) {
            const int oi = w_k_fast ? e / kBK : e % BO, kk = w_k_fast ? e % kBK : e / BO, o = o0 + oi, k = k0 + kk;
            Bs[kk][oi] = (o < a.O && k < a.K) ? bf16_float(w[o * a.swo + k * a.swk]) : 0.0F;
        }
        __syncthreads();
        for (int kk = 0; kk < kBK; ++kk) {
            float av[8], bv[kTO];
            for (int i = 0; i < 8; ++i) av[i] = As[kk][ty + 16 * i];
            for (int j = 0; j < kTO; ++j) bv[j] = Bs[kk][tx + 16 * j];
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < kTO; ++j) acc[i][j] = __builtin_fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 8; ++i) {
        const int m = m0 + ty + 16 * i;
        if (m >= a.M) continue;
        for (int j = 0; j < kTO; ++j) {
            const int o = o0 + tx + 16 * j;
            if (o >= a.O) continue;
            const long long at = a.bzy * blockIdx.z + m * a.sym + o * a.syo;
            if constexpr (kF32Out) {
                static_cast<float*>(a.y)[at] = acc[i][j] * a.scale;
            } else {
                const float v = a.bias ? acc[i][j] + bf16_float(a.bias[o]) : acc[i][j];
                static_cast<std::uint16_t*>(a.y)[at] = bf16_round(v);
            }
        }
    }
}

__device__ __forceinline__ float wave_sum(float v) {
    for (int d = 16; d != 0; d /= 2) v = v + __shfl_xor(v, d, 32);
    return v;
}
__device__ __forceinline__ float wave_max(float v) {
    for (int d = 16; d != 0; d /= 2) v = fmaxf(v, __shfl_xor(v, d, 32));
    return v;
}

// One wave per row: ss = wave-order sum of fma(x, x), out = bf16(w * (x * r)).
__global__ __launch_bounds__(kThreads) void rms_norm_kernel(const std::uint16_t* x, const std::uint16_t* w, std::uint16_t* out, int rows, int D,
                                                            float eps) {
    const int row = static_cast<int>(blockIdx.x * (kThreads / 32U) + threadIdx.x / 32U), lane = static_cast<int>(threadIdx.x % 32U);
    if (row >= rows) return;
    const std::uint16_t* xr = x + static_cast<long long>(row) * D;
    float part = 0.0F;
    for (int k = lane; k < D; k += 32) {
        const float v = bf16_float(xr[k]);
        part = __builtin_fmaf(v, v, part);
    }
    const float r = vision_inv_root(wave_sum(part), static_cast<float>(D), eps);
    for (int k = lane; k < D; k += 32) out[static_cast<long long>(row) * D + k] = vision_norm(xr[k], r, w[k]);
}

// Attention for 64 query rows of one head (online softmax over key tiles of kVisionKeyBlock, the host reference's
// declared order). Thread t = 16 tr + tc owns rows 4 tr .. 4 tr + 3 with keys 4 tc .. 4 tc + 3 (scores) and with dims
// 4 tc .. 4 tc + 3 (output). A row lives on the 16 lanes tc = 0..15 of a half wave: its running max, rescale factor
// and sum are kept identically in those lanes, combined by xor shuffles 8, 4, 2, 1.
constexpr int kQT = 64, kKB = static_cast<int>(kVisionKeyBlock), kHD = 64;
static_assert(kKB == 64, "the attention tiles assume 64 keys per block");
__device__ __forceinline__ float half_max(float v) {
    for (int d = 8; d != 0; d /= 2) v = fmaxf(v, __shfl_xor(v, d, 32));
    return v;
}
__device__ __forceinline__ float half_sum(float v) {
    for (int d = 8; d != 0; d /= 2) v = v + __shfl_xor(v, d, 32);
    return v;
}
__global__ __launch_bounds__(kThreads) void attention_kernel(const std::uint16_t* qkv, std::uint16_t* attn, int N, int D) {
    // FP32 tiles (padded strides: aligned vector reads, few-bank transposed stores); the K tile and the P tile share KP
    constexpr int kS = 68;
    __shared__ __attribute__((aligned(16))) float Qs[kHD][kS];  // [d][row]
    __shared__ __attribute__((aligned(16))) float KP[kHD][kS];  // [d][key] for the scores, then [key][row] for P
    __shared__ __attribute__((aligned(16))) float Vs[kKB][kS];  // [key][d]
    const int hh = static_cast<int>(blockIdx.y), q0 = static_cast<int>(blockIdx.x) * kQT;
    const int t = static_cast<int>(threadIdx.x), tr = t / 16, tc = t % 16;
    const long long row3 = 3LL * D;
    const auto unpack = [](uint2 w, float* f) {
        f[0] = __builtin_bit_cast(float, w.x << 16U);
        f[1] = __builtin_bit_cast(float, w.x & 0xFFFF0000U);
        f[2] = __builtin_bit_cast(float, w.y << 16U);
        f[3] = __builtin_bit_cast(float, w.y & 0xFFFF0000U);
    };
    for (int e = t; e < kQT * kHD / 4; e += kThreads) {
        const int r = e / (kHD / 4), dq = (e % (kHD / 4)) * 4;
        float f[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        if (q0 + r < N) unpack(*reinterpret_cast<const uint2*>(qkv + (q0 + r) * row3 + hh * kHD + dq), f);
        for (int i = 0; i < 4; ++i) Qs[dq + i][r] = f[i];
    }
    float o[4][4], m[4], lsum[4];
    for (int a = 0; a < 4; ++a) {
        m[a] = -__builtin_huge_valf();
        lsum[a] = 0.0F;
        for (int b = 0; b < 4; ++b) o[a][b] = 0.0F;
    }
    // the next key tile's K and V are fetched into registers while the current tile is computed
    constexpr int kLoads = kKB * kHD / 4 / kThreads;  // uint2 per thread per tile and tensor
    static_assert(kKB * kHD / 4 % kThreads == 0, "tile load split");
    uint2 nk[kLoads], nv[kLoads];
    const auto fetch = [&](int jstart) {
        const int kc = min(kKB, N - jstart);
        for (int u = 0; u < kLoads; ++u) {
            const int e = t + u * kThreads, jj = e / (kHD / 4), dq = (e % (kHD / 4)) * 4;
            nk[u] = make_uint2(0U, 0U);
            nv[u] = make_uint2(0U, 0U);
            if (jj < kc) {
                const std::uint16_t* src = qkv + (jstart + jj) * row3 + hh * kHD + dq;
                nk[u] = *reinterpret_cast<const uint2*>(src + D);
                nv[u] = *reinterpret_cast<const uint2*>(src + 2LL * D);
            }
        }
    };
    fetch(0);
    for (int j0 = 0; j0 < N; j0 += kKB) {
        const int kn = min(kKB, N - j0);
        __syncthreads();  // the previous block's P and V reads are done
        for (int u = 0; u < kLoads; ++u) {
            const int e = t + u * kThreads, jj = e / (kHD / 4), dq = (e % (kHD / 4)) * 4;
            float fk[4], fv[4];
            unpack(nk[u], fk);
            unpack(nv[u], fv);
            for (int i = 0; i < 4; ++i) KP[dq + i][jj] = fk[i];
            *reinterpret_cast<float4*>(&Vs[jj][dq]) = make_float4(fv[0], fv[1], fv[2], fv[3]);
        }
        __syncthreads();
        if (j0 + kKB < N) fetch(j0 + kKB);
        // scores: s = (sum over d ascending of fma(q, k)) * 0.125
        float acc[4][4];
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) acc[a][b] = 0.0F;
#pragma unroll 8
        for (int d = 0; d < kHD; ++d) {
            const float4 qv = *reinterpret_cast<const float4*>(&Qs[d][4 * tr]);
            const float4 kv = *reinterpret_cast<const float4*>(&KP[d][4 * tc]);
            const float qa[4] = {qv.x, qv.y, qv.z, qv.w}, kb[4] = {kv.x, kv.y, kv.z, kv.w};
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) acc[a][b] = __builtin_fmaf(qa[a], kb[b], acc[a][b]);
        }
        __syncthreads();  // every thread is done with the K tile; KP becomes P
        // per row: block max over valid keys, m' = max(m, max), a = exp(m - m'), p = exp(s - m'), l = l * a + block sum
        float al[4];
        for (int a = 0; a < 4; ++a) {
            float s[4], mb = -__builtin_huge_valf();
            for (int b = 0; b < 4; ++b) {
                s[b] = acc[a][b] * 0.125F;
                if (4 * tc + b < kn) mb = fmaxf(mb, s[b]);
            }
            mb = half_max(mb);
            const float mn = fmaxf(m[a], mb);
            al[a] = ds41::declared_exp(m[a] - mn);
            m[a] = mn;
            float p[4];
            for (int b = 0; b < 4; ++b) p[b] = 4 * tc + b < kn ? ds41::declared_exp(s[b] - mn) : 0.0F;
            const float g = half_sum(((p[0] + p[1]) + p[2]) + p[3]);
            lsum[a] = lsum[a] * al[a];
            lsum[a] = lsum[a] + g;
            for (int b = 0; b < 4; ++b) KP[4 * tc + b][4 * tr + a] = p[b];
        }
        __syncthreads();
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) o[a][b] = o[a][b] * al[a];
#pragma unroll 8
        for (int jj = 0; jj < kn; ++jj) {
            const float4 pv = *reinterpret_cast<const float4*>(&KP[jj][4 * tr]);
            const float4 vv = *reinterpret_cast<const float4*>(&Vs[jj][4 * tc]);
            const float pa[4] = {pv.x, pv.y, pv.z, pv.w}, vb[4] = {vv.x, vv.y, vv.z, vv.w};
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) o[a][b] = __builtin_fmaf(pa[a], vb[b], o[a][b]);
        }
    }
    for (int a = 0; a < 4; ++a) {
        const int i = q0 + 4 * tr + a;
        if (i >= N) continue;
        for (int b = 0; b < 4; ++b) attn[static_cast<long long>(i) * D + hh * kHD + 4 * tc + b] = bf16_round(o[a][b] / lsum[a]);
    }
}

// Rotary on q and k (rotate-half within each head): one thread per (row, q|k, head, pair).
__global__ __launch_bounds__(kThreads) void rope_kernel(std::uint16_t* qkv, const float* cosv, const float* sinv, int N, int D, int heads, int hd) {
    const int half = hd / 2;
    const long long idx = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    const long long total = static_cast<long long>(N) * 2 * heads * half;
    if (idx >= total) return;
    const int j = static_cast<int>(idx % half), hh = static_cast<int>((idx / half) % heads), part = static_cast<int>((idx / half / heads) % 2);
    const int n = static_cast<int>(idx / half / heads / 2);
    std::uint16_t* v = qkv + static_cast<long long>(n) * 3 * D + static_cast<long long>(part) * D + static_cast<long long>(hh) * hd;
    float o1, o2;
    vision_rotate(bf16_float(v[j]), bf16_float(v[j + half]), cosv[static_cast<long long>(n) * half + j], sinv[static_cast<long long>(n) * half + j], o1, o2);
    v[j] = bf16_round(o1);
    v[j + half] = bf16_round(o2);
}

__global__ __launch_bounds__(kThreads) void add_kernel(std::uint16_t* x, const std::uint16_t* y, long long n) {
    const long long i = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    if (i < n) x[i] = bf16_round(bf16_float(x[i]) + bf16_float(y[i]));
}

__global__ __launch_bounds__(kThreads) void silu_mul_kernel(const std::uint16_t* gu, std::uint16_t* mid, int N, int I) {
    const long long i = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= static_cast<long long>(N) * I) return;
    const long long n = i / I, k = i % I;
    mid[i] = vision_silu_mul(gu[n * 2 * I + k], gu[n * 2 * I + I + k]);
}

__global__ __launch_bounds__(kThreads) void gelu_kernel(std::uint16_t* v, long long n) {
    const long long i = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    if (i < n) v[i] = bf16_round(vision_gelu(bf16_float(v[i])));
}

// 3 x 3 unfold: feature ch * 9 + ki * 3 + kj of cell (lh, lw) = patch (3 lh + ki, 3 lw + kj), zero beyond the grid.
__global__ __launch_bounds__(kThreads) void unfold_kernel(const std::uint16_t* h, std::uint16_t* unf, int vit_w, int vit_h, int llm_w, int llm_h, int D,
                                                          int M) {
    const long long F = static_cast<long long>(D) * M * M;
    const long long i = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= static_cast<long long>(llm_w) * llm_h * F) return;
    const long long cell = i / F, f = i % F;
    const int ch = static_cast<int>(f / (M * M)), ki = static_cast<int>((f / M) % M), kj = static_cast<int>(f % M);
    const int lh = static_cast<int>(cell / llm_w), lw = static_cast<int>(cell % llm_w);
    const int py = lh * M + ki, px = lw * M + kj;
    unf[i] = (py < vit_h && px < vit_w) ? h[(static_cast<long long>(py) * vit_w + px) * D + ch] : std::uint16_t{0};
}

// Span rows: START, (cell rows, NEWLINE) per row of cells, END.
__global__ __launch_bounds__(kThreads) void span_kernel(const std::uint16_t* rows, const std::uint16_t* start, const std::uint16_t* newline,
                                                        const std::uint16_t* end, std::uint16_t* out, int llm_w, int llm_h, int D) {
    const int span = llm_h * (llm_w + 1) + 2;
    const long long i = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= static_cast<long long>(span) * D) return;
    const int t = static_cast<int>(i / D), c = static_cast<int>(i % D);
    std::uint16_t v;
    if (t == 0) v = start[c];
    else if (t == span - 1) v = end[c];
    else {
        const int k = t - 1, row = k / (llm_w + 1), col = k % (llm_w + 1);
        v = col == llm_w ? newline[c] : rows[(static_cast<long long>(row) * llm_w + col) * D + c];
    }
    out[i] = v;
}

unsigned blocks_for(long long n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

}  // namespace

std::uint32_t max_cells(const VisionConfig& c) noexcept {
    std::uint32_t best = 0;
    for (std::uint32_t h = 1; h + 2U <= c.max_tokens; ++h) {
        const std::uint32_t w1 = (c.max_tokens - 2U) / h;  // llm_w + 1
        if (w1 < 2U) break;
        best = std::max(best, h * (w1 - 1U));
    }
    return best;
}
std::uint32_t max_patches(const VisionConfig& c) noexcept { return max_cells(c) * c.merge * c.merge; }

struct DeviceTower::Impl {
    VisionConfig cfg;
    hipStream_t stream = nullptr;
    std::vector<std::pair<std::string, VisionWeights::Tensor>> tensors;
    std::uint16_t* weights = nullptr;
    std::size_t weight_bytes = 0, work_bytes = 0;
    std::uint32_t max_n = 0, max_l = 0;
    bool own_stream = false;
    std::uint16_t* pool = nullptr;
    std::uint32_t pool_tokens = 0;
    // work buffers
    std::uint16_t *patches = nullptr, *x = nullptr, *h = nullptr, *qkv = nullptr, *attn = nullptr, *tmp = nullptr, *gu = nullptr, *mid = nullptr;
    std::uint16_t *unf = nullptr, *a1 = nullptr, *rows = nullptr;
    float *cosv = nullptr, *sinv = nullptr;
    void* pinned = nullptr;
    std::size_t pinned_bytes = 0;
    std::vector<void*> allocations;
    // STRIX_VISION_PROFILE=1: per-step device time (events around each step, summed over layers)
    bool profile = false;
    std::vector<std::pair<std::string, double>> prof;
    hipEvent_t ev0 = nullptr, ev1 = nullptr;
    void tick() {
        if (profile) check(hipEventRecord(ev0, stream), "event");
    }
    void tock(const char* name) {
        if (!profile) return;
        check(hipEventRecord(ev1, stream), "event");
        check(hipEventSynchronize(ev1), "event sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, ev0, ev1), "elapsed");
        for (auto& [n, t] : prof)
            if (n == name) {
                t += ms;
                return;
            }
        prof.emplace_back(name, ms);
    }

    const std::uint16_t* w(const std::string& name) const {
        for (const auto& [n, t] : tensors)
            if (n == name) return weights + t.offset;
        throw std::runtime_error("ds41 vision: no tensor " + name);
    }
    template <class T>
    T* alloc(std::size_t count) {
        void* p = nullptr;
        check(hipMalloc(&p, count * sizeof(T)), "hipMalloc");
        allocations.push_back(p);
        work_bytes += count * sizeof(T);
        return static_cast<T*>(p);
    }

    template <typename XT, bool F32, int TO>
    void gemm(const GemmArgs& a, unsigned batch = 1) {
        const dim3 grid(static_cast<unsigned>((a.O + 16 * TO - 1) / (16 * TO)), static_cast<unsigned>((a.M + kBM - 1) / kBM), batch);
        hipLaunchKernelGGL((gemm_kernel<XT, F32, TO>), grid, dim3(kThreads), 0, stream, a);
        check(hipGetLastError(), "gemm launch");
    }
    // y[M][O] = bf16(x[M][K] w[O][K]^T + bias), all contiguous
    void linear(const std::uint16_t* xp, int M, int K, const std::uint16_t* wp, const std::uint16_t* bias, int O, std::uint16_t* yp) {
        GemmArgs a{xp, K, 1, wp, K, 1, bias, yp, O, 1, M, O, K, 1.0F, 0, 0, 0};
        gemm<std::uint16_t, false, 8>(a);
    }
    void observe(const StepObserver& obs, const char* step, std::uint32_t layer, const std::uint16_t* p, std::size_t n) {
        if (!obs) return;
        std::vector<std::uint16_t> host(n);
        check(hipStreamSynchronize(stream), "sync");
        check(hipMemcpy(host.data(), p, n * 2U, hipMemcpyDeviceToHost), "download");
        obs(step, layer, host);
    }
};

DeviceTower::DeviceTower(const VisionWeights& weights, void* stream, std::uint32_t pool_tokens) : impl_(std::make_unique<Impl>()) {
    auto& m = *impl_;
    m.cfg = weights.config;
    m.stream = static_cast<hipStream_t>(stream);
    if (m.stream == nullptr) {
        check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking), "hipStreamCreate");
        m.own_stream = true;
    }
    m.tensors = weights.tensors;
    if (weights.blob.empty()) throw std::runtime_error("ds41 vision: weights not loaded");
    m.weight_bytes = weights.blob.size() * 2U;
    check(hipMalloc(reinterpret_cast<void**>(&m.weights), m.weight_bytes), "hipMalloc weights");
    check(hipMemcpy(m.weights, weights.blob.data(), m.weight_bytes, hipMemcpyHostToDevice), "upload weights");
    const auto& c = m.cfg;
    m.max_n = max_patches(c);
    m.max_l = max_cells(c);
    const std::size_t N = m.max_n, D = c.dim, I = c.inter, L = m.max_l;
    m.patches = m.alloc<std::uint16_t>(N * c.patch_in());
    m.x = m.alloc<std::uint16_t>(N * D);
    m.h = m.alloc<std::uint16_t>(N * D);
    m.qkv = m.alloc<std::uint16_t>(N * 3U * D);
    m.attn = m.alloc<std::uint16_t>(N * D);
    m.tmp = m.alloc<std::uint16_t>(N * D);
    m.gu = m.alloc<std::uint16_t>(N * 2U * I);
    m.mid = m.alloc<std::uint16_t>(N * I);
    m.cosv = m.alloc<float>(N * c.head_dim() / 2U);
    m.sinv = m.alloc<float>(N * c.head_dim() / 2U);
    m.unf = m.alloc<std::uint16_t>(L * c.merged_in());
    m.a1 = m.alloc<std::uint16_t>(L * c.out_dim);
    m.rows = m.alloc<std::uint16_t>(L * c.out_dim);
    if (pool_tokens != 0U) {
        m.pool_tokens = pool_tokens;
        check(hipMalloc(reinterpret_cast<void**>(&m.pool), std::size_t{pool_tokens} * c.out_dim * 2U), "hipMalloc pool");
    }
    m.pinned_bytes = N * c.patch_in() * 2U + 2U * N * (c.head_dim() / 2U) * 4U;
    check(hipHostMalloc(&m.pinned, m.pinned_bytes, hipHostMallocDefault), "hipHostMalloc");
    if (const char* e = std::getenv("STRIX_VISION_PROFILE"); e && *e == '1') {
        m.profile = true;
        check(hipEventCreate(&m.ev0), "event");
        check(hipEventCreate(&m.ev1), "event");
    }
}

DeviceTower::~DeviceTower() {
    if (!impl_) return;
    if (impl_->stream) (void)hipStreamSynchronize(impl_->stream);
    for (void* p : impl_->allocations) (void)hipFree(p);
    if (impl_->weights) (void)hipFree(impl_->weights);
    if (impl_->pinned) (void)hipHostFree(impl_->pinned);
    if (impl_->pool) (void)hipFree(impl_->pool);
    if (impl_->ev0) (void)hipEventDestroy(impl_->ev0);
    if (impl_->ev1) (void)hipEventDestroy(impl_->ev1);
    if (impl_->own_stream && impl_->stream) (void)hipStreamDestroy(impl_->stream);
}

std::uint16_t* DeviceTower::pool() const noexcept { return impl_->pool; }
std::uint32_t DeviceTower::pool_tokens() const noexcept { return impl_->pool_tokens; }
std::size_t DeviceTower::pool_bytes() const noexcept { return std::size_t{impl_->pool_tokens} * impl_->cfg.out_dim * 2U; }
void DeviceTower::synchronize() { check(hipStreamSynchronize(impl_->stream), "tower sync"); }

std::size_t DeviceTower::weight_bytes() const noexcept { return impl_->weight_bytes; }
std::size_t DeviceTower::work_bytes() const noexcept { return impl_->work_bytes; }
const VisionConfig& DeviceTower::config() const noexcept { return impl_->cfg; }

void DeviceTower::encode(const PreparedImage& img, std::uint16_t* out, const StepObserver& obs) {
    auto& m = *impl_;
    const auto& c = m.cfg;
    const auto& g = img.plan;
    const int N = static_cast<int>(g.patches()), D = static_cast<int>(c.dim), I = static_cast<int>(c.inter), H = static_cast<int>(c.heads);
    const int HD = static_cast<int>(c.head_dim()), half = HD / 2;
    if (HD != kHD) throw std::runtime_error("ds41 vision: the attention kernel is written for head dim 64");
    if (g.patches() > m.max_n || g.cells() > m.max_l) throw std::runtime_error("ds41 vision: picture larger than the tower's buffers");
    if (img.patches.size() != std::size_t{g.patches()} * c.patch_in()) throw std::runtime_error("ds41 vision: patch tensor size");
    // stage patches and rotary tables in pinned memory (the previous encode's uploads are done after this sync)
    check(hipStreamSynchronize(m.stream), "sync");
    std::vector<float> cosv, sinv;
    rope_tables(g, c, cosv, sinv);
    auto* pin = static_cast<std::uint8_t*>(m.pinned);
    const std::size_t pbytes = img.patches.size() * 2U, tbytes = cosv.size() * 4U;
    std::memcpy(pin, img.patches.data(), pbytes);
    std::memcpy(pin + pbytes, cosv.data(), tbytes);
    std::memcpy(pin + pbytes + tbytes, sinv.data(), tbytes);
    check(hipMemcpyAsync(m.patches, pin, pbytes, hipMemcpyHostToDevice, m.stream), "upload patches");
    check(hipMemcpyAsync(m.cosv, pin + pbytes, tbytes, hipMemcpyHostToDevice, m.stream), "upload cos");
    check(hipMemcpyAsync(m.sinv, pin + pbytes + tbytes, tbytes, hipMemcpyHostToDevice, m.stream), "upload sin");

    m.prof.clear();
    m.tick();
    m.linear(m.patches, N, static_cast<int>(c.patch_in()), m.w("vision.patch_embed.proj.weight"), m.w("vision.patch_embed.proj.bias"), D, m.x);
    m.tock("patch embed");
    m.observe(obs, "embed", 0, m.x, std::size_t(N) * D);
    const unsigned norm_blocks = static_cast<unsigned>((N + 7) / 8);
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const std::string b = "vision.blocks." + std::to_string(l) + ".";
        m.tick();
        hipLaunchKernelGGL(rms_norm_kernel, dim3(norm_blocks), dim3(kThreads), 0, m.stream, m.x, m.w(b + "norm1.weight"), m.h, N, D, c.eps);
        m.tock("norms");
        m.tick();
        m.linear(m.h, N, D, m.w(b + "attn.wqkv.weight"), m.w(b + "attn.wqkv.bias"), 3 * D, m.qkv);
        m.tock("qkv linear");
        m.tick();
        hipLaunchKernelGGL(rope_kernel, dim3(blocks_for(static_cast<long long>(N) * 2 * H * half)), dim3(kThreads), 0, m.stream, m.qkv, m.cosv, m.sinv, N, D,
                           H, HD);
        m.tock("rope");
        m.tick();
        hipLaunchKernelGGL(attention_kernel, dim3(static_cast<unsigned>((N + kQT - 1) / kQT), static_cast<unsigned>(H)), dim3(kThreads), 0, m.stream,
                           m.qkv, m.attn, N, D);
        m.tock("attention");
        m.tick();
        m.linear(m.attn, N, D, m.w(b + "attn.wo.weight"), m.w(b + "attn.wo.bias"), D, m.tmp);
        m.tock("wo linear");
        m.tick();
        hipLaunchKernelGGL(add_kernel, dim3(blocks_for(static_cast<long long>(N) * D)), dim3(kThreads), 0, m.stream, m.x, m.tmp, static_cast<long long>(N) * D);
        hipLaunchKernelGGL(rms_norm_kernel, dim3(norm_blocks), dim3(kThreads), 0, m.stream, m.x, m.w(b + "norm2.weight"), m.h, N, D, c.eps);
        m.tock("norms");
        m.tick();
        m.linear(m.h, N, D, m.w(b + "mlp.w1.weight"), nullptr, 2 * I, m.gu);
        m.tock("mlp w1");
        m.tick();
        hipLaunchKernelGGL(silu_mul_kernel, dim3(blocks_for(static_cast<long long>(N) * I)), dim3(kThreads), 0, m.stream, m.gu, m.mid, N, I);
        m.tock("silu");
        m.tick();
        m.linear(m.mid, N, I, m.w(b + "mlp.w2.weight"), nullptr, D, m.tmp);
        m.tock("mlp w2");
        hipLaunchKernelGGL(add_kernel, dim3(blocks_for(static_cast<long long>(N) * D)), dim3(kThreads), 0, m.stream, m.x, m.tmp, static_cast<long long>(N) * D);
        check(hipGetLastError(), "block kernels");
        m.observe(obs, "block", l, m.x, std::size_t(N) * D);
    }
    hipLaunchKernelGGL(rms_norm_kernel, dim3(norm_blocks), dim3(kThreads), 0, m.stream, m.x, m.w("vision.norm.weight"), m.h, N, D, c.eps);
    m.observe(obs, "norm", 0, m.h, std::size_t(N) * D);
    const int L = static_cast<int>(g.cells()), M = static_cast<int>(c.merge), F = static_cast<int>(c.merged_in()), OD = static_cast<int>(c.out_dim);
    hipLaunchKernelGGL(unfold_kernel, dim3(blocks_for(static_cast<long long>(L) * F)), dim3(kThreads), 0, m.stream, m.h, m.unf, static_cast<int>(g.vit_w),
                       static_cast<int>(g.vit_h), static_cast<int>(g.llm_w), static_cast<int>(g.llm_h), D, M);
    m.linear(m.unf, L, F, m.w("aligner.w1.weight"), m.w("aligner.w1.bias"), OD, m.a1);
    hipLaunchKernelGGL(gelu_kernel, dim3(blocks_for(static_cast<long long>(L) * OD)), dim3(kThreads), 0, m.stream, m.a1, static_cast<long long>(L) * OD);
    m.linear(m.a1, L, OD, m.w("aligner.w2.weight"), m.w("aligner.w2.bias"), OD, m.rows);
    m.observe(obs, "rows", 0, m.rows, std::size_t(L) * OD);
    if (m.profile) {
        check(hipStreamSynchronize(m.stream), "sync");
        double total = 0;
        for (const auto& [n, t] : m.prof) total += t;
        std::fprintf(stderr, "vision profile (%d patches):", N);
        for (const auto& [n, t] : m.prof) std::fprintf(stderr, " %s %.0f ms;", n.c_str(), t);
        std::fprintf(stderr, " total %.0f ms\n", total);
    }
    hipLaunchKernelGGL(span_kernel, dim3(blocks_for(static_cast<long long>(g.span_tokens()) * OD)), dim3(kThreads), 0, m.stream, m.rows, m.w("image_start"),
                       m.w("image_newline"), m.w("image_end"), out, static_cast<int>(g.llm_w), static_cast<int>(g.llm_h), OD);
    check(hipGetLastError(), "tower kernels");
}

std::vector<std::uint16_t> DeviceTower::encode_to_host(const PreparedImage& img, const StepObserver& obs) {
    const std::size_t n = std::size_t{img.plan.span_tokens()} * impl_->cfg.out_dim;
    std::uint16_t* d = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&d), n * 2U), "hipMalloc span");
    std::vector<std::uint16_t> host(n);
    try {
        encode(img, d, obs);
        check(hipStreamSynchronize(impl_->stream), "sync");
        check(hipMemcpy(host.data(), d, n * 2U, hipMemcpyDeviceToHost), "download span");
    } catch (...) {
        (void)hipFree(d);
        throw;
    }
    (void)hipFree(d);
    return host;
}

}  // namespace strix::deepseek41::vision
