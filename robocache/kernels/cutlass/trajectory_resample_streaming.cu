// Streaming Resampler Kernel for H100 (SM90)
// Two-pointer scan + vectorized BF16 lerp + coalesced memory access
// Target: 22x-500x speedup by fixing memory access patterns

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// Tunables
constexpr int TT         = 1024;   // targets per CTA tile
constexpr int THREADS    = 256;    // threads per CTA (8 warps)
constexpr int VECLEN     = 8;      // features per vector step (16B for BF16)
constexpr int STAGES     = 2;      // double buffer for copy/compute overlap
constexpr float EPS_DT   = 1e-20f; // avoid divide-by-zero

// BF16 <-> FP32 conversions
__device__ inline float b2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ inline __nv_bfloat16 f2b(float x) { return __float2bfloat16_rn(x); }

// Vectorized load/store helpers (16B at a time)
struct bf16x8 { __nv_bfloat16 v[8]; }; // 8 * 2B = 16B

__device__ inline void gmem_load_bf16x8(const __nv_bfloat16* gptr, int d, bf16x8& r) {
    const bf16x8* p = reinterpret_cast<const bf16x8*>(gptr + d);
    r = *p;
}

__device__ inline void gmem_store_bf16x8(__nv_bfloat16* gptr, int d, const bf16x8& r) {
    bf16x8* p = reinterpret_cast<bf16x8*>(gptr + d);
    *p = r;
}

template <int TT_PARAM, int VECLEN_PARAM, int STAGES_PARAM>
__global__ void streaming_resampler_cccl_kernel(
    const __nv_bfloat16* __restrict__ src,   // [B,S,D]
    const float*         __restrict__ st,    // [B,S]
    const float*         __restrict__ tt,    // [B,T]
    __nv_bfloat16*       __restrict__ out,   // [B,T,D]
    int S, int T, int D)                     // sizes (B in grid.x)
{
    extern __shared__ unsigned char smem_raw[];
    
    // Layout SMEM: times chunk + feature double buffer
    float* s_times = reinterpret_cast<float*>(smem_raw);
    int TIMES_CHUNK = min(S, 4096);
    __nv_bfloat16* s_feat = reinterpret_cast<__nv_bfloat16*>(s_times + TIMES_CHUNK);
    size_t feat_stage_stride = 2 * D; // [row_i, row_ip1] contiguous

    const int b   = blockIdx.x;
    const int tj  = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int t0  = tj * TT_PARAM;
    const int t1  = min(T, t0 + TT_PARAM);

    // Pointers to batch slices
    const float*         st_b  = st  + (size_t)b * S;
    const float*         tt_b  = tt  + (size_t)b * T;
    const __nv_bfloat16* src_b = src + (size_t)b * S * D;
    __nv_bfloat16*       out_b = out + (size_t)b * T * D;

    // Thread block for synchronization
    auto block = cg::this_thread_block();

    // Stage source times chunks (coalesced, warp-cooperative)
    int s_base = 0;
    
    while (s_base < S) {
        const int chunk = min(TIMES_CHUNK, S - s_base);
        
        // Cooperative load times into SMEM (coalesced 128B)
        for (int idx = tid; idx < chunk; idx += blockDim.x) {
            s_times[idx] = st_b[s_base + idx];
        }
        __syncthreads();

        // Two-pointer merge across this times chunk and target tile
        int i = 0;  // local index into s_times
        
        // Find starting interval covering tt_b[t0] (once per CTA tile)
        if (s_base == 0 && t0 < T) {
            float tstart = tt_b[t0];
            // Linear advance to find first interval containing tstart
            while (i + 1 < chunk && s_times[i+1] < tstart) {
                ++i;
            }
        }

        // Stream intervals in this chunk
        int k = t0;  // walk targets in [t0, t1)
        
        while (i + 1 < chunk && k < t1) {
            float si   = s_times[i];
            float sip1 = s_times[i+1];

            // Find contiguous target range [k_lo, k_hi) in interval [si, sip1]
            int k_lo = k;
            while (k < t1 && tt_b[k] <= sip1) ++k;
            int k_hi = k;

            if (k_lo < k_hi) {  // only process if targets exist in this interval
                // Load two feature rows for this interval (coalesced vectorized loads)
                int stage = (i & (STAGES_PARAM - 1));
                __nv_bfloat16* s_row_i   = s_feat + stage * feat_stage_stride + 0 * D;
                __nv_bfloat16* s_row_ip1 = s_feat + stage * feat_stage_stride + 1 * D;

                // Coalesced vectorized loads over D (128B transactions)
                for (int d = tid * VECLEN_PARAM; d < D; d += blockDim.x * VECLEN_PARAM) {
                    if (d + VECLEN_PARAM <= D) {
                        // Row i
                        bf16x8 r0;
                        gmem_load_bf16x8(src_b + (size_t)(s_base + i) * D, d, r0);
                        *reinterpret_cast<bf16x8*>(s_row_i + d) = r0;

                        // Row i+1
                        bf16x8 r1;
                        gmem_load_bf16x8(src_b + (size_t)(s_base + i + 1) * D, d, r1);
                        *reinterpret_cast<bf16x8*>(s_row_ip1 + d) = r1;
                    }
                }
                __syncthreads();

                // Compute lerp for all targets in this interval (vectorized over D)
                float inv_dt = 1.0f / fmaxf(sip1 - si, EPS_DT);

                // Distribute targets across warps
                for (int kk = k_lo + warp; kk < k_hi; kk += (blockDim.x >> 5)) {
                    float t = tt_b[kk];
                    float a = (t - si) * inv_dt;  // alpha in [0,1]
                    float b_val = 1.0f - a;

                    // Vectorized over feature dimension
                    for (int d = lane * VECLEN_PARAM; d < D; d += 32 * VECLEN_PARAM) {
                        if (d + VECLEN_PARAM <= D) {
                            bf16x8 vi = *reinterpret_cast<const bf16x8*>(s_row_i   + d);
                            bf16x8 vj = *reinterpret_cast<const bf16x8*>(s_row_ip1 + d);
                            bf16x8 vo;

                            // Compute in FP32, store BF16
                            #pragma unroll
                            for (int u = 0; u < VECLEN_PARAM; ++u) {
                                float xi = b2f(vi.v[u]);
                                float xj = b2f(vj.v[u]);
                                float y  = fmaf(a, (xj - xi), xi);  // y = xi + a*(xj - xi)
                                vo.v[u]  = f2b(y);
                            }

                            gmem_store_bf16x8(out_b + (size_t)kk * D, d, vo);
                        }
                    }
                }
                __syncthreads();
            }

            ++i;  // next interval
        }

        s_base += chunk - 1;  // overlap last time as first time of next chunk
        if (s_base >= S - 1) break;
    }
}

// Launch helper
extern "C" cudaError_t launch_trajectory_resample_streaming(
    const void* src, const float* st, const float* tt, void* out,
    int B, int S, int T, int D, cudaStream_t stream)
{
    if (D % VECLEN != 0) {
        return cudaErrorInvalidValue;  // TODO: add tail handling
    }

    dim3 grid(B, (T + TT - 1) / TT);
    dim3 block(THREADS);
    
    size_t smem_bytes =
        /* times */ sizeof(float) * min(S, 4096)
      + /* feats */ sizeof(__nv_bfloat16) * (size_t)(STAGES * 2 * D);

    streaming_resampler_cccl_kernel<TT, VECLEN, STAGES>
        <<<grid, block, smem_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(src), st, tt,
            static_cast<__nv_bfloat16*>(out), S, T, D);

    return cudaGetLastError();
}

