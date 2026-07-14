// ============================================================================
// pcp_dispatch — definitions of the sparse-Kendall launch-config policy
// (see pcp_dispatch.cuh). Moved verbatim out of metrics/kendall.cu.
// ============================================================================

#include "pcp_dispatch.cuh"
#include <cstdlib>          // getenv, atoi
#include <cuda_runtime.h>

// The two sparse-Kendall dispatch kernels live in metrics/kendall.cu.
// pcp_dispatch_cfg only takes their host address (cudaFuncGetAttributes) to clamp
// W to the device register budget — a host reference to the launch stub resolved
// at host link (NO -rdc). Declared here with matching signatures.
__global__ void RkendallPCP_Dispatch_same_block(
    const int* __restrict__ csc_p, const int* __restrict__ csc_i, const float* __restrict__ csc_x,
    int n_genes, int n_cells, int cap_smem, int small_k,
    float* scratch_arena, int cap_global, int* __restrict__ discordant_out,
    unsigned long long* counters);
__global__ void RkendallPCP_Dispatch_different_blocks(
    const int* __restrict__ a_csc_p, const int* __restrict__ a_csc_i, const float* __restrict__ a_csc_x,
    const int* __restrict__ b_csc_p, const int* __restrict__ b_csc_i, const float* __restrict__ b_csc_x,
    int n_genes, int n_cells_a, int n_cells_b, int cap_smem, int small_k,
    float* scratch_arena, int cap_global, int* __restrict__ discordant_out,
    unsigned long long* counters);

int pcp_max_col_nnz(const int* csc_p, int n_cells) {
  int mx = 0;
  for (int c = 0; c < n_cells; ++c) { int nz = csc_p[c + 1] - csc_p[c]; if (nz > mx) mx = nz; }
  return mx;
}

// Persistent-thread grid sizing for A/Hybrid, bounded by a scratch budget.
void pcp_grid_for_scratch(long long P, int cap, int* nblocks, int* nthreads_total) {
  const size_t budget = (size_t)2 * 1024 * 1024 * 1024; // 2 GB scratch arena
  size_t per_thread = (size_t)4 * cap * sizeof(float);
  long long want = (long long)(budget / (per_thread > 0 ? per_thread : 1));
  if (want > P) want = P;
  const int tpb = 128;
  if (want < tpb) want = tpb;
  int nb = (int)((want + tpb - 1) / tpb);
  if (nb > 65535) nb = 65535;
  *nblocks = nb;
  *nthreads_total = nb * tpb;
}

// Warps per block from a shared-memory budget; opt-in up to 96 KB.
int pcp_warps_per_block(int cap) {
  const size_t smem_budget = 96 * 1024;
  size_t per_warp = (size_t)4 * cap * sizeof(float);
  int W = (int)(smem_budget / (per_warp > 0 ? per_warp : 1));
  if (W < 1) W = 1;
  if (W > 16) W = 16;
  return W;
}

size_t pcp_device_smem_budget() {
  static size_t cached = 0;
  if (cached) return cached;
  int dev = 0; cudaGetDevice(&dev);
  int v = 0;
  cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (v <= 0) cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlock, dev);
  if (v <= 0) v = 48 * 1024;
  cached = (size_t)v;
  return cached;
}

// Derive (cap_smem, W warps/block, dynamic smem bytes) from the device budget for
// a given heaviest-pair cap_global. cap_smem = min(cap_global, budget/16) so that
// when the whole tile fits the budget there is NEVER a fallback (identical to the
// old variant B, just device-sized); otherwise cap_smem saturates at the device
// ceiling and only genuinely oversized pairs fall back per-pair.
PcpDispatchCfg pcp_dispatch_cfg(int cap_global) {
  size_t budget = pcp_device_smem_budget();
  size_t usable = budget > 2048 ? budget - 2048 : budget / 2;   // margin for static smem
  int cap_smem_max = (int)(usable / (4 * sizeof(float)));        // one warp fills the budget
  if (cap_smem_max < 1) cap_smem_max = 1;
  int cap_smem = cap_global < cap_smem_max ? cap_global : cap_smem_max;
  const char* ev = getenv("HOBO_PCP_CAPSMEM");                   // test hook: force fallback
  bool cap_forced = false;
  if (ev && ev[0]) { int v = atoi(ev); if (v >= 1) { cap_smem = v < cap_global ? v : cap_global; cap_forced = true; } }
  if (cap_smem < 1) cap_smem = 1;
  int W = (int)(usable / ((size_t)4 * cap_smem * sizeof(float)));
  if (W < 1) W = 1; if (W > 32) W = 32;
  // Occupancy: a large cap_smem -- whether from a heavy tail OR just a moderate
  // cap_global (~2000-6000) -- leaves only W=1-2 warps/block => poor occupancy.
  // Since oversized pairs run warp-cooperatively in global scratch anyway (not 1
  // thread), it ALWAYS pays to shrink cap_smem to reach TARGET_W warps/block: the
  // small-k majority stays shared at high occupancy, larger-k pairs spill to
  // global-warp. (Previously gated on cap_smem<cap_global, which missed the
  // moderate-cap_global case, e.g. Anime where every block ran W=1 ~ 12s/block.)
  // Measured on Anime: monotone up to 16 then regresses -> default 16. Tunable
  // via HOBO_PCP_WARPS (1..32). s_k/s_nsf are [32], so W<=32 is safe.
  if (!cap_forced) {
    int target_W = 16;
    const char* ew = getenv("HOBO_PCP_WARPS");
    if (ew && ew[0]) { int v = atoi(ew); if (v >= 1 && v <= 32) target_W = v; }
    if (W < target_W) {
      W = target_W;
      cap_smem = (int)(usable / ((size_t)W * 4 * sizeof(float)));
      if (cap_smem < 1) cap_smem = 1;
      if (cap_smem > cap_global) cap_smem = cap_global;   // never exceed the heaviest pair
    }
  }
  // Device register budget. A small cap_smem (e.g. high sparsity => few nnz/col)
  // drives W to 32 => 32x32 = 1024 threads/block. On architectures with a smaller
  // per-block register file (e.g. sm_86) this register-heavy kernel then exceeds
  // the budget at launch ("too many resources requested for launch"). Query each
  // dispatch kernel's device-permitted max threads/block and clamp W to the
  // tighter of the two -- the same device-aware approach as the shared-memory
  // budget above. No-op on sm_120 (1024 fit); on sm_86 it lands W on the design
  // target. cap_smem is unchanged (fewer warps only frees shared memory).
  {
    cudaFuncAttributes fa;
    int max_threads = 32 * 32;
    if (cudaFuncGetAttributes(&fa, RkendallPCP_Dispatch_same_block) == cudaSuccess &&
        fa.maxThreadsPerBlock > 0 && fa.maxThreadsPerBlock < max_threads)
      max_threads = fa.maxThreadsPerBlock;
    if (cudaFuncGetAttributes(&fa, RkendallPCP_Dispatch_different_blocks) == cudaSuccess &&
        fa.maxThreadsPerBlock > 0 && fa.maxThreadsPerBlock < max_threads)
      max_threads = fa.maxThreadsPerBlock;
    int max_W = max_threads / 32; if (max_W < 1) max_W = 1;
    if (W > max_W) W = max_W;
  }
  PcpDispatchCfg c;
  c.cap_smem = cap_smem;
  c.W = W;
  c.smem_bytes = (size_t)W * 4 * cap_smem * sizeof(float);
  c.needs_fallback = (cap_smem < cap_global);
  return c;
}
