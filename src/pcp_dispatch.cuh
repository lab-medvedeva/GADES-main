#ifndef PCP_DISPATCH_CUH
#define PCP_DISPATCH_CUH

// ============================================================================
// pcp_dispatch — host-only launch-configuration policy for the sparse Kendall
// per-cell-pair kernels.
//
// Derives the occupancy / shared-memory budget / warp count (cap_smem, W,
// smem_bytes, needs_fallback) from the DEVICE budget for a given heaviest-pair
// cap_global. This is the FibrocardRNA 87->15min dynamic-occupancy policy —
// isolated here as a small, testable interface.
//
// The dispatch KERNELS (RkendallPCP_Dispatch_*) and their __device__ helpers
// stay in metrics/kendall.cu (rdc constraint). pcp_dispatch_cfg only
// takes the kernels' host address (cudaFuncGetAttributes) to clamp W to the
// device register budget — a host reference resolved at host link, NOT a device
// call, so no -rdc=true.
// ============================================================================

#include <cstddef>

struct PcpDispatchCfg { int cap_smem; int W; size_t smem_bytes; bool needs_fallback; };

int    pcp_max_col_nnz(const int* csc_p, int n_cells);            // heaviest column nnz
void   pcp_grid_for_scratch(long long P, int cap, int* nblocks, int* nthreads_total);
int    pcp_warps_per_block(int cap);                             // variant-B warps from 96 KB
size_t pcp_device_smem_budget();                                // cached opt-in shared ceiling
PcpDispatchCfg pcp_dispatch_cfg(int cap_global);                // dynamic-occupancy dispatch cfg

#endif // PCP_DISPATCH_CUH
