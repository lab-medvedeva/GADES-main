#ifndef PC_RUNTIME_CPU_H
#define PC_RUNTIME_CPU_H

// ============================================================================
// pc_runtime_cpu — CPU-side runtime policy (the CPU analog of pc_runtime).
//
// Unlike the GPU side there is NO cached resource handle to own: the only shared
// global state is OpenBLAS's process-global thread count, and that is
// save/restored per driver call by PCBlasThreadsGuard (never cached). So this
// header is stateless BLAS/OpenMP policy + work-splitting helpers (ADR-0002).
// The CPU RAM guard lives in R (memory_limit_gb), not here.
// ============================================================================

#include <algorithm>
#include <omp.h>
#include <cblas.h>   // OpenBLAS: openblas_get/set_num_threads

// ---- BLAS thread / dispatch policy -----------------------------------------
// Pick a sane BLAS thread count for this problem. OpenBLAS pays heavy pthread
// overhead for small GEMMs — shrink the pool rather than eat it.
static inline int pc_pick_blas_threads(long long flops) {
    int hw = openblas_get_num_threads();
    if (flops < 20000000LL)   return 1;
    if (flops < 200000000LL)  return std::min(hw, 4);
    if (flops < 2000000000LL) return std::min(hw, 8);
    return hw;
}

// RAII: set the OpenBLAS thread count for the current scope, restore on exit.
struct PCBlasThreadsGuard {
    int saved;
    PCBlasThreadsGuard(int n) : saved(openblas_get_num_threads()) {
        openblas_set_num_threads(n);
    }
    ~PCBlasThreadsGuard() { openblas_set_num_threads(saved); }
};

// Threshold below which BLAS is a loss: pure dot-product work is memory-bound
// and naive OMP beats sgemm's launch overhead.
static inline bool pc_use_blas(long long flops) { return flops >= 10000000LL; }

// ---- OpenMP work-splitting -------------------------------------------------
template<typename F>
static void parallel_for_with_id(int num_items, int num_threads, F&& worker) {
    if (num_threads > num_items) num_threads = num_items;
    #pragma omp parallel num_threads(num_threads)
    {
        int t = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int chunk = num_items / nt;
        int remainder = num_items % nt;
        int start = 0;
        for (int i = 0; i < t; ++i) start += chunk + (i < remainder ? 1 : 0);
        int end = start + chunk + (t < remainder ? 1 : 0);
        worker(t, start, end);
    }
}

static inline int get_num_threads(int num_items) {
    int nt = omp_get_max_threads();
    if (nt <= 0) nt = 1;
    if (nt > num_items) nt = num_items;
    return nt;
}

template<typename F>
static void parallel_for_cols(int num_cols, F&& worker) {
    int num_threads = omp_get_max_threads();
    if (num_threads <= 0) num_threads = 1;
    if (num_threads > num_cols) num_threads = num_cols;

    #pragma omp parallel num_threads(num_threads)
    {
        int t = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int chunk = num_cols / nt;
        int remainder = num_cols % nt;
        int start = 0;
        for (int i = 0; i < t; ++i) start += chunk + (i < remainder ? 1 : 0);
        int end = start + chunk + (t < remainder ? 1 : 0);
        worker(start, end);
    }
}

#endif // PC_RUNTIME_CPU_H
