#ifndef PC_RUNTIME_CUH
#define PC_RUNTIME_CUH

// ============================================================================
// pc_runtime — shared GPU runtime state behind a small interface.
//
// Owns the cross-translation-unit mutable state that every metric path depends
// on: the two cuBLAS handles, the compute-only kernel-time accumulator, and the
// persistent pinned staging pool. The mutable state is DEFINED once in
// pc_runtime.cu and only declared here, so every TU that includes this header
// shares one handle / one timer / one pinned pool instead of a private copy.
//
// Thread-safety contract: single-threaded R driver ONLY. The cuBLAS handle
// singletons and the pinned pool are unlocked globals; concurrent host callers
// would race. GADES deploys as one R process with one driver thread (ADR-0002).
// ============================================================================

#include <cstdlib>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <R.h>
#include <Rinternals.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// ---- CUDA error check ------------------------------------------------------
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// ---- cuBLAS handles (lazy-init singletons; DEFINED in pc_runtime.cu) --------
//
//  pc_cublas_handle()       — PEDANTIC_MATH (TF32 OFF). Used by EUCLIDEAN only:
//    `sq[i] + sq[j] - 2*G` cancels catastrophically when G comes from TF32.
//  pc_cublas_handle_tf32()  — TF32 tensor-core SGEMM. Used by COSINE/PEARSON
//    (and SPEARMAN, cosine-on-ranks): 1 - normalized dot products in [-1,1],
//    well-conditioned, so TF32's ~10-bit mantissa is fine and ~3x faster.
cublasHandle_t& pc_cublas_handle();
cublasHandle_t& pc_cublas_handle_tf32();

// ---- compute-only kernel timer ---------------------------------------------
// Accumulates GPU kernel time across ALL launches within one mtrx_distance call
// via CUDA events (batch-robust). Off unless env HOBO_KERNEL_US is set -> zero
// overhead in normal runs. Reset/read from R via C_kernel_us_reset/get().
extern double g_pc_kernel_us;
inline int pc_kernel_timing_on() {
    static int on = -1;
    if (on < 0) { const char* e = getenv("HOBO_KERNEL_US"); on = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return on;
}
// RAII: declare `PcKernelTimer _kt;` as the first line of a device-compute
// function. ctor records a start event; dtor records stop, syncs, ADDS the
// elapsed kernel time to g_pc_kernel_us. Block-loop/streaming totals sum right.
struct PcKernelTimer {
    cudaEvent_t s, e; bool on;
    PcKernelTimer() : s(0), e(0), on(pc_kernel_timing_on() != 0) {
        if (on) { cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s); }
    }
    ~PcKernelTimer() {
        if (on) {
            cudaEventRecord(e); cudaEventSynchronize(e);
            float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
            g_pc_kernel_us += (double)ms * 1000.0;
            cudaEventDestroy(s); cudaEventDestroy(e);
        }
    }
};

// ---- host buffer / conversion helpers (pure, no persistent state) ----------
// Pinned memory ~2x's the PCIe H2D/D2H bandwidth vs pageable. Caller frees a
// pc_pinned_f/pc_host_d2f buffer with cudaFreeHost.
inline float* pc_pinned_f(size_t sz) {
    float* p = nullptr;
    cudaMallocHost((void**)&p, sz * sizeof(float));
    return p;
}
inline float* pc_host_d2f(const double* src, size_t sz) {
    float* h = pc_pinned_f(sz);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)sz; ++i) h[i] = (float)src[i];
    return h;
}
inline void pc_host_f2d(const float* src, double* dst, size_t sz) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)sz; ++i) dst[i] = (double)src[i];
}
inline bool pc_rt_log() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("HOBO_RT_LOG"); v = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return v;
}

// ---- persistent pinned staging pool ----------------------------------------
// Grown on demand, never freed (persist for process lifetime), reused across
// calls. Globals are extern here (single-thread contract) so the .Call batched
// paths can still address &g_pin_in / &g_pin_in_b directly. Accessors DEFINED in
// pc_runtime.cu.
extern float* g_pin_in;    extern size_t g_pin_in_sz;
extern float* g_pin_out;   extern size_t g_pin_out_sz;
extern float* g_pin_in_b;  extern size_t g_pin_in_b_sz;

float* pc_pin(float** buf, size_t* cur, size_t sz);      // generic grow-if-needed
float* pc_d2f_pin(const double* src, size_t sz);         // -> g_pin_in
float* pc_d2f_pin_b(const double* src, size_t sz);       // -> g_pin_in_b
float* pc_srcblk_pin(SEXP a, size_t off, size_t sz, float** buf, size_t* cur);

#endif // PC_RUNTIME_CUH
