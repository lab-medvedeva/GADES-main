// ============================================================================
// pc_runtime — definitions of the shared GPU runtime state (see pc_runtime.cuh).
// Exactly ONE definition per mutable symbol lives here; metric TUs see only the
// accessors / extern decls in the header.
// ============================================================================

#include "pc_runtime.cuh"

// ---- cuBLAS handles --------------------------------------------------------
cublasHandle_t& pc_cublas_handle() {
    static cublasHandle_t h = nullptr;
    if (h == nullptr) {
        cublasCreate(&h);
        cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH);   // TF32 OFF (euclidean fp32)
    }
    return h;
}
cublasHandle_t& pc_cublas_handle_tf32() {
    static cublasHandle_t h = nullptr;
    if (h == nullptr) {
        cublasCreate(&h);
        cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return h;
}

// ---- compute-only kernel timer accumulator ---------------------------------
double g_pc_kernel_us = 0.0;

// ---- persistent pinned staging pool ----------------------------------------
float* g_pin_in = nullptr;    size_t g_pin_in_sz  = 0;
float* g_pin_out = nullptr;   size_t g_pin_out_sz = 0;
float* g_pin_in_b = nullptr;  size_t g_pin_in_b_sz = 0;

float* pc_pin(float** buf, size_t* cur, size_t sz) {
    if (sz > *cur) { if (*buf) cudaFreeHost(*buf); cudaMallocHost((void**)buf, sz * sizeof(float)); *cur = sz; }
    return *buf;
}
float* pc_d2f_pin(const double* src, size_t sz) {
    float* h = pc_pin(&g_pin_in, &g_pin_in_sz, sz);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)sz; ++i) h[i] = (float)src[i];
    return h;
}
float* pc_d2f_pin_b(const double* src, size_t sz) {
    float* h = pc_pin(&g_pin_in_b, &g_pin_in_b_sz, sz);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)sz; ++i) h[i] = (float)src[i];
    return h;
}
// Type-aware column-block -> float pinned, reading DIRECTLY from the R SEXP
// (REALSXP double or INTSXP integer) to skip R's as.double() allocate-and-convert.
float* pc_srcblk_pin(SEXP a, size_t off, size_t sz, float** buf, size_t* cur) {
    float* h = pc_pin(buf, cur, sz);
    if (TYPEOF(a) == INTSXP) {
        const int* s = INTEGER(a) + off;
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)sz; ++i) h[i] = (float)s[i];
    } else {
        const double* s = REAL(a) + off;
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)sz; ++i) h[i] = (float)s[i];
    }
    return h;
}

// ---- compute-only kernel timer accessors (R-facing) ------------------------
//   .Call("C_kernel_us_reset"); <run mtrx_distance with HOBO_KERNEL_US=1>;
//   kernel_us <- .Call("C_kernel_us_get")
extern "C" SEXP C_kernel_us_reset() { g_pc_kernel_us = 0.0; return R_NilValue; }
extern "C" SEXP C_kernel_us_get() {
    SEXP out = PROTECT(allocVector(REALSXP, 1));
    REAL(out)[0] = g_pc_kernel_us;
    UNPROTECT(1);
    return out;
}
