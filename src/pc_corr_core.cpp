#include <iostream>
#include <fstream>
#include <R.h>
#include <Rinternals.h>
#include <stdio.h>
#include <cstring>
#include <math.h>
#include <cassert>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <cblas.h>
#include "pc_runtime_cpu.h"
#include "pc_linalg_cpu.h"
#include "pc_corr_core_cpu.h"

//=========================================

// ==================== cblas_sgemm-based Cosine / Pearson helpers ====================
//
// The four Cosine/Pearson dense CPU drivers below were rewritten to use
// cblas_ssyrk (same-block, symmetric A^T A) and cblas_sgemm (different-blocks,
// A^T B). This gives 3–9x speedup over the original triple-loop version for
// realistic sizes. For tiny problems we fall back to a simple OMP loop,
// because OpenBLAS pays heavy pthread-pool overhead for small GEMMs.
// See prototype_cublas/ for the validation harness.

// pc_col_sqnorms / pc_center_columns moved to pc_linalg_cpu;
// pc_pick_blas_threads / PCBlasThreadsGuard moved to pc_runtime_cpu.

// Fallback: naive OMP triple loop — used for tiny problems where BLAS
// thread-pool overhead dominates. Output col-major m x m_b.
void pc_fallback_cosine_diff(const float* A, const float* B, int n,
                                    int m, int m_b, float* D,
                                    const float* xn, const float* yn) {
    #pragma omp parallel for schedule(dynamic)
    for (int c1 = 0; c1 < m; ++c1) {
        for (int c2 = 0; c2 < m_b; ++c2) {
            const float* p1 = A + (size_t)n * c1;
            const float* p2 = B + (size_t)n * c2;
            float s = 0.0f;
            for (int r = 0; r < n; ++r) s += p1[r] * p2[r];
            D[(size_t)c1 + (size_t)c2 * m] = 1.0f - s / (xn[c1] * yn[c2]);
        }
    }
}


void pc_fallback_cosine_same(const float* A, int n, int m, float* D,
                                    const float* norms) {
    #pragma omp parallel for schedule(dynamic)
    for (int c1 = 0; c1 < m; ++c1) {
        D[(size_t)c1 + (size_t)c1 * m] = 0.0f;
        for (int c2 = c1 + 1; c2 < m; ++c2) {
            const float* p1 = A + (size_t)n * c1;
            const float* p2 = A + (size_t)n * c2;
            float s = 0.0f;
            for (int r = 0; r < n; ++r) s += p1[r] * p2[r];
            float v = 1.0f - s / (norms[c1] * norms[c2]);
            D[(size_t)c1 + (size_t)c2 * m] = v;
            D[(size_t)c2 + (size_t)c1 * m] = v;
        }
    }
}


// pc_use_blas moved to pc_runtime_cpu.

// Core: cosine same_block, output col-major m x m.
void pc_cosine_same_block_cpu(const float* A, int n, int m, float* D) {
    std::vector<float> norms(m);
    pc_col_sqnorms(A, n, m, norms.data());

    long long flops = (long long)m * m * n;  // ssyrk is m*m*n (half of sgemm).
    if (!pc_use_blas(flops)) {
        pc_fallback_cosine_same(A, n, m, D, norms.data());
        return;
    }

    PCBlasThreadsGuard g(pc_pick_blas_threads(flops));
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans, m, n, 1.0f, A, n, 0.0f,
                D, m);

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < m; ++j) {
        D[(size_t)j + (size_t)j * m] = 0.0f;
        for (int i = 0; i < j; ++i) {
            float v = 1.0f - D[(size_t)i + (size_t)j * m] / (norms[i] * norms[j]);
            D[(size_t)i + (size_t)j * m] = v;
            D[(size_t)j + (size_t)i * m] = v;
        }
    }
}


// Core: cosine different_blocks, output col-major m x m_b.
void pc_cosine_different_blocks_cpu(const float* A, const float* B,
                                           int n, int m, int m_b, float* D) {
    std::vector<float> xn(m), yn(m_b);
    pc_col_sqnorms(A, n, m,   xn.data());
    pc_col_sqnorms(B, n, m_b, yn.data());

    long long flops = 2LL * m * m_b * n;
    if (!pc_use_blas(flops)) {
        pc_fallback_cosine_diff(A, B, n, m, m_b, D, xn.data(), yn.data());
        return;
    }

    PCBlasThreadsGuard g(pc_pick_blas_threads(flops));
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m_b, n, 1.0f, A, n,
                B, n, 0.0f, D, m);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < m_b; ++j) {
        for (int i = 0; i < m; ++i) {
            size_t idx = (size_t)i + (size_t)j * m;
            D[idx] = 1.0f - D[idx] / (xn[i] * yn[j]);
        }
    }
}


// R-facing glue: double* → float, run, float → double*.
void pc_drive_cpu_cosine_same(double* a, double* c, int n, int m,
                                     bool center) {
    size_t sz = (size_t)n * m;
    std::vector<float> h(sz);
    for (size_t i = 0; i < sz; ++i) h[i] = (float)a[i];
    if (center) pc_center_columns(h.data(), n, m);

    std::vector<float> D((size_t)m * m);
    pc_cosine_same_block_cpu(h.data(), n, m, D.data());
    for (size_t i = 0; i < (size_t)m * m; ++i) c[i] = (double)D[i];
}


// pc_col_sq moved to pc_linalg_cpu.

void pc_euclidean_same_block_cpu(const float* A, int n, int m,
                                        float* D) {
    std::vector<float> sq(m);
    pc_col_sq(A, n, m, sq.data());

    long long flops = (long long)m * m * n;
    if (!pc_use_blas(flops)) {
        #pragma omp parallel for schedule(dynamic)
        for (int c1 = 0; c1 < m; ++c1) {
            D[(size_t)c1 + (size_t)c1 * m] = 0.0f;
            for (int c2 = c1 + 1; c2 < m; ++c2) {
                const float* p1 = A + (size_t)n * c1;
                const float* p2 = A + (size_t)n * c2;
                float s = 0.0f;
                for (int r = 0; r < n; ++r) { float d = p1[r] - p2[r]; s += d * d; }
                float v = std::sqrt(s > 0.0f ? s : 0.0f);
                D[(size_t)c1 + (size_t)c2 * m] = v;
                D[(size_t)c2 + (size_t)c1 * m] = v;
            }
        }
        return;
    }

    PCBlasThreadsGuard g(pc_pick_blas_threads(flops));
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans, m, n, 1.0f, A, n, 0.0f,
                D, m);

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < m; ++j) {
        D[(size_t)j + (size_t)j * m] = 0.0f;
        for (int i = 0; i < j; ++i) {
            float si = sq[i], sj = sq[j];
            float v = si + sj - 2.0f * D[(size_t)i + (size_t)j * m];
            float floor_noise = 1.0e-5f * (si + sj);
            v = (v <= floor_noise) ? 0.0f : std::sqrt(v);
            D[(size_t)i + (size_t)j * m] = v;
            D[(size_t)j + (size_t)i * m] = v;
        }
    }
}


void pc_euclidean_different_blocks_cpu(const float* A, const float* B,
                                              int n, int m, int m_b,
                                              float* D) {
    std::vector<float> sq_a(m), sq_b(m_b);
    pc_col_sq(A, n, m,   sq_a.data());
    pc_col_sq(B, n, m_b, sq_b.data());

    long long flops = 2LL * m * m_b * n;
    if (!pc_use_blas(flops)) {
        #pragma omp parallel for schedule(dynamic)
        for (int c1 = 0; c1 < m; ++c1) {
            for (int c2 = 0; c2 < m_b; ++c2) {
                const float* p1 = A + (size_t)n * c1;
                const float* p2 = B + (size_t)n * c2;
                float s = 0.0f;
                for (int r = 0; r < n; ++r) { float d = p1[r] - p2[r]; s += d * d; }
                D[(size_t)c1 + (size_t)c2 * m] = std::sqrt(s > 0.0f ? s : 0.0f);
            }
        }
        return;
    }

    PCBlasThreadsGuard g(pc_pick_blas_threads(flops));
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m_b, n, 1.0f, A, n,
                B, n, 0.0f, D, m);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < m_b; ++j) {
        for (int i = 0; i < m; ++i) {
            size_t idx = (size_t)i + (size_t)j * m;
            float si = sq_a[i], sj = sq_b[j];
            float v = si + sj - 2.0f * D[idx];
            float floor_noise = 1.0e-5f * (si + sj);
            D[idx] = (v <= floor_noise) ? 0.0f : std::sqrt(v);
        }
    }
}


void pc_drive_cpu_cosine_diff(double* a, double* b, double* c, int n,
                                     int m, int m_b, bool center) {
    size_t szA = (size_t)n * m;
    size_t szB = (size_t)n * m_b;
    std::vector<float> hA(szA), hB(szB);
    for (size_t i = 0; i < szA; ++i) hA[i] = (float)a[i];
    for (size_t i = 0; i < szB; ++i) hB[i] = (float)b[i];
    if (center) {
        pc_center_columns(hA.data(), n, m);
        pc_center_columns(hB.data(), n, m_b);
    }

    std::vector<float> D((size_t)m * m_b);
    pc_cosine_different_blocks_cpu(hA.data(), hB.data(), n, m, m_b, D.data());
    for (size_t i = 0; i < (size_t)m * m_b; ++i) c[i] = (double)D[i];
}
