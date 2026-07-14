#include <time.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cublas_v2.h>
#include <R.h>
#include <Rinternals.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "pc_runtime.cuh"
#include "pc_linalg.cuh"
#include "pc_corr_core.cuh"


// ==================== cuBLAS-based Cosine / Pearson helpers ====================
//
// The four Cosine/Pearson dense drivers below were rewritten to use
// cublasSgemm: Pearson/Cosine reduce to A^T * A (or A^T * B) plus per-column
// L2 norms, which cuBLAS computes orders of magnitude faster than the original
// atomic-add kernels. See prototype_cublas/ for the validation harness.

// Column reductions, centering and gram-normalize kernels moved to pc_linalg.

// cuBLAS handles + compute-only kernel timer moved to pc_runtime.

// Core: D = cosine distance between columns of d_A (same-block), col-major m x m.
void pc_cosine_same_block_device(const float* d_A, int n, int m,
                                        float* d_D) {
    PcKernelTimer _kt;
    float* d_norms;
    cudaMalloc(&d_norms, m * sizeof(float));
    int t = 256;
    PCCol_sqnorm_kernel<<<m, t, t * sizeof(float)>>>(d_A, n, m, d_norms);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(pc_cublas_handle_tf32(), CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &alpha,
                d_A, n, d_A, n, &beta, d_D, m);

    dim3 tb(16, 16);
    dim3 gb((m + 15) / 16, (m + 15) / 16);
    PCNormalize_gram_same_kernel<<<gb, tb>>>(m, d_D, d_norms);
    cudaFree(d_norms);
}


// Core: D = cosine distance A vs B, col-major m x m_b.
void pc_cosine_different_blocks_device(const float* d_A,
                                              const float* d_B, int n, int m,
                                              int m_b, float* d_D) {
    PcKernelTimer _kt;
    float *d_xn, *d_yn;
    cudaMalloc(&d_xn, m * sizeof(float));
    cudaMalloc(&d_yn, m_b * sizeof(float));
    int t = 256;
    PCCol_sqnorm_kernel<<<m,   t, t * sizeof(float)>>>(d_A, n, m,   d_xn);
    PCCol_sqnorm_kernel<<<m_b, t, t * sizeof(float)>>>(d_B, n, m_b, d_yn);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(pc_cublas_handle_tf32(), CUBLAS_OP_T, CUBLAS_OP_N, m, m_b, n, &alpha,
                d_A, n, d_B, n, &beta, d_D, m);

    dim3 tb(16, 16);
    dim3 gb((m + 15) / 16, (m_b + 15) / 16);
    PCNormalize_gram_xy_kernel<<<gb, tb>>>(m, m_b, d_D, d_xn, d_yn);
    cudaFree(d_xn); cudaFree(d_yn);
}


// pc_center_columns_device and PCCol_sq_kernel moved to pc_linalg.

// D[i,j] = sqrt(max(sq[i] + sq[j] - 2*G[i,j], 0)); diagonal = 0.
// Column-major m x m. The max(...,0) guards against catastrophic cancellation
// when sq[i] ≈ sq[j] ≈ G[i,j] (nearly-identical columns). The extra threshold
// `v < float32_eps * (sq[i]+sq[j])` snaps residual float32 noise to 0 for
// truly-equal columns, where the naive sqrt would otherwise emit ~0.05 noise.
__global__ void PCEuclidean_finalize_same_kernel(int m, float* G,
                                                 const float* sq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= m) return;
    size_t idx = (size_t)i + (size_t)j * m;
    if (i == j) {
        G[idx] = 0.0f;
    } else {
        float si = sq[i], sj = sq[j];
        float v = si + sj - 2.0f * G[idx];
        // Float32 accumulated error in G[i,j] scales like sqrt(n)*eps*||x||*||y|| ≈
        // O(1e-5)*sqrt(si*sj) for typical scRNA-seq n. Clip residues at or
        // below that band — anything smaller is unrecoverable float32 noise
        // and should report 0.
        float floor_noise = 1.0e-5f * (si + sj);
        G[idx] = (v <= floor_noise) ? 0.0f : sqrtf(v);
    }
}


// Same but with separate sq norms for A and B (different_blocks).
__global__ void PCEuclidean_finalize_xy_kernel(int m, int m_b, float* G,
                                               const float* sq_a,
                                               const float* sq_b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= m_b) return;
    size_t idx = (size_t)i + (size_t)j * m;
    float si = sq_a[i], sj = sq_b[j];
    float v = si + sj - 2.0f * G[idx];
    float floor_noise = 1.0e-5f * (si + sj);
    G[idx] = (v <= floor_noise) ? 0.0f : sqrtf(v);
}


void pc_euclidean_same_block_device(const float* d_A, int n, int m,
                                           float* d_D) {
    PcKernelTimer _kt;
    float* d_sq;
    cudaMalloc(&d_sq, m * sizeof(float));
    int t = 256;
    PCCol_sq_kernel<<<m, t, t * sizeof(float)>>>(d_A, n, m, d_sq);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(pc_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &alpha,
                d_A, n, d_A, n, &beta, d_D, m);

    dim3 tb(16, 16);
    dim3 gb((m + 15) / 16, (m + 15) / 16);
    PCEuclidean_finalize_same_kernel<<<gb, tb>>>(m, d_D, d_sq);
    cudaFree(d_sq);
}


void pc_euclidean_different_blocks_device(const float* d_A,
                                                 const float* d_B, int n,
                                                 int m, int m_b, float* d_D) {
    PcKernelTimer _kt;
    float *d_sq_a, *d_sq_b;
    cudaMalloc(&d_sq_a, m   * sizeof(float));
    cudaMalloc(&d_sq_b, m_b * sizeof(float));
    int t = 256;
    PCCol_sq_kernel<<<m,   t, t * sizeof(float)>>>(d_A, n, m,   d_sq_a);
    PCCol_sq_kernel<<<m_b, t, t * sizeof(float)>>>(d_B, n, m_b, d_sq_b);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(pc_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, m, m_b, n, &alpha,
                d_A, n, d_B, n, &beta, d_D, m);

    dim3 tb(16, 16);
    dim3 gb((m + 15) / 16, (m_b + 15) / 16);
    PCEuclidean_finalize_xy_kernel<<<gb, tb>>>(m, m_b, d_D, d_sq_a, d_sq_b);
    cudaFree(d_sq_a); cudaFree(d_sq_b);
}


// PCCsr_to_dense_kernel and pc_sparse_to_dense_device moved to pc_linalg.

// R-facing driver glue: converts R's double* input to float, runs core, copies
// col-major m x m_b float back to R's double output.
void pc_drive_cosine_same(double* a, double* c, int n, int m,
                                 bool center) {
    size_t sz = (size_t)n * m;
    std::vector<float> h(sz);
    for (size_t i = 0; i < sz; ++i) h[i] = (float)a[i];

    float *d_A, *d_D;
    cudaMalloc(&d_A, sz * sizeof(float));
    cudaMalloc(&d_D, (size_t)m * m * sizeof(float));
    cudaMemcpy(d_A, h.data(), sz * sizeof(float), cudaMemcpyHostToDevice);
    if (center) pc_center_columns_device(d_A, n, m);
    pc_cosine_same_block_device(d_A, n, m, d_D);

    std::vector<float> out((size_t)m * m);
    cudaMemcpy(out.data(), d_D, (size_t)m * m * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < (size_t)m * m; ++i) c[i] = (double)out[i];
    cudaFree(d_A); cudaFree(d_D);
}


void pc_drive_cosine_diff(double* a, double* b, double* c, int n,
                                 int m, int m_b, bool center) {
    size_t szA = (size_t)n * m;
    size_t szB = (size_t)n * m_b;
    std::vector<float> hA(szA), hB(szB);
    for (size_t i = 0; i < szA; ++i) hA[i] = (float)a[i];
    for (size_t i = 0; i < szB; ++i) hB[i] = (float)b[i];

    float *d_A, *d_B, *d_D;
    cudaMalloc(&d_A, szA * sizeof(float));
    cudaMalloc(&d_B, szB * sizeof(float));
    cudaMalloc(&d_D, (size_t)m * m_b * sizeof(float));
    cudaMemcpy(d_A, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, hB.data(), szB * sizeof(float), cudaMemcpyHostToDevice);
    if (center) {
        pc_center_columns_device(d_A, n, m);
        pc_center_columns_device(d_B, n, m_b);
    }
    pc_cosine_different_blocks_device(d_A, d_B, n, m, m_b, d_D);

    std::vector<float> out((size_t)m * m_b);
    cudaMemcpy(out.data(), d_D, (size_t)m * m_b * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < (size_t)m * m_b; ++i) c[i] = (double)out[i];
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
}


// ─── Finalize: float upper-triangle → double symmetric matrix ───

__global__ void FinalizePerCellPairFloat(
    int n_cells,
    const float* __restrict__ in,
    double* __restrict__ out)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_cells || col >= n_cells) return;
  int idx = row * n_cells + col;
  if (row < col) {
    out[idx] = (double)in[idx];
  } else if (row > col) {
    out[idx] = (double)in[col * n_cells + row];
  } else {
    out[idx] = 0.0;
  }
}
