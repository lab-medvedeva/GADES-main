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



__global__ void FinalizePearsonSparse(int columns, float* results, float* squares,
                                      float* col_sums, int rows) {
    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
    int column_index = blockDim.y * blockIdx.y + threadIdx.y;

    int index = row_index * columns + column_index;

    if ((row_index < columns) && (column_index < columns)) {
        if (row_index < column_index) {
            float dot_centered = results[index]
                - col_sums[row_index] * col_sums[column_index] / rows;
            float norm1 = sqrtf(squares[row_index]
                - col_sums[row_index] * col_sums[row_index] / rows);
            float norm2 = sqrtf(squares[column_index]
                - col_sums[column_index] * col_sums[column_index] / rows);
            results[index] = 1.0f - dot_centered / (norm1 * norm2);
        } else if (row_index > column_index) {
            results[index] = results[columns * column_index + row_index];
        }
    }
}


__global__ void compute_cell_sum_sumsq(
    const int* __restrict__ csc_p,
    const float* __restrict__ csc_x,
    int n_cells,
    float* __restrict__ sum_out,
    float* __restrict__ sumsq_out)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) return;
  float s = 0, s2 = 0;
  for (int k = csc_p[c]; k < csc_p[c + 1]; ++k) {
    float v = csc_x[k];
    s  += v;
    s2 += v * v;
  }
  sum_out[c]   = s;
  sumsq_out[c] = s2;
}


// ─── Pearson ───

__global__ void RpearsonSparse_per_cell_pair_same_block(
    const int* __restrict__ csc_p,
    const int* __restrict__ csc_i,
    const float* __restrict__ csc_x,
    const float* __restrict__ cell_sum,
    const float* __restrict__ cell_sumsq,
    int n_genes, int n_cells,
    float* __restrict__ result_out)
{
  int cell_a = blockIdx.y * blockDim.y + threadIdx.y;
  int cell_b = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_a >= cell_b || cell_b >= n_cells) return;

  int ia = csc_p[cell_a], ea = csc_p[cell_a + 1];
  int ib = csc_p[cell_b], eb = csc_p[cell_b + 1];

  float dot = 0;
  while (ia < ea && ib < eb) {
    int ga = csc_i[ia], gb = csc_i[ib];
    if      (ga < gb) ++ia;
    else if (gb < ga) ++ib;
    else { dot += csc_x[ia] * csc_x[ib]; ++ia; ++ib; }
  }

  float n  = (float)n_genes;
  float sa = cell_sum[cell_a],   sb = cell_sum[cell_b];
  float qa = cell_sumsq[cell_a], qb = cell_sumsq[cell_b];

  float cov = dot - sa * sb / n;
  float va  = qa  - sa * sa / n;
  float vb  = qb  - sb * sb / n;

  result_out[cell_a * n_cells + cell_b] = 1.0f - cov / sqrtf(va * vb);
}


__global__ void RpearsonSparse_per_cell_pair_different_blocks(
    const int* __restrict__ a_csc_p,
    const int* __restrict__ a_csc_i,
    const float* __restrict__ a_csc_x,
    const float* __restrict__ a_sum,
    const float* __restrict__ a_sumsq,
    const int* __restrict__ b_csc_p,
    const int* __restrict__ b_csc_i,
    const float* __restrict__ b_csc_x,
    const float* __restrict__ b_sum,
    const float* __restrict__ b_sumsq,
    int n_genes, int n_cells_a, int n_cells_b,
    float* __restrict__ result_out)
{
  int cell_a = blockIdx.y * blockDim.y + threadIdx.y;
  int cell_b = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_a >= n_cells_a || cell_b >= n_cells_b) return;

  int ia = a_csc_p[cell_a], ea = a_csc_p[cell_a + 1];
  int ib = b_csc_p[cell_b], eb = b_csc_p[cell_b + 1];

  float dot = 0;
  while (ia < ea && ib < eb) {
    int ga = a_csc_i[ia], gb = b_csc_i[ib];
    if      (ga < gb) ++ia;
    else if (gb < ga) ++ib;
    else { dot += a_csc_x[ia] * b_csc_x[ib]; ++ia; ++ib; }
  }

  float n  = (float)n_genes;
  float sa = a_sum[cell_a],   sb = b_sum[cell_b];
  float qa = a_sumsq[cell_a], qb = b_sumsq[cell_b];

  float cov = dot - sa * sb / n;
  float va  = qa  - sa * sa / n;
  float vb  = qb  - sb * sb / n;

  result_out[cell_b * n_cells_a + cell_a] = 1.0f - cov / sqrtf(va * vb);
}


// ─── Drivers: Pearson ───

extern "C" void matrix_Pearson_sparse_per_cell_pair_distance_same_block(
    int* csc_i_in, int* csc_p_in, double* csc_x_in,
    int* /*b*/, int* /*b*/, double* /*b*/,
    double* result, int* num_rows, int* num_columns,
    int* /*num_columns_b*/, int* num_elements, int* /*num_elements_b*/)
{
  int n_genes = *num_rows;
  int n_cells = *num_columns;
  int nnz     = *num_elements;

  std::vector<float> csc_x_f(nnz);
  for (int k = 0; k < nnz; ++k) csc_x_f[k] = (float)csc_x_in[k];

  int* d_i; int* d_p; float* d_x;
  float* d_sum; float* d_sumsq;
  float* d_res; double* d_out;
  cudaMalloc(&d_i, nnz * sizeof(int));
  cudaMalloc(&d_p, (n_cells + 1) * sizeof(int));
  cudaMalloc(&d_x, nnz * sizeof(float));
  cudaMalloc(&d_sum, n_cells * sizeof(float));
  cudaMalloc(&d_sumsq, n_cells * sizeof(float));
  cudaMalloc(&d_res, n_cells * n_cells * sizeof(float));
  cudaMalloc(&d_out, n_cells * n_cells * sizeof(double));

  cudaMemcpy(d_i, csc_i_in, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p, csc_p_in, (n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, csc_x_f.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);

  int prep_threads = 256;
  int prep_blocks = (n_cells + prep_threads - 1) / prep_threads;
  compute_cell_sum_sumsq<<<prep_blocks, prep_threads>>>(d_p, d_x, n_cells, d_sum, d_sumsq);
  gpuErrchk(cudaPeekAtLastError());

  dim3 threads(16, 16);
  dim3 blocks((n_cells + 15) / 16, (n_cells + 15) / 16);

  RpearsonSparse_per_cell_pair_same_block<<<blocks, threads>>>(
      d_p, d_i, d_x, d_sum, d_sumsq, n_genes, n_cells, d_res);
  gpuErrchk(cudaPeekAtLastError());

  FinalizePerCellPairFloat<<<blocks, threads>>>(n_cells, d_res, d_out);
  gpuErrchk(cudaPeekAtLastError());

  cudaMemcpy(result, d_out, n_cells * n_cells * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_i); cudaFree(d_p); cudaFree(d_x);
  cudaFree(d_sum); cudaFree(d_sumsq);
  cudaFree(d_res); cudaFree(d_out);
}


extern "C" void matrix_Pearson_sparse_per_cell_pair_distance_different_blocks(
    int* a_i_in, int* a_p_in, double* a_x_in,
    int* b_i_in, int* b_p_in, double* b_x_in,
    double* result, int* num_rows, int* num_columns,
    int* num_columns_b, int* num_elements_a, int* num_elements_b)
{
  int n_genes   = *num_rows;
  int n_cells_a = *num_columns;
  int n_cells_b = *num_columns_b;
  int nnz_a = *num_elements_a;
  int nnz_b = *num_elements_b;

  std::vector<float> a_xf(nnz_a), b_xf(nnz_b);
  for (int k = 0; k < nnz_a; ++k) a_xf[k] = (float)a_x_in[k];
  for (int k = 0; k < nnz_b; ++k) b_xf[k] = (float)b_x_in[k];

  int* d_ai; int* d_ap; float* d_ax; float* d_as; float* d_aq;
  int* d_bi; int* d_bp; float* d_bx; float* d_bs; float* d_bq;
  float* d_res;
  cudaMalloc(&d_ai, nnz_a * sizeof(int));
  cudaMalloc(&d_ap, (n_cells_a + 1) * sizeof(int));
  cudaMalloc(&d_ax, nnz_a * sizeof(float));
  cudaMalloc(&d_as, n_cells_a * sizeof(float));
  cudaMalloc(&d_aq, n_cells_a * sizeof(float));
  cudaMalloc(&d_bi, nnz_b * sizeof(int));
  cudaMalloc(&d_bp, (n_cells_b + 1) * sizeof(int));
  cudaMalloc(&d_bx, nnz_b * sizeof(float));
  cudaMalloc(&d_bs, n_cells_b * sizeof(float));
  cudaMalloc(&d_bq, n_cells_b * sizeof(float));
  cudaMalloc(&d_res, n_cells_a * n_cells_b * sizeof(float));

  cudaMemcpy(d_ai, a_i_in, nnz_a * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ap, a_p_in, (n_cells_a + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ax, a_xf.data(), nnz_a * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bi, b_i_in, nnz_b * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bp, b_p_in, (n_cells_b + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bx, b_xf.data(), nnz_b * sizeof(float), cudaMemcpyHostToDevice);

  int prep_threads = 256;
  compute_cell_sum_sumsq<<<(n_cells_a + prep_threads - 1) / prep_threads, prep_threads>>>(d_ap, d_ax, n_cells_a, d_as, d_aq);
  compute_cell_sum_sumsq<<<(n_cells_b + prep_threads - 1) / prep_threads, prep_threads>>>(d_bp, d_bx, n_cells_b, d_bs, d_bq);
  gpuErrchk(cudaPeekAtLastError());

  dim3 threads(16, 16);
  dim3 blocks((n_cells_b + 15) / 16, (n_cells_a + 15) / 16);

  RpearsonSparse_per_cell_pair_different_blocks<<<blocks, threads>>>(
      d_ap, d_ai, d_ax, d_as, d_aq,
      d_bp, d_bi, d_bx, d_bs, d_bq,
      n_genes, n_cells_a, n_cells_b, d_res);
  gpuErrchk(cudaPeekAtLastError());

  std::vector<float> h_res(n_cells_a * n_cells_b);
  cudaMemcpy(h_res.data(), d_res, n_cells_a * n_cells_b * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n_cells_a * n_cells_b; ++i) result[i] = (double)h_res[i];

  cudaFree(d_ai); cudaFree(d_ap); cudaFree(d_ax); cudaFree(d_as); cudaFree(d_aq);
  cudaFree(d_bi); cudaFree(d_bp); cudaFree(d_bx); cudaFree(d_bs); cudaFree(d_bq);
  cudaFree(d_res);
}


extern "C" void matrix_Pearson_distance_same_block(double* a, double* b, double* c, int* n, int* m, int* m_b) {
    pc_drive_cosine_same(a, c, *n, *m, /*center=*/true);
}


extern "C" void matrix_Pearson_distance_different_blocks(double* a, double* b, double* c, int* n, int* m, int* m_b) {
    pc_drive_cosine_diff(a, b, c, *n, *m, *m_b, /*center=*/true);
}


extern "C" void matrix_Pearson_sparse_distance_same_block(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
    pc_drive_sparse_cosine_same(a_index, a_positions, a_double_values, result,
                                *num_rows, *num_columns, *num_elements_a,
                                /*center=*/true);
}


extern "C" void matrix_Pearson_sparse_distance_different_blocks(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
    pc_drive_sparse_cosine_diff(a_index, a_positions, a_double_values,
                                b_index, b_positions, b_double_values,
                                result, *num_rows, *num_columns, *num_columns_b,
                                *num_elements_a, *num_elements_b,
                                /*center=*/true);
}
