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



// ─── Spearman per_cell_pair ───
//
// Host-side preprocessing produces v_a[g] = centered_rank_a[g] - zr_a stored
// in csc_v (same layout as csc_x). Kernel performs Cosine-style merge over
// nnz∩nnz, then corrects by -n_genes·zr_a·zr_b. norm_sq accounts for all
// n_genes positions (active + zero-gap). See main.cpp for derivation.

static void spearman_per_cell_pair_preprocess_host(
    const int* csc_p, const double* csc_x_in,
    int n_genes, int n_cells,
    float* v_out, float* zr_out, float* norm_sq_out)
{
    float mean_rank = (n_genes + 1) / 2.0f;
    std::vector<std::pair<float, int>> entries;
    for (int c = 0; c < n_cells; ++c) {
        int start = csc_p[c];
        int end   = csc_p[c + 1];
        int nnz   = end - start;

        entries.resize(nnz);
        for (int k = 0; k < nnz; ++k) {
            entries[k] = { static_cast<float>(csc_x_in[start + k]), k };
        }
        std::sort(entries.begin(), entries.end());

        int neg_count = 0, explicit_zero_count = 0;
        for (int k = 0; k < nnz; ++k) {
            if (entries[k].first < 0.0f) neg_count++;
            else if (entries[k].first == 0.0f) explicit_zero_count++;
        }
        int total_zeros = (n_genes - nnz) + explicit_zero_count;

        float zr = 0.0f;
        if (total_zeros > 0) {
            zr = (neg_count + 1 + neg_count + total_zeros) / 2.0f - mean_rank;
        }
        zr_out[c] = zr;

        float active_sq = 0.0f;

        int i = 0;
        int global_pos = 0;
        while (i < neg_count) {
            int eq = i + 1;
            while (eq < neg_count && entries[eq].first == entries[i].first) ++eq;
            float rank = (global_pos + 1 + global_pos + (eq - i)) / 2.0f - mean_rank;
            float vshift = rank - zr;
            for (int k = i; k < eq; ++k) {
                v_out[start + entries[k].second] = vshift;
                active_sq += rank * rank;
            }
            global_pos += (eq - i);
            i = eq;
        }

        for (int k = neg_count; k < neg_count + explicit_zero_count; ++k) {
            v_out[start + entries[k].second] = 0.0f;
            active_sq += zr * zr;
        }

        global_pos = neg_count + total_zeros;
        i = neg_count + explicit_zero_count;
        while (i < nnz) {
            int eq = i + 1;
            while (eq < nnz && entries[eq].first == entries[i].first) ++eq;
            float rank = (global_pos + 1 + global_pos + (eq - i)) / 2.0f - mean_rank;
            float vshift = rank - zr;
            for (int k = i; k < eq; ++k) {
                v_out[start + entries[k].second] = vshift;
                active_sq += rank * rank;
            }
            global_pos += (eq - i);
            i = eq;
        }

        int implicit_zeros = n_genes - nnz;
        norm_sq_out[c] = active_sq + (float)implicit_zeros * zr * zr;
    }
}


__global__ void RspearmanSparse_per_cell_pair_same_block(
    const int* __restrict__ csc_p,
    const int* __restrict__ csc_i,
    const float* __restrict__ csc_v,
    const float* __restrict__ zr,
    const float* __restrict__ norm_sq,
    int n_genes, int n_cells,
    float* __restrict__ result_out)
{
  int cell_a = blockIdx.y * blockDim.y + threadIdx.y;
  int cell_b = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_a >= cell_b || cell_b >= n_cells) return;

  int ia = csc_p[cell_a], ea = csc_p[cell_a + 1];
  int ib = csc_p[cell_b], eb = csc_p[cell_b + 1];

  float dot_v = 0.0f;
  while (ia < ea && ib < eb) {
    int ga = csc_i[ia], gb = csc_i[ib];
    if      (ga < gb) ++ia;
    else if (gb < ga) ++ib;
    else { dot_v += csc_v[ia] * csc_v[ib]; ++ia; ++ib; }
  }
  float dot = dot_v - (float)n_genes * zr[cell_a] * zr[cell_b];
  result_out[cell_a * n_cells + cell_b] =
      1.0f - dot / sqrtf(norm_sq[cell_a] * norm_sq[cell_b]);
}


__global__ void RspearmanSparse_per_cell_pair_different_blocks(
    const int* __restrict__ a_csc_p,
    const int* __restrict__ a_csc_i,
    const float* __restrict__ a_csc_v,
    const float* __restrict__ a_zr,
    const float* __restrict__ a_norm_sq,
    const int* __restrict__ b_csc_p,
    const int* __restrict__ b_csc_i,
    const float* __restrict__ b_csc_v,
    const float* __restrict__ b_zr,
    const float* __restrict__ b_norm_sq,
    int n_genes, int n_cells_a, int n_cells_b,
    float* __restrict__ result_out)
{
  int cell_a = blockIdx.y * blockDim.y + threadIdx.y;
  int cell_b = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_a >= n_cells_a || cell_b >= n_cells_b) return;

  int ia = a_csc_p[cell_a], ea = a_csc_p[cell_a + 1];
  int ib = b_csc_p[cell_b], eb = b_csc_p[cell_b + 1];

  float dot_v = 0.0f;
  while (ia < ea && ib < eb) {
    int ga = a_csc_i[ia], gb = b_csc_i[ib];
    if      (ga < gb) ++ia;
    else if (gb < ga) ++ib;
    else { dot_v += a_csc_v[ia] * b_csc_v[ib]; ++ia; ++ib; }
  }
  float dot = dot_v - (float)n_genes * a_zr[cell_a] * b_zr[cell_b];
  result_out[cell_b * n_cells_a + cell_a] =
      1.0f - dot / sqrtf(a_norm_sq[cell_a] * b_norm_sq[cell_b]);
}


extern "C" void matrix_Spearman_sparse_per_cell_pair_distance_same_block(
    int* csc_i_in, int* csc_p_in, double* csc_x_in,
    int* /*b*/, int* /*b*/, double* /*b*/,
    double* result, int* num_rows, int* num_columns,
    int* /*num_columns_b*/, int* num_elements, int* /*num_elements_b*/)
{
  int n_genes = *num_rows;
  int n_cells = *num_columns;
  int nnz     = *num_elements;

  std::vector<float> v_host(nnz);
  std::vector<float> zr_host(n_cells);
  std::vector<float> nsq_host(n_cells);
  spearman_per_cell_pair_preprocess_host(
      csc_p_in, csc_x_in, n_genes, n_cells,
      v_host.data(), zr_host.data(), nsq_host.data());

  int* d_i; int* d_p; float* d_v; float* d_zr; float* d_nsq;
  float* d_res; double* d_out;
  cudaMalloc(&d_i, nnz * sizeof(int));
  cudaMalloc(&d_p, (n_cells + 1) * sizeof(int));
  cudaMalloc(&d_v, nnz * sizeof(float));
  cudaMalloc(&d_zr, n_cells * sizeof(float));
  cudaMalloc(&d_nsq, n_cells * sizeof(float));
  cudaMalloc(&d_res, n_cells * n_cells * sizeof(float));
  cudaMalloc(&d_out, n_cells * n_cells * sizeof(double));

  cudaMemcpy(d_i, csc_i_in, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p, csc_p_in, (n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v_host.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_zr, zr_host.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nsq, nsq_host.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threads(16, 16);
  dim3 blocks((n_cells + 15) / 16, (n_cells + 15) / 16);

  RspearmanSparse_per_cell_pair_same_block<<<blocks, threads>>>(
      d_p, d_i, d_v, d_zr, d_nsq, n_genes, n_cells, d_res);
  gpuErrchk(cudaPeekAtLastError());

  FinalizePerCellPairFloat<<<blocks, threads>>>(n_cells, d_res, d_out);
  gpuErrchk(cudaPeekAtLastError());

  cudaMemcpy(result, d_out, n_cells * n_cells * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_i); cudaFree(d_p); cudaFree(d_v); cudaFree(d_zr); cudaFree(d_nsq);
  cudaFree(d_res); cudaFree(d_out);
}


extern "C" void matrix_Spearman_sparse_per_cell_pair_distance_different_blocks(
    int* a_i_in, int* a_p_in, double* a_x_in,
    int* b_i_in, int* b_p_in, double* b_x_in,
    double* result, int* num_rows, int* num_columns,
    int* num_columns_b, int* num_elements_a, int* num_elements_b)
{
  int n_genes = *num_rows;
  int n_cells_a = *num_columns;
  int n_cells_b = *num_columns_b;
  int nnz_a = *num_elements_a;
  int nnz_b = *num_elements_b;

  std::vector<float> a_v(nnz_a), b_v(nnz_b);
  std::vector<float> a_zr(n_cells_a), b_zr(n_cells_b);
  std::vector<float> a_nsq(n_cells_a), b_nsq(n_cells_b);
  spearman_per_cell_pair_preprocess_host(
      a_p_in, a_x_in, n_genes, n_cells_a, a_v.data(), a_zr.data(), a_nsq.data());
  spearman_per_cell_pair_preprocess_host(
      b_p_in, b_x_in, n_genes, n_cells_b, b_v.data(), b_zr.data(), b_nsq.data());

  int* d_ai; int* d_ap; float* d_av; float* d_azr; float* d_ansq;
  int* d_bi; int* d_bp; float* d_bv; float* d_bzr; float* d_bnsq;
  float* d_res;
  cudaMalloc(&d_ai, nnz_a * sizeof(int));
  cudaMalloc(&d_ap, (n_cells_a + 1) * sizeof(int));
  cudaMalloc(&d_av, nnz_a * sizeof(float));
  cudaMalloc(&d_azr, n_cells_a * sizeof(float));
  cudaMalloc(&d_ansq, n_cells_a * sizeof(float));
  cudaMalloc(&d_bi, nnz_b * sizeof(int));
  cudaMalloc(&d_bp, (n_cells_b + 1) * sizeof(int));
  cudaMalloc(&d_bv, nnz_b * sizeof(float));
  cudaMalloc(&d_bzr, n_cells_b * sizeof(float));
  cudaMalloc(&d_bnsq, n_cells_b * sizeof(float));
  cudaMalloc(&d_res, n_cells_a * n_cells_b * sizeof(float));

  cudaMemcpy(d_ai, a_i_in, nnz_a * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ap, a_p_in, (n_cells_a + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_av, a_v.data(), nnz_a * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_azr, a_zr.data(), n_cells_a * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ansq, a_nsq.data(), n_cells_a * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bi, b_i_in, nnz_b * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bp, b_p_in, (n_cells_b + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bv, b_v.data(), nnz_b * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bzr, b_zr.data(), n_cells_b * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bnsq, b_nsq.data(), n_cells_b * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threads(16, 16);
  dim3 blocks((n_cells_b + 15) / 16, (n_cells_a + 15) / 16);

  RspearmanSparse_per_cell_pair_different_blocks<<<blocks, threads>>>(
      d_ap, d_ai, d_av, d_azr, d_ansq,
      d_bp, d_bi, d_bv, d_bzr, d_bnsq,
      n_genes, n_cells_a, n_cells_b, d_res);
  gpuErrchk(cudaPeekAtLastError());

  std::vector<float> h_res(n_cells_a * n_cells_b);
  cudaMemcpy(h_res.data(), d_res, n_cells_a * n_cells_b * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n_cells_a * n_cells_b; ++i) result[i] = (double)h_res[i];

  cudaFree(d_ai); cudaFree(d_ap); cudaFree(d_av); cudaFree(d_azr); cudaFree(d_ansq);
  cudaFree(d_bi); cudaFree(d_bp); cudaFree(d_bv); cudaFree(d_bzr); cudaFree(d_bnsq);
  cudaFree(d_res);
}


// ==================== Spearman ====================

// Rank columns and center (subtract mean rank = (n+1)/2) so that
// the existing Cosine kernels (which compute cosine-style correlation)
// produce the correct Pearson-on-ranks = Spearman result.
// rank_columns and csr_to_ranked_dense (Spearman rank transforms) moved to pc_linalg.

extern "C" void matrix_Spearman_distance_same_block(double* a, double* b, double* c, int* n, int* m, int* m_b) {
    int array_size = *n * *m;
    float* array_new = new float[array_size];
    for (int i = 0; i < array_size; ++i) {
        array_new[i] = static_cast<float>(a[i]);
    }
    rank_columns(array_new, *n, *m);

    float* d_array;
    cudaMalloc(&d_array, array_size * sizeof(float));
    cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (*n + threads - 1) / threads;

    float* d_result;
    float* h_result = new float[(*m) * (*m)];
    cudaMalloc(&d_result, (*m) * (*m) * sizeof(float));
    cudaMemset(d_result, 0, (*m) * (*m) * sizeof(float));

    float* d_x_norm_result;
    cudaMalloc(&d_x_norm_result, (*m) * (*m) * sizeof(float));
    cudaMemset(d_x_norm_result, 0, (*m) * (*m) * sizeof(float));

    float* d_y_norm_result;
    cudaMalloc(&d_y_norm_result, (*m) * (*m) * sizeof(float));
    cudaMemset(d_y_norm_result, 0, (*m) * (*m) * sizeof(float));

    RcosineCorr_gpu_atomic_float_same_block<<<blocks, threads>>>(
        d_array, *n, *m, d_result, d_x_norm_result, d_y_norm_result
    );

    int columns = *m;
    dim3 block_size(32, 32);
    dim3 num_blocks((columns + 31) / 32, (columns + 31) / 32);
    FinalizeCosine<<<num_blocks, block_size>>>(columns, d_result, d_x_norm_result, d_y_norm_result);

    cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);

    // FinalizeCosine computes upper triangle (row < col) in row-major.
    // Mirror on host to fill the lower triangle reliably.
    for (int row = 0; row < columns; ++row) {
        for (int col = 0; col < columns; ++col) {
            if (row < col) {
                c[row * columns + col] = h_result[row * columns + col];
            } else if (row > col) {
                c[row * columns + col] = h_result[col * columns + row];
            } else {
                c[row * columns + col] = 0.0;
            }
        }
    }

    free(array_new);
    free(h_result);
    cudaFree(d_result);
    cudaFree(d_x_norm_result);
    cudaFree(d_y_norm_result);
    cudaFree(d_array);
}


extern "C" void matrix_Spearman_distance_different_blocks(double* a, double* b, double* c, int* n, int* m, int* m_b) {
    int array_size = *n * *m;
    float* array_new = new float[array_size];
    for (int i = 0; i < array_size; ++i) {
        array_new[i] = static_cast<float>(a[i]);
    }
    rank_columns(array_new, *n, *m);

    int array2_size = *n * (*m_b);
    float* array2_new = new float[array2_size];
    for (int i = 0; i < array2_size; ++i) {
        array2_new[i] = static_cast<float>(b[i]);
    }
    rank_columns(array2_new, *n, *m_b);

    float* d_array;
    float* d_array2;
    cudaMalloc(&d_array, array_size * sizeof(float));
    cudaMalloc(&d_array2, array2_size * sizeof(float));
    cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, array2_new, array2_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks_in_row = (*n + threads - 1) / threads;

    float* scalar;
    float* h_scalar = new float[(*m) * (*m_b)];
    float* prod1;
    float* h_prod1 = new float[(*m) * (*m_b)];
    float* prod2;
    float* h_prod2 = new float[(*m) * (*m_b)];

    cudaMalloc(&scalar, (*m) * (*m_b) * sizeof(float));
    cudaMemset(scalar, 0, (*m) * (*m_b) * sizeof(float));
    cudaMalloc(&prod1, (*m) * (*m_b) * sizeof(float));
    cudaMemset(prod1, 0, (*m) * (*m_b) * sizeof(float));
    cudaMalloc(&prod2, (*m) * (*m_b) * sizeof(float));
    cudaMemset(prod2, 0, (*m) * (*m_b) * sizeof(float));

    RcosineCorr_gpu_atomic_float_different_blocks<<<blocks_in_row, threads>>>(
        d_array, d_array2, *n, *m, *m_b, scalar, prod1, prod2
    );

    cudaMemcpy(h_scalar, scalar, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prod1, prod1, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prod2, prod2, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);

    // GPU kernel writes row-major [col1*m_b + col2], R needs column-major [col2*m + col1]
    for (int col1 = 0; col1 < *m; ++col1) {
        for (int col2 = 0; col2 < *m_b; ++col2) {
            int gpu_idx = col1 * (*m_b) + col2;
            int r_idx = col2 * (*m) + col1;
            c[r_idx] = 1.0 - h_scalar[gpu_idx] / sqrtf(h_prod1[gpu_idx]) / sqrtf(h_prod2[gpu_idx]);
        }
    }

    delete[] h_scalar;
    delete[] h_prod2;
    delete[] h_prod1;
    delete[] array_new;
    delete[] array2_new;
    cudaFree(scalar);
    cudaFree(prod2);
    cudaFree(prod1);
    cudaFree(d_array);
    cudaFree(d_array2);
}


extern "C" void matrix_Spearman_sparse_distance_same_block(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
    int rows = *num_rows;
    int columns = *num_columns;

    float* dense = csr_to_ranked_dense(a_index, a_positions, a_double_values, rows, columns);

    float* d_array;
    cudaMalloc(&d_array, rows * columns * sizeof(float));
    cudaMemcpy(d_array, dense, rows * columns * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (rows + threads - 1) / threads;

    float* d_result;
    float* h_result = new float[columns * columns];
    cudaMalloc(&d_result, columns * columns * sizeof(float));
    cudaMemset(d_result, 0, columns * columns * sizeof(float));

    float* d_x_norm;
    cudaMalloc(&d_x_norm, columns * columns * sizeof(float));
    cudaMemset(d_x_norm, 0, columns * columns * sizeof(float));

    float* d_y_norm;
    cudaMalloc(&d_y_norm, columns * columns * sizeof(float));
    cudaMemset(d_y_norm, 0, columns * columns * sizeof(float));

    RcosineCorr_gpu_atomic_float_same_block<<<blocks, threads>>>(
        d_array, rows, columns, d_result, d_x_norm, d_y_norm
    );

    dim3 block_size(32, 32);
    dim3 num_blocks((columns + 31) / 32, (columns + 31) / 32);
    FinalizeCosine<<<num_blocks, block_size>>>(columns, d_result, d_x_norm, d_y_norm);

    cudaMemcpy(h_result, d_result, columns * columns * sizeof(float), cudaMemcpyDeviceToHost);

    // Mirror on host: FinalizeCosine computes upper triangle (row < col) row-major
    for (int row = 0; row < columns; ++row) {
        for (int col = 0; col < columns; ++col) {
            if (row < col) {
                result[row * columns + col] = h_result[row * columns + col];
            } else if (row > col) {
                result[row * columns + col] = h_result[col * columns + row];
            } else {
                result[row * columns + col] = 0.0;
            }
        }
    }

    delete[] dense;
    delete[] h_result;
    cudaFree(d_array);
    cudaFree(d_result);
    cudaFree(d_x_norm);
    cudaFree(d_y_norm);
}


extern "C" void matrix_Spearman_sparse_distance_different_blocks(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
    int rows = *num_rows;
    int columns = *num_columns;
    int columns_b = *num_columns_b;

    float* dense_a = csr_to_ranked_dense(a_index, a_positions, a_double_values, rows, columns);
    float* dense_b = csr_to_ranked_dense(b_index, b_positions, b_double_values, rows, columns_b);

    float* d_array;
    float* d_array2;
    cudaMalloc(&d_array, rows * columns * sizeof(float));
    cudaMalloc(&d_array2, rows * columns_b * sizeof(float));
    cudaMemcpy(d_array, dense_a, rows * columns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, dense_b, rows * columns_b * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks_in_row = (rows + threads - 1) / threads;

    float* scalar;
    float* h_scalar = new float[columns * columns_b];
    float* prod1;
    float* h_prod1 = new float[columns * columns_b];
    float* prod2;
    float* h_prod2 = new float[columns * columns_b];

    cudaMalloc(&scalar, columns * columns_b * sizeof(float));
    cudaMemset(scalar, 0, columns * columns_b * sizeof(float));
    cudaMalloc(&prod1, columns * columns_b * sizeof(float));
    cudaMemset(prod1, 0, columns * columns_b * sizeof(float));
    cudaMalloc(&prod2, columns * columns_b * sizeof(float));
    cudaMemset(prod2, 0, columns * columns_b * sizeof(float));

    RcosineCorr_gpu_atomic_float_different_blocks<<<blocks_in_row, threads>>>(
        d_array, d_array2, rows, columns, columns_b, scalar, prod1, prod2
    );

    cudaMemcpy(h_scalar, scalar, columns * columns_b * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prod1, prod1, columns * columns_b * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prod2, prod2, columns * columns_b * sizeof(float), cudaMemcpyDeviceToHost);

    // GPU kernel writes row-major [col1*columns_b + col2], R needs column-major [col2*columns + col1]
    for (int col1 = 0; col1 < columns; ++col1) {
        for (int col2 = 0; col2 < columns_b; ++col2) {
            int gpu_idx = col1 * columns_b + col2;
            int r_idx = col2 * columns + col1;
            result[r_idx] = 1.0 - h_scalar[gpu_idx] / sqrtf(h_prod1[gpu_idx]) / sqrtf(h_prod2[gpu_idx]);
        }
    }

    delete[] dense_a;
    delete[] dense_b;
    delete[] h_scalar;
    delete[] h_prod1;
    delete[] h_prod2;
    cudaFree(scalar);
    cudaFree(prod1);
    cudaFree(prod2);
    cudaFree(d_array);
    cudaFree(d_array2);
}
