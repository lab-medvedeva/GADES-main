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



// (euclidean/cosine same/diff device glue above is shared by the unified
//  .Call dispatcher C_dense_block, defined at the end of this file.)

// Sparse CSR input -> densify -> existing dense cuBLAS path. For typical
// scRNA-seq density (5-20%) SGEMM on tensor cores beats an SpMM pass. For
// ultra-sparse data (<1%), cusparseSpMM would be preferable — add if needed.
void pc_drive_sparse_cosine_same(int* a_index, int* a_positions,
                                        double* a_values, double* c,
                                        int n, int m, int nnz, bool center) {
    float* d_A = pc_sparse_to_dense_device(a_index, a_positions, a_values,
                                           n, m, nnz);
    if (center) pc_center_columns_device(d_A, n, m);

    float* d_D;
    cudaMalloc(&d_D, (size_t)m * m * sizeof(float));
    pc_cosine_same_block_device(d_A, n, m, d_D);

    std::vector<float> out((size_t)m * m);
    cudaMemcpy(out.data(), d_D, (size_t)m * m * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < (size_t)m * m; ++i) c[i] = (double)out[i];
    cudaFree(d_A); cudaFree(d_D);
}


void pc_drive_sparse_cosine_diff(int* a_index, int* a_positions,
                                        double* a_values,
                                        int* b_index, int* b_positions,
                                        double* b_values,
                                        double* c, int n, int m, int m_b,
                                        int nnz_a, int nnz_b, bool center) {
    float* d_A = pc_sparse_to_dense_device(a_index, a_positions, a_values,
                                           n, m, nnz_a);
    float* d_B = pc_sparse_to_dense_device(b_index, b_positions, b_values,
                                           n, m_b, nnz_b);
    if (center) {
        pc_center_columns_device(d_A, n, m);
        pc_center_columns_device(d_B, n, m_b);
    }

    float* d_D;
    cudaMalloc(&d_D, (size_t)m * m_b * sizeof(float));
    pc_cosine_different_blocks_device(d_A, d_B, n, m, m_b, d_D);

    std::vector<float> out((size_t)m * m_b);
    cudaMemcpy(out.data(), d_D, (size_t)m * m_b * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < (size_t)m * m_b; ++i) c[i] = (double)out[i];
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
}


__global__ void FinalizeCosine(int columns, float* results, float* x_squares, float* y_squares) {
    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
    int column_index = blockDim.y * blockIdx.y + threadIdx.y;

    int index = row_index * columns + column_index;

    if ((row_index < columns) && (column_index < columns)) {
        if (row_index < column_index) {
            results[index] = 1.0 - results[index] / sqrtf(x_squares[index]) / sqrtf(y_squares[index]);
        } else if (row_index > column_index) {
            results[index] = results[columns * column_index + row_index];
        }
    }
}


__global__ void FinalizeCosineSparse(int columns, float* results, float* squares) {
    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
    int column_index = blockDim.y * blockIdx.y + threadIdx.y;

    int index = row_index * columns + column_index;

    if ((row_index < columns) && (column_index < columns)) {
        if (row_index < column_index) {
            results[index] = 1.0 - results[index] / sqrtf(squares[row_index]) / sqrtf(squares[column_index]);
        } else if (row_index > column_index) {
            results[index] = results[columns * column_index + row_index];
        }
    }
}


__global__ void RcosineCorr_gpu_atomic_float_same_block(
  float* array,
  const int n, const int m,
  float* scalar_product,
  float* x_norm,
  float* y_norm
) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  for (int col1_num = 0; col1_num < m; ++col1_num) {
      for (int col2_num = col1_num+1; col2_num < m; ++col2_num) {
          float* col1 = array + n * col1_num;
          float* col2 = array + n * col2_num;

          if (row < n) {
            float diff = col1[row] * col2[row];
            atomicAdd(scalar_product + col1_num * m + col2_num, diff);

            float x_element_norm = col1[row] * col1[row];
            float y_element_norm = col2[row] * col2[row];

            atomicAdd(x_norm + col1_num * m + col2_num, x_element_norm);
            atomicAdd(y_norm + col1_num * m + col2_num, y_element_norm);
	
          }
      }
  }
}


__global__ void RcosineCorr_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, float* scalar_prod,float *x_norm, float* y_norm ){
    int row = blockIdx.x*blockDim.x +threadIdx.x;
    for (int col1_num=0;col1_num<m;++col1_num){
	for(int col2_num=0;col2_num<m_b;++col2_num){
		float* col1 = array + n * col1_num;
		float* col2 = array2 + n * col2_num;
		if(row<n) {
				float num = (col1[row] * col2[row]);
				float sum1 = (col1[row] * col1[row]);
				float sum2 = (col2[row] * col2[row]);
				//if(dist==1){}
				atomicAdd(scalar_prod+col1_num*m_b+col2_num,num);
				atomicAdd(x_norm+col1_num*m_b+col2_num,sum1);
				atomicAdd(y_norm+col1_num*m_b+col2_num,sum2);
				//atomicAdd(scalar_prod+col2_num*m+col1_num,num);
				//atomicAdd(x_norm+col2_num*m+col1_num,sum1);
				//atomicAdd(y_norm+col2_num*m+col1_num,sum2);
				//!debug if(threadIdx.x==0){printf("val1=%4.2f, val2=%4.2f, num=%4.2f, sum1=%4.2f, sum2=%4.2f, \n", col1[row],col2[row],num,sum1,sum2);}
			}
		}
	}
}



//' Driver Function for calculation of Cosine matrix for same block.
//'
//' Allocates Memory required for the operation. Then,
//' efficiently calculate the distance matrix using the kernel, 
//' which is translated to appropriate R tables.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//' 
extern "C" void matrix_Cosine_distance_same_block(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){
  pc_drive_cosine_same(a, c, *n, *m, /*center=*/false);
}


//' Driver Function for calculation of Cosine matrix for different block.
//'
//' Allocates Memory required for the operation. Then,
//' efficiently calculate the distance matrix using the kernel, 
//' which is translated to appropriate R tables.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//' 
extern "C" void matrix_Cosine_distance_different_blocks(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){
  pc_drive_cosine_diff(a, b, c, *n, *m, *m_b, /*center=*/false);
}



__global__ void RcosineSparseCorr_gpu_atomic_float_different_blocks(
    int* a_index, int* a_positions, float* a_double_values,
    int* b_index, int* b_positions, float* b_double_values,
    float* result, int rows, int columns, int columns_b,
    float* squares_a, float* squares_b) {

    int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index < rows) {
        int start_column = a_positions[row_index];
        int end_column = a_positions[row_index + 1];

        int start_column_b = b_positions[row_index];
        int end_column_b = b_positions[row_index + 1];

        for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
            float value1 = a_double_values[col1_index];
            int col1 = a_index[col1_index];

            for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
                float value2 = b_double_values[col2_index];
                int col2 = b_index[col2_index];

                atomicAdd(&result[col2 * columns + col1], value1 * value2);
            }
        }

        for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
            float value1 = a_double_values[col1_index];
            int col1 = a_index[col1_index];

            atomicAdd(&squares_a[col1], value1 * value1);
        }

        for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
            
            float value2 = b_double_values[col2_index];
            int col2 = b_index[col2_index];

            // printf(
            //   "ROW: %d %d %d %d %d %f\n",
            //   row_index,
            //   start_column_b,
            //   col2_index,
            //   end_column_b,
            //   col2,
            //   value2
            // );

            atomicAdd(&squares_b[col2], value2 * value2);
        }
    }
}


extern "C" void matrix_Cosine_sparse_distance_different_blocks(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b) {
    pc_drive_sparse_cosine_diff(a_index, a_positions, a_double_values,
                                b_index, b_positions, b_double_values,
                                result, *num_rows, *num_columns, *num_columns_b,
                                *num_elements_a, *num_elements_b,
                                /*center=*/false);
}



__global__ void RcosineSparseCorr_gpu_atomic_float_same_block(
    int* a_index, int* a_positions, float* a_double_values,
    float* result, int rows, int columns, float* squares) {

    int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("%d %d\n", row_index, rows);
    if (row_index < rows) {
        // printf("Inside kernel\n");
        int start_column = a_positions[row_index];
        int end_column = a_positions[row_index + 1];
        // printf("Test: %d %d\n", start_column, end_column);
        for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
            // printf("Here\n");
            // int prev_col_index = col1_index - 1;
            // int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
            int col1 = a_index[col1_index];
            float value1 = a_double_values[col1_index];
            atomicAdd(&squares[col1], value1 * value1);
            
            for (int col2_index = col1_index + 1; col2_index < end_column; ++col2_index) {
                //int next_col_index = col2_index + 1;
                //int next_col = (next_col_index < end_column) ? a_index[next_col_index] : columns;
                int col2 = a_index[col2_index];
                float value2 = a_double_values[col2_index];
                
                atomicAdd(&result[col1 * columns + col2], value1 * value2);
            }
        }
    }
}


extern "C" void matrix_Cosine_sparse_distance_same_block(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
    pc_drive_sparse_cosine_same(a_index, a_positions, a_double_values, result,
                                *num_rows, *num_columns, *num_elements_a,
                                /*center=*/false);
}


// ==================== Per-cell-pair sparse: Euclidean / Manhattan / Cosine / Pearson ====================
//
// 1 thread per (cell_a, cell_b), single two-pointer merge over CSC columns.
// O(nnz_a + nnz_b) per thread. No atomics.
// See plans/PER_CELL_PAIR_OTHER_METRICS.md

// ─── Preprocessing kernels ───

__global__ void compute_cell_norms(
    const int* __restrict__ csc_p,
    const float* __restrict__ csc_x,
    int n_cells,
    float* __restrict__ norm_out)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) return;
  float s = 0;
  for (int k = csc_p[c]; k < csc_p[c + 1]; ++k) s += csc_x[k] * csc_x[k];
  norm_out[c] = sqrtf(s);
}


// ─── Cosine ───

__global__ void RcosineSparse_per_cell_pair_same_block(
    const int* __restrict__ csc_p,
    const int* __restrict__ csc_i,
    const float* __restrict__ csc_x,
    const float* __restrict__ norms,
    int n_cells,
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
  result_out[cell_a * n_cells + cell_b] = 1.0f - dot / (norms[cell_a] * norms[cell_b]);
}


__global__ void RcosineSparse_per_cell_pair_different_blocks(
    const int* __restrict__ a_csc_p,
    const int* __restrict__ a_csc_i,
    const float* __restrict__ a_csc_x,
    const float* __restrict__ a_norms,
    const int* __restrict__ b_csc_p,
    const int* __restrict__ b_csc_i,
    const float* __restrict__ b_csc_x,
    const float* __restrict__ b_norms,
    int n_cells_a, int n_cells_b,
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
  result_out[cell_b * n_cells_a + cell_a] = 1.0f - dot / (a_norms[cell_a] * b_norms[cell_b]);
}


// ─── Drivers: Cosine ───

extern "C" void matrix_Cosine_sparse_per_cell_pair_distance_same_block(
    int* csc_i_in, int* csc_p_in, double* csc_x_in,
    int* /*b*/, int* /*b*/, double* /*b*/,
    double* result, int* num_rows, int* num_columns,
    int* /*num_columns_b*/, int* num_elements, int* /*num_elements_b*/)
{
  int n_cells = *num_columns;
  int nnz     = *num_elements;

  std::vector<float> csc_x_f(nnz);
  for (int k = 0; k < nnz; ++k) csc_x_f[k] = (float)csc_x_in[k];

  int* d_i; int* d_p; float* d_x; float* d_norms; float* d_res; double* d_out;
  cudaMalloc(&d_i, nnz * sizeof(int));
  cudaMalloc(&d_p, (n_cells + 1) * sizeof(int));
  cudaMalloc(&d_x, nnz * sizeof(float));
  cudaMalloc(&d_norms, n_cells * sizeof(float));
  cudaMalloc(&d_res, n_cells * n_cells * sizeof(float));
  cudaMalloc(&d_out, n_cells * n_cells * sizeof(double));

  cudaMemcpy(d_i, csc_i_in, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p, csc_p_in, (n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, csc_x_f.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);

  int prep_threads = 256;
  int prep_blocks = (n_cells + prep_threads - 1) / prep_threads;
  compute_cell_norms<<<prep_blocks, prep_threads>>>(d_p, d_x, n_cells, d_norms);
  gpuErrchk(cudaPeekAtLastError());

  dim3 threads(16, 16);
  dim3 blocks((n_cells + 15) / 16, (n_cells + 15) / 16);

  RcosineSparse_per_cell_pair_same_block<<<blocks, threads>>>(d_p, d_i, d_x, d_norms, n_cells, d_res);
  gpuErrchk(cudaPeekAtLastError());

  FinalizePerCellPairFloat<<<blocks, threads>>>(n_cells, d_res, d_out);
  gpuErrchk(cudaPeekAtLastError());

  cudaMemcpy(result, d_out, n_cells * n_cells * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_i); cudaFree(d_p); cudaFree(d_x); cudaFree(d_norms); cudaFree(d_res); cudaFree(d_out);
}


extern "C" void matrix_Cosine_sparse_per_cell_pair_distance_different_blocks(
    int* a_i_in, int* a_p_in, double* a_x_in,
    int* b_i_in, int* b_p_in, double* b_x_in,
    double* result, int* num_rows, int* num_columns,
    int* num_columns_b, int* num_elements_a, int* num_elements_b)
{
  int n_cells_a = *num_columns;
  int n_cells_b = *num_columns_b;
  int nnz_a = *num_elements_a;
  int nnz_b = *num_elements_b;

  std::vector<float> a_xf(nnz_a), b_xf(nnz_b);
  for (int k = 0; k < nnz_a; ++k) a_xf[k] = (float)a_x_in[k];
  for (int k = 0; k < nnz_b; ++k) b_xf[k] = (float)b_x_in[k];

  int* d_ai; int* d_ap; float* d_ax; float* d_an;
  int* d_bi; int* d_bp; float* d_bx; float* d_bn;
  float* d_res;
  cudaMalloc(&d_ai, nnz_a * sizeof(int));
  cudaMalloc(&d_ap, (n_cells_a + 1) * sizeof(int));
  cudaMalloc(&d_ax, nnz_a * sizeof(float));
  cudaMalloc(&d_an, n_cells_a * sizeof(float));
  cudaMalloc(&d_bi, nnz_b * sizeof(int));
  cudaMalloc(&d_bp, (n_cells_b + 1) * sizeof(int));
  cudaMalloc(&d_bx, nnz_b * sizeof(float));
  cudaMalloc(&d_bn, n_cells_b * sizeof(float));
  cudaMalloc(&d_res, n_cells_a * n_cells_b * sizeof(float));

  cudaMemcpy(d_ai, a_i_in, nnz_a * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ap, a_p_in, (n_cells_a + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ax, a_xf.data(), nnz_a * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bi, b_i_in, nnz_b * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bp, b_p_in, (n_cells_b + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bx, b_xf.data(), nnz_b * sizeof(float), cudaMemcpyHostToDevice);

  int prep_threads = 256;
  compute_cell_norms<<<(n_cells_a + prep_threads - 1) / prep_threads, prep_threads>>>(d_ap, d_ax, n_cells_a, d_an);
  compute_cell_norms<<<(n_cells_b + prep_threads - 1) / prep_threads, prep_threads>>>(d_bp, d_bx, n_cells_b, d_bn);
  gpuErrchk(cudaPeekAtLastError());

  dim3 threads(16, 16);
  dim3 blocks((n_cells_b + 15) / 16, (n_cells_a + 15) / 16);

  RcosineSparse_per_cell_pair_different_blocks<<<blocks, threads>>>(
      d_ap, d_ai, d_ax, d_an, d_bp, d_bi, d_bx, d_bn, n_cells_a, n_cells_b, d_res);
  gpuErrchk(cudaPeekAtLastError());

  std::vector<float> h_res(n_cells_a * n_cells_b);
  cudaMemcpy(h_res.data(), d_res, n_cells_a * n_cells_b * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n_cells_a * n_cells_b; ++i) result[i] = (double)h_res[i];

  cudaFree(d_ai); cudaFree(d_ap); cudaFree(d_ax); cudaFree(d_an);
  cudaFree(d_bi); cudaFree(d_bp); cudaFree(d_bx); cudaFree(d_bn);
  cudaFree(d_res);
}
