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



// Dense Euclidean drivers (R-facing).
static void pc_drive_euclidean_same(double* a, double* c, int n, int m) {
    size_t sz = (size_t)n * m;
    std::vector<float> h(sz);
    for (size_t i = 0; i < sz; ++i) h[i] = (float)a[i];

    float *d_A, *d_D;
    cudaMalloc(&d_A, sz * sizeof(float));
    cudaMalloc(&d_D, (size_t)m * m * sizeof(float));
    cudaMemcpy(d_A, h.data(), sz * sizeof(float), cudaMemcpyHostToDevice);
    pc_euclidean_same_block_device(d_A, n, m, d_D);

    std::vector<float> out((size_t)m * m);
    cudaMemcpy(out.data(), d_D, (size_t)m * m * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < (size_t)m * m; ++i) c[i] = (double)out[i];
    cudaFree(d_A); cudaFree(d_D);
}


static void pc_drive_euclidean_diff(double* a, double* b, double* c, int n,
                                    int m, int m_b) {
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
    pc_euclidean_different_blocks_device(d_A, d_B, n, m, m_b, d_D);

    std::vector<float> out((size_t)m * m_b);
    cudaMemcpy(out.data(), d_D, (size_t)m * m_b * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < (size_t)m * m_b; ++i) c[i] = (double)out[i];
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
}


static void pc_drive_sparse_euclidean_same(int* a_index, int* a_positions,
                                           double* a_values, double* c,
                                           int n, int m, int nnz) {
    bool L = pc_rt_log(); double t0=L?omp_get_wtime():0;
    float* d_A = pc_sparse_to_dense_device(a_index, a_positions, a_values,
                                           n, m, nnz);
    if(L) cudaDeviceSynchronize(); double t1=L?omp_get_wtime():0;
    float* d_D;
    cudaMalloc(&d_D, (size_t)m * m * sizeof(float));
    pc_euclidean_same_block_device(d_A, n, m, d_D);
    if(L) cudaDeviceSynchronize(); double t2=L?omp_get_wtime():0;

    std::vector<float> out((size_t)m * m);
    cudaMemcpy(out.data(), d_D, (size_t)m * m * sizeof(float),
               cudaMemcpyDeviceToHost);
    double t3=L?omp_get_wtime():0;
    for (size_t i = 0; i < (size_t)m * m; ++i) c[i] = (double)out[i];
    cudaFree(d_A); cudaFree(d_D);
    if(L) Rprintf("[rt-sparse-euc n=%d m=%d nnz=%d] densify+upload=%.0f gemm=%.0f D2H+f2d=%.0f | total=%.0f ms\n",
        n,m,nnz,(t1-t0)*1e3,(t2-t1)*1e3,(omp_get_wtime()-t2)*1e3,(omp_get_wtime()-t0)*1e3);
}


static void pc_drive_sparse_euclidean_diff(int* a_index, int* a_positions,
                                           double* a_values,
                                           int* b_index, int* b_positions,
                                           double* b_values,
                                           double* c, int n, int m, int m_b,
                                           int nnz_a, int nnz_b) {
    float* d_A = pc_sparse_to_dense_device(a_index, a_positions, a_values,
                                           n, m, nnz_a);
    float* d_B = pc_sparse_to_dense_device(b_index, b_positions, b_values,
                                           n, m_b, nnz_b);
    float* d_D;
    cudaMalloc(&d_D, (size_t)m * m_b * sizeof(float));
    pc_euclidean_different_blocks_device(d_A, d_B, n, m, m_b, d_D);

    std::vector<float> out((size_t)m * m_b);
    cudaMemcpy(out.data(), d_D, (size_t)m * m_b * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < (size_t)m * m_b; ++i) c[i] = (double)out[i];
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
}



__global__ void FinalizeEuclidean(int columns, float* results) {
    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
    int column_index = blockDim.y * blockIdx.y + threadIdx.y;

    int index = row_index * columns + column_index;

    if ((row_index < columns) && (column_index < columns)) {
        if (row_index < column_index) {
            results[index] = sqrtf(results[index]);
        } else if (row_index > column_index) {
            results[index] = results[columns * column_index + row_index];
        }
    }
}


__global__ void Reuclidean_gpu_row_atomic_float(float* array, const int n, const int m, float* result) {
    int col1_num = blockIdx.x * blockDim.x + threadIdx.x;
    int col2_num = blockIdx.y * blockDim.y + threadIdx.y;
    
    int sum = 0.0f;
    
    int local_tid = threadIdx.x * blockDim.x + threadIdx.y;

    if (col1_num < col2_num && col2_num < m) {
        float* col1 = array + n * col1_num;
        float* col2 = array + n * col2_num;
        float diff = 0;
        for (int i = 0; i < n; ++i) {
            diff = col1[i] - col2[i];
            diff = diff * diff;
            sum += diff;
        }
        result[col1_num * m + col2_num] = sum;
    }
}


__global__ void Reuclidean_gpu_atomic_float(float* array, const int n, const int m, float* result) {
  
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  for (int col1_num = 0; col1_num < m; ++col1_num) {
      for (int col2_num = col1_num + 1; col2_num < m; ++col2_num) {
          float* col1 = array + n * col1_num;
          float* col2 = array + n * col2_num;

          if (row < n) {
            float diff = col1[row] - col2[row];
            diff = diff * diff;
            atomicAdd(result + col1_num * m + col2_num, diff);

          }
      }
  }
}

__global__ void Reuclidean_gpu_atomic_row_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, float* result) {
    int col1_num = blockIdx.x * blockDim.x + threadIdx.x;
    int col2_num = blockIdx.y * blockDim.y + threadIdx.y;
    
    int sum = 0.0f;
    
    int local_tid = threadIdx.x * blockDim.x + threadIdx.y;

    if (col1_num < m && col2_num < m_b) {
        float* col1 = array + n * col1_num;
        float* col2 = array2 + n * col2_num;
        float diff = 0;
        for (int i = 0; i < n; ++i) {
            diff = col1[i] - col2[i];
            diff = diff * diff;
            sum += diff;
        }
        result[col2_num * m + col1_num] = sqrtf(sum);
    }

}


__global__ void Reuclidean_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, float* result) {
  
  int row = blockIdx.x * blockDim.x + threadIdx.x;


  for (int col1_num = 0; col1_num < m; ++col1_num) {
      for (int col2_num = 0; col2_num < m_b; ++col2_num) {
          float* col1 = array + n * col1_num;
          float* col2 = array2 + n * col2_num;

          if (row < n) {
            float diff = col1[row] - col2[row];
            diff = diff * diff;
            //atomicAdd(result + col1_num * m + col2_num, diff);
            atomicAdd(result + col2_num * m + col1_num, diff);

          }
      }
  }
}



//' Driver Function for calculation of Euclidean matrix for same block.
//'
//' Allocates Memory required for the operation. Then,
//' efficiently calculate the distance matrix using the kernel, 
//' which is translated to appropriate R tables.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//' 
extern "C" void matrix_Euclidean_distance_same_block(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){
  pc_drive_euclidean_same(a, c, *n, *m);
}



//' Driver Function for calculation of Euclidean matrix for different block.
//'
//' Allocates Memory required for the operation. Then,
//' efficiently calculate the distance matrix using the kernel, 
//' which is translated to appropriate R tables.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//' 
extern "C" void matrix_Euclidean_distance_different_blocks(double* a, double* b, double* c, int* n, int* m, int* m_b){
  pc_drive_euclidean_diff(a, b, c, *n, *m, *m_b);
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
//'`
__global__ void ReuclideanSparse_gpu_atomic_float_same_block(
    int* a_index, int* a_positions, float* a_double_values,
    int rows, int columns, float* result) {

    int row_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_index < rows) {
        int start_column = a_positions[row_index];
        int end_column = a_positions[row_index + 1];

        for (int col1_index = start_column; col1_index < end_column; ++col1_index) {

            for (int col2_index = col1_index; col2_index < end_column; ++col2_index) {
                int prev_col_index = col1_index - 1;
                int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;

                int next_col_index = col2_index + 1;
                int next_col = (next_col_index < end_column) ? a_index[next_col_index] : columns;

                int col1 = a_index[col1_index];
                int col2 = a_index[col2_index];

                float value1 = a_double_values[col1_index];
                float value2 = a_double_values[col2_index];

                for (int left = prev_col + 1; left < col1; ++left) {
                    atomicAdd(&result[left * columns + col2], value2 * value2);
                }

                for (int right = col2 + 1; right < next_col; ++right) {
                    atomicAdd(&result[col1 * columns + right], value1 * value1);
                }
		// using diff calculation directcly passing to n,m col directly
                float diff = (value1 - value2) * (value1 - value2);
                atomicAdd(&result[col1 * columns + col2], diff);
            }
        }
    }
}


extern "C" void matrix_Euclidean_sparse_distance_same_block(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b) {
    pc_drive_sparse_euclidean_same(a_index, a_positions, a_double_values,
                                   result, *num_rows, *num_columns,
                                   *num_elements_a);
}


//====================================TESTING SPARSE METHODS==============================
__global__ void ReuclideanSparse_gpu_atomic_float_different_blocks(
    int* a_index, int* a_positions, float* a_double_values,
    int* b_index, int* b_positions, float* b_double_values,
    int rows, int columns, int columns_b, float* result) {

    int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index < rows) {
        int start_column = a_positions[row_index];
        int end_column = a_positions[row_index + 1];

        int start_column_b = b_positions[row_index];
        int end_column_b = b_positions[row_index + 1];

        for (int col1_index = start_column; col1_index <= end_column; ++col1_index) {
            int prev_col_index = col1_index - 1;
            int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
            float value1 = (col1_index < end_column) ? a_double_values[col1_index] : 0.0f;

            int col1 = (col1_index < end_column) ? a_index[col1_index] : columns;
            for (int col2_index = start_column_b; col2_index <= end_column_b; ++col2_index) {
                int prev_col_b_index = col2_index - 1;
                int prev_col2 = (prev_col_b_index >= start_column_b) ? b_index[prev_col_b_index] : -1;

                int col2 = (col2_index < end_column_b) ? b_index[col2_index] : columns_b;

                float value2 = (col2_index < end_column_b) ? b_double_values[col2_index] : 0.0f;

                if (col2 < columns_b) {
                    for (int left = prev_col + 1; left < col1; ++left) {
                        atomicAdd(&result[col2 * columns + left], value2 * value2);
                    }
                }

                if (col1 < columns) {
                    for (int left = prev_col2 + 1; left < col2; ++left) {
                        atomicAdd(&result[left * columns + col1], value1 * value1);
                    }
                }

                if (col1 < columns && col2 < columns_b) {
                    float diff = (value1 - value2) * (value1 - value2);
                    atomicAdd(&result[col2 * columns + col1], diff);
                }
            }
        }
    }
}


extern "C" void matrix_Euclidean_sparse_distance_different_blocks(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b) {
    pc_drive_sparse_euclidean_diff(a_index, a_positions, a_double_values,
                                   b_index, b_positions, b_double_values,
                                   result, *num_rows, *num_columns,
                                   *num_columns_b, *num_elements_a,
                                   *num_elements_b);
}


// ─── Euclidean ───

__global__ void ReuclideanSparse_per_cell_pair_same_block(
    const int* __restrict__ csc_p,
    const int* __restrict__ csc_i,
    const float* __restrict__ csc_x,
    int n_cells,
    float* __restrict__ result_out)
{
  int cell_a = blockIdx.y * blockDim.y + threadIdx.y;
  int cell_b = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_a >= cell_b || cell_b >= n_cells) return;

  int ia = csc_p[cell_a], ea = csc_p[cell_a + 1];
  int ib = csc_p[cell_b], eb = csc_p[cell_b + 1];

  float acc = 0;
  while (ia < ea || ib < eb) {
    if (ia < ea && (ib >= eb || csc_i[ia] < csc_i[ib])) {
      float v = csc_x[ia]; acc += v * v; ++ia;
    } else if (ib < eb && (ia >= ea || csc_i[ib] < csc_i[ia])) {
      float v = csc_x[ib]; acc += v * v; ++ib;
    } else {
      float d = csc_x[ia] - csc_x[ib]; acc += d * d; ++ia; ++ib;
    }
  }
  result_out[cell_a * n_cells + cell_b] = sqrtf(acc);
}


__global__ void ReuclideanSparse_per_cell_pair_different_blocks(
    const int* __restrict__ a_csc_p,
    const int* __restrict__ a_csc_i,
    const float* __restrict__ a_csc_x,
    const int* __restrict__ b_csc_p,
    const int* __restrict__ b_csc_i,
    const float* __restrict__ b_csc_x,
    int n_cells_a, int n_cells_b,
    float* __restrict__ result_out)
{
  int cell_a = blockIdx.y * blockDim.y + threadIdx.y;
  int cell_b = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_a >= n_cells_a || cell_b >= n_cells_b) return;

  int ia = a_csc_p[cell_a], ea = a_csc_p[cell_a + 1];
  int ib = b_csc_p[cell_b], eb = b_csc_p[cell_b + 1];

  float acc = 0;
  while (ia < ea || ib < eb) {
    if (ia < ea && (ib >= eb || a_csc_i[ia] < b_csc_i[ib])) {
      float v = a_csc_x[ia]; acc += v * v; ++ia;
    } else if (ib < eb && (ia >= ea || b_csc_i[ib] < a_csc_i[ia])) {
      float v = b_csc_x[ib]; acc += v * v; ++ib;
    } else {
      float d = a_csc_x[ia] - b_csc_x[ib]; acc += d * d; ++ia; ++ib;
    }
  }
  result_out[cell_b * n_cells_a + cell_a] = sqrtf(acc);
}


// ─── Drivers: Euclidean ───

extern "C" void matrix_Euclidean_sparse_per_cell_pair_distance_same_block(
    int* csc_i_in, int* csc_p_in, double* csc_x_in,
    int* /*b*/, int* /*b*/, double* /*b*/,
    double* result, int* num_rows, int* num_columns,
    int* /*num_columns_b*/, int* num_elements, int* /*num_elements_b*/)
{
  int n_genes = *num_rows;  (void)n_genes;
  int n_cells = *num_columns;
  int nnz     = *num_elements;

  std::vector<float> csc_x_f(nnz);
  for (int k = 0; k < nnz; ++k) csc_x_f[k] = (float)csc_x_in[k];

  int* d_i; int* d_p; float* d_x; float* d_res; double* d_out;
  cudaMalloc(&d_i, nnz * sizeof(int));
  cudaMalloc(&d_p, (n_cells + 1) * sizeof(int));
  cudaMalloc(&d_x, nnz * sizeof(float));
  cudaMalloc(&d_res, n_cells * n_cells * sizeof(float));
  cudaMalloc(&d_out, n_cells * n_cells * sizeof(double));

  cudaMemcpy(d_i, csc_i_in, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p, csc_p_in, (n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, csc_x_f.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threads(16, 16);
  dim3 blocks((n_cells + 15) / 16, (n_cells + 15) / 16);

  ReuclideanSparse_per_cell_pair_same_block<<<blocks, threads>>>(d_p, d_i, d_x, n_cells, d_res);
  gpuErrchk(cudaPeekAtLastError());

  FinalizePerCellPairFloat<<<blocks, threads>>>(n_cells, d_res, d_out);
  gpuErrchk(cudaPeekAtLastError());

  cudaMemcpy(result, d_out, n_cells * n_cells * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_i); cudaFree(d_p); cudaFree(d_x); cudaFree(d_res); cudaFree(d_out);
}


extern "C" void matrix_Euclidean_sparse_per_cell_pair_distance_different_blocks(
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

  int* d_ai; int* d_ap; float* d_ax;
  int* d_bi; int* d_bp; float* d_bx;
  float* d_res;
  cudaMalloc(&d_ai, nnz_a * sizeof(int));
  cudaMalloc(&d_ap, (n_cells_a + 1) * sizeof(int));
  cudaMalloc(&d_ax, nnz_a * sizeof(float));
  cudaMalloc(&d_bi, nnz_b * sizeof(int));
  cudaMalloc(&d_bp, (n_cells_b + 1) * sizeof(int));
  cudaMalloc(&d_bx, nnz_b * sizeof(float));
  cudaMalloc(&d_res, n_cells_a * n_cells_b * sizeof(float));

  cudaMemcpy(d_ai, a_i_in, nnz_a * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ap, a_p_in, (n_cells_a + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ax, a_xf.data(), nnz_a * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bi, b_i_in, nnz_b * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bp, b_p_in, (n_cells_b + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bx, b_xf.data(), nnz_b * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threads(16, 16);
  dim3 blocks((n_cells_b + 15) / 16, (n_cells_a + 15) / 16);

  ReuclideanSparse_per_cell_pair_different_blocks<<<blocks, threads>>>(
      d_ap, d_ai, d_ax, d_bp, d_bi, d_bx, n_cells_a, n_cells_b, d_res);
  gpuErrchk(cudaPeekAtLastError());

  std::vector<float> h_res(n_cells_a * n_cells_b);
  cudaMemcpy(h_res.data(), d_res, n_cells_a * n_cells_b * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n_cells_a * n_cells_b; ++i) result[i] = (double)h_res[i];

  cudaFree(d_ai); cudaFree(d_ap); cudaFree(d_ax);
  cudaFree(d_bi); cudaFree(d_bp); cudaFree(d_bx);
  cudaFree(d_res);
}
