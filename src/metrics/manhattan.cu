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




__global__ void FinalizeManhattan(int columns, float* results) {
    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
    int column_index = blockDim.y * blockIdx.y + threadIdx.y;

    int index = row_index * columns + column_index;

    if ((row_index < columns) && (column_index < columns)) {
        if (row_index > column_index) {
            results[index] = results[columns * column_index + row_index];
        }
    }
}


__global__ void Rmanhattan_gpu_row_atomic_float(float* array, const int n, const int m, float* result) {
    int col1_num = blockIdx.x * blockDim.x + threadIdx.x;
    int col2_num = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    if (col1_num < col2_num && col2_num < m) {
        float* col1 = array + n * col1_num;
        float* col2 = array + n * col2_num;
        for (int i = 0; i < n; ++i) {
            sum += fabsf(col1[i] - col2[i]);
        }
        result[col1_num * m + col2_num] = sum;
    }
}


__global__ void Rmanhattan_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, float* result) {
    int col1_num = blockIdx.x * blockDim.x + threadIdx.x;
    int col2_num = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    if (col1_num < m && col2_num < m_b) {
        float* col1 = array + n * col1_num;
        float* col2 = array2 + n * col2_num;
        for (int i = 0; i < n; ++i) {
            sum += fabsf(col1[i] - col2[i]);
        }
        result[col2_num * m + col1_num] = sum;
    }
}


// Register-blocked tiled Manhattan (L1): coalesced loads (fast thread index = feature),
// shared-memory column reuse, and a 4x4 register micro-tile per thread. ~4.5-7.5x faster
// than the naive one-thread-per-pair kernel above, bit-identical (prototype:
// manhattan_tiled_proto.cu). Used by both the .Call fast path and the .C block loop.
#define MANH_BM 64

#define MANH_BN 64

#define MANH_BK 16

#define MANH_RM 4

#define MANH_RN 4

// same block: writes the upper triangle D[row*m+col] (row<col); FinalizeManhattan mirrors.
__global__ void Rmanhattan_reg_same_block(const float* A, int n, int m, float* D) {
    int row0 = blockIdx.y * MANH_BM, col0 = blockIdx.x * MANH_BN;
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
    __shared__ float As[MANH_BK][MANH_BM];
    __shared__ float Bs[MANH_BK][MANH_BN];
    float acc[MANH_RM][MANH_RN];
    #pragma unroll
    for (int i = 0; i < MANH_RM; ++i) for (int j = 0; j < MANH_RN; ++j) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < n; k0 += MANH_BK) {
        for (int idx = tid; idx < MANH_BK * MANH_BM; idx += 256) {  // coalesced: consecutive feature
            int f = idx % MANH_BK, o = idx / MANH_BK, kf = k0 + f;
            int ra = row0 + o, ca = col0 + o;
            As[f][o] = (kf < n && ra < m) ? A[kf + (size_t)n * ra] : 0.f;
            Bs[f][o] = (kf < n && ca < m) ? A[kf + (size_t)n * ca] : 0.f;
        }
        __syncthreads();
        int kmax = (n - k0) < MANH_BK ? (n - k0) : MANH_BK;
        for (int kk = 0; kk < kmax; ++kk) {
            float ar[MANH_RM], bc[MANH_RN];
            #pragma unroll
            for (int i = 0; i < MANH_RM; ++i) ar[i] = As[kk][ty * MANH_RM + i];
            #pragma unroll
            for (int j = 0; j < MANH_RN; ++j) bc[j] = Bs[kk][tx * MANH_RN + j];
            #pragma unroll
            for (int i = 0; i < MANH_RM; ++i)
                #pragma unroll
                for (int j = 0; j < MANH_RN; ++j) acc[i][j] += fabsf(ar[i] - bc[j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < MANH_RM; ++i) for (int j = 0; j < MANH_RN; ++j) {
        int row = row0 + ty * MANH_RM + i, col = col0 + tx * MANH_RN + j;
        if (row < m && col < m && row < col) D[(size_t)row * m + col] = acc[i][j];
    }
}

// different blocks: full output D[col2*m+col1], A=array(n x m), B=array2(n x m_b).
__global__ void Rmanhattan_reg_different_blocks(const float* A, const float* B,
                                                int n, int m, int m_b, float* D) {
    int row0 = blockIdx.x * MANH_BM, col0 = blockIdx.y * MANH_BN;  // row0 in m (col1), col0 in m_b (col2)
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
    __shared__ float As[MANH_BK][MANH_BM];
    __shared__ float Bs[MANH_BK][MANH_BN];
    float acc[MANH_RM][MANH_RN];
    #pragma unroll
    for (int i = 0; i < MANH_RM; ++i) for (int j = 0; j < MANH_RN; ++j) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < n; k0 += MANH_BK) {
        for (int idx = tid; idx < MANH_BK * MANH_BM; idx += 256) {
            int f = idx % MANH_BK, o = idx / MANH_BK, kf = k0 + f;
            int ra = row0 + o, cb = col0 + o;
            As[f][o] = (kf < n && ra < m)   ? A[kf + (size_t)n * ra] : 0.f;
            Bs[f][o] = (kf < n && cb < m_b) ? B[kf + (size_t)n * cb] : 0.f;
        }
        __syncthreads();
        int kmax = (n - k0) < MANH_BK ? (n - k0) : MANH_BK;
        for (int kk = 0; kk < kmax; ++kk) {
            float ar[MANH_RM], bc[MANH_RN];
            #pragma unroll
            for (int i = 0; i < MANH_RM; ++i) ar[i] = As[kk][ty * MANH_RM + i];
            #pragma unroll
            for (int j = 0; j < MANH_RN; ++j) bc[j] = Bs[kk][tx * MANH_RN + j];
            #pragma unroll
            for (int i = 0; i < MANH_RM; ++i)
                #pragma unroll
                for (int j = 0; j < MANH_RN; ++j) acc[i][j] += fabsf(ar[i] - bc[j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < MANH_RM; ++i) for (int j = 0; j < MANH_RN; ++j) {
        int col1 = row0 + ty * MANH_RM + i, col2 = col0 + tx * MANH_RN + j;
        if (col1 < m && col2 < m_b) D[(size_t)col2 * m + col1] = acc[i][j];
    }
}




//' Driver Function for calculation of Manhattan matrix for same block.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//'
extern "C" void matrix_Manhattan_distance_same_block(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){

  int array_size = *n * *m;

  float* array_new = new float[*n * *m];

  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  float* d_array;

  cudaMalloc(&d_array, array_size * sizeof(float));

  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);

  float* d_result;
  float* h_result = new float[(*m) * (*m)];
  cudaMalloc(&d_result, (*m) * (*m) * sizeof(float));
  cudaMemset(d_result, 0, (*m) * (*m) * sizeof(float));
  int columns = *m;
  dim3 block_size(16, 16);
  dim3 num_blocks((columns + MANH_BN - 1) / MANH_BN, (columns + MANH_BM - 1) / MANH_BM);

  Rmanhattan_reg_same_block<<<num_blocks, block_size>>>(d_array, *n, *m, d_result);

  gpuErrchk(cudaPeekAtLastError());

  dim3 fin_bs(32, 32), fin_nb((columns + 31) / 32, (columns + 31) / 32);
  FinalizeManhattan<<<fin_nb, fin_bs>>>(columns, d_result);

  cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < (*m) * (*m); ++i) {
    c[i] = h_result[i];
  }

  free(h_result);
  free(array_new);
  cudaFree(d_result);
  cudaFree(d_array);
}


//' Driver Function for calculation of Manhattan matrix for different block.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//'
extern "C" void matrix_Manhattan_distance_different_blocks(double* a, double* b, double* c, int* n, int* m, int* m_b){

  int array_size = *n * *m;
  float* array_new = new float[*n * *m];

  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  int array2_size = *n * (*m_b);
  float* array2_new = new float[array2_size];

  for (int i = 0; i < array2_size; ++i) {
    array2_new[i] = b[i];
  }

  float* d_array;
  float* d_array2;

  cudaMalloc(&d_array, array_size * sizeof(float));
  cudaMalloc(&d_array2, array2_size * sizeof(float));

  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_array2, array2_new, array2_size * sizeof(float), cudaMemcpyHostToDevice);
  int columns = *m;
  int columns_b = *m_b;
  dim3 block_size(16, 16);
  dim3 num_blocks((columns + MANH_BM - 1) / MANH_BM, (columns_b + MANH_BN - 1) / MANH_BN);

  float* d_result;
  float* h_result = new float[(*m) * (*m_b)];
  cudaMalloc(&d_result, (*m) * (*m_b) * sizeof(float));
  cudaMemset(d_result, 0, (*m) * (*m_b) * sizeof(float));

  Rmanhattan_reg_different_blocks<<<num_blocks, block_size>>>(d_array, d_array2, *n, *m, *m_b, d_result);

  cudaMemcpy(h_result, d_result, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
  gpuErrchk(cudaPeekAtLastError());

  for (int i = 0; i < (*m) * (*m_b); ++i) {
    c[i] = h_result[i];
  }

  free(h_result);
  free(array_new);
  free(array2_new);
  cudaFree(d_result);
  cudaFree(d_array);
  cudaFree(d_array2);
}



__global__ void RmanhattanSparse_gpu_atomic_float_same_block(
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
                    atomicAdd(&result[left * columns + col2], fabsf(value2));
                }

                for (int right = col2 + 1; right < next_col; ++right) {
                    atomicAdd(&result[col1 * columns + right], fabsf(value1));
                }

                float diff = fabsf(value1 - value2);
                atomicAdd(&result[col1 * columns + col2], diff);
            }
        }
    }
}


extern "C" void matrix_Manhattan_sparse_distance_same_block(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b) {
    int rows = *num_rows;
    int columns = *num_columns;
    int num_elements_a_int = *num_elements_a;

    float* a_values = new float[num_elements_a_int];
    float* float_result = new float[columns * columns];

    for (int i = 0; i < num_elements_a_int; ++i) {
        a_values[i] = static_cast<float>(a_double_values[i]);
    }

    for (int i = 0; i < columns * columns; ++i) {
        float_result[i] = 0.0f;
    }

    int* d_a_index;
    int* d_a_positions;
    float* d_a_values;
    float* d_result;

    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
    cudaMalloc(&d_result, columns * columns * sizeof(float));

    cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, columns * columns * sizeof(float));

    int threadsPerBlock = 128;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    RmanhattanSparse_gpu_atomic_float_same_block<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_index, d_a_positions, d_a_values, rows, columns, d_result);

    gpuErrchk( cudaPeekAtLastError() );

    dim3 block_size(32, 32);
    dim3 num_blocks((columns + 31) / 32, (columns + 31) / 32);

    FinalizeManhattan<<<num_blocks, block_size>>>(columns, d_result);
    gpuErrchk( cudaPeekAtLastError() );

    cudaMemcpy(float_result, d_result, columns * columns * sizeof(float), cudaMemcpyDeviceToHost);
    gpuErrchk( cudaPeekAtLastError() );

    for (int i = 0; i < columns * columns; ++i) {
        result[i] = float_result[i];
    }

    cudaFree(d_a_index);
    cudaFree(d_a_positions);
    cudaFree(d_a_values);
    cudaFree(d_result);
    gpuErrchk(cudaPeekAtLastError());
    delete[] a_values;
    delete[] float_result;
}


__global__ void RmanhattanSparse_gpu_atomic_float_different_blocks(
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
                        atomicAdd(&result[col2 * columns + left], fabsf(value2));
                    }
                }

                if (col1 < columns) {
                    for (int left = prev_col2 + 1; left < col2; ++left) {
                        atomicAdd(&result[left * columns + col1], fabsf(value1));
                    }
                }

                if (col1 < columns && col2 < columns_b) {
                    float diff = fabsf(value1 - value2);
                    atomicAdd(&result[col2 * columns + col1], diff);
                }
            }
        }
    }
}


extern "C" void matrix_Manhattan_sparse_distance_different_blocks(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b) {

    int rows = *num_rows;
    int columns = *num_columns;
    int columns_b = *num_columns_b;
    int num_elements_a_int = *num_elements_a;
    int num_elements_b_int = *num_elements_b;

    float* a_values = new float[num_elements_a_int];
    float* b_values = new float[num_elements_b_int];
    float* float_result = new float[columns * columns_b];

    for (int i = 0; i < num_elements_a_int; ++i) {
        a_values[i] = static_cast<float>(a_double_values[i]);
    }

    for (int i = 0; i < num_elements_b_int; ++i) {
        b_values[i] = static_cast<float>(b_double_values[i]);
    }

    for (int i = 0; i < columns * columns_b; ++i) {
        float_result[i] = 0.0f;
    }

    int* d_a_index;
    int* d_a_positions;
    float* d_a_values;
    int* d_b_index;
    int* d_b_positions;
    float* d_b_values;
    float* d_result;

    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
    cudaMalloc(&d_b_index, num_elements_b_int * sizeof(int));
    cudaMalloc(&d_b_positions, (rows + 1) * sizeof(int));
    cudaMalloc(&d_b_values, num_elements_b_int * sizeof(float));
    cudaMalloc(&d_result, columns * columns_b * sizeof(float));
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b_index, b_index, num_elements_b_int * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b_positions, b_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b_values, b_values, num_elements_b_int * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_result, 0, columns * columns_b * sizeof(float)));

    gpuErrchk(cudaPeekAtLastError());
    int threadsPerBlock = 128;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    RmanhattanSparse_gpu_atomic_float_different_blocks<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_index, d_a_positions, d_a_values,
        d_b_index, d_b_positions, d_b_values, rows, columns, columns_b, d_result);

    gpuErrchk( cudaPeekAtLastError() );
    cudaMemcpy(float_result, d_result, columns * columns_b * sizeof(float), cudaMemcpyDeviceToHost);
    gpuErrchk( cudaPeekAtLastError() );

    for (int i = 0; i < columns * columns_b; ++i) {
        result[i] = float_result[i];
    }

    cudaFree(d_a_index);
    cudaFree(d_a_positions);
    cudaFree(d_a_values);
    cudaFree(d_b_index);
    cudaFree(d_b_positions);
    cudaFree(d_b_values);
    cudaFree(d_result);

    delete[] a_values;
    delete[] b_values;
    delete[] float_result;
}


// ─── Manhattan ───

__global__ void RmanhattanSparse_per_cell_pair_same_block(
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
      acc += fabsf(csc_x[ia]); ++ia;
    } else if (ib < eb && (ia >= ea || csc_i[ib] < csc_i[ia])) {
      acc += fabsf(csc_x[ib]); ++ib;
    } else {
      acc += fabsf(csc_x[ia] - csc_x[ib]); ++ia; ++ib;
    }
  }
  result_out[cell_a * n_cells + cell_b] = acc;
}


__global__ void RmanhattanSparse_per_cell_pair_different_blocks(
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
      acc += fabsf(a_csc_x[ia]); ++ia;
    } else if (ib < eb && (ia >= ea || b_csc_i[ib] < a_csc_i[ia])) {
      acc += fabsf(b_csc_x[ib]); ++ib;
    } else {
      acc += fabsf(a_csc_x[ia] - b_csc_x[ib]); ++ia; ++ib;
    }
  }
  result_out[cell_b * n_cells_a + cell_a] = acc;
}


// ─── Drivers: Manhattan ───

extern "C" void matrix_Manhattan_sparse_per_cell_pair_distance_same_block(
    int* csc_i_in, int* csc_p_in, double* csc_x_in,
    int* /*b*/, int* /*b*/, double* /*b*/,
    double* result, int* num_rows, int* num_columns,
    int* /*num_columns_b*/, int* num_elements, int* /*num_elements_b*/)
{
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

  RmanhattanSparse_per_cell_pair_same_block<<<blocks, threads>>>(d_p, d_i, d_x, n_cells, d_res);
  gpuErrchk(cudaPeekAtLastError());

  FinalizePerCellPairFloat<<<blocks, threads>>>(n_cells, d_res, d_out);
  gpuErrchk(cudaPeekAtLastError());

  cudaMemcpy(result, d_out, n_cells * n_cells * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_i); cudaFree(d_p); cudaFree(d_x); cudaFree(d_res); cudaFree(d_out);
}


extern "C" void matrix_Manhattan_sparse_per_cell_pair_distance_different_blocks(
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

  RmanhattanSparse_per_cell_pair_different_blocks<<<blocks, threads>>>(
      d_ap, d_ai, d_ax, d_bp, d_bi, d_bx, n_cells_a, n_cells_b, d_res);
  gpuErrchk(cudaPeekAtLastError());

  std::vector<float> h_res(n_cells_a * n_cells_b);
  cudaMemcpy(h_res.data(), d_res, n_cells_a * n_cells_b * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n_cells_a * n_cells_b; ++i) result[i] = (double)h_res[i];

  cudaFree(d_ai); cudaFree(d_ap); cudaFree(d_ax);
  cudaFree(d_bi); cudaFree(d_bp); cudaFree(d_bx);
  cudaFree(d_res);
}


// ============== .Call fast path: ALL dense metrics, full matrix ==============
// Device-side same-block compute factored out of the per-block .C drivers so
// the .Call entry below can run the WHOLE m x m matrix in one launch off a
// resident float copy (one upload, one download), bypassing the block loop's
// per-batch re-conversion + re-upload. Kernels grid-stride over all pairs, so
// a single launch is correct for any m.

// Manhattan: L1 atomic kernel + on-device symmetry mirror (FinalizeManhattan).
void pc_manhattan_same_block_device(const float* d_A, int n, int m,
                                           float* d_D) {
    PcKernelTimer _kt;
    cudaMemset(d_D, 0, (size_t)m * m * sizeof(float));
    dim3 bs(16, 16), nb((m + MANH_BN - 1) / MANH_BN, (m + MANH_BM - 1) / MANH_BM);
    Rmanhattan_reg_same_block<<<nb, bs>>>(d_A, n, m, d_D);   // tiled + register-blocked
    dim3 fbs(32, 32), fnb((m + 31) / 32, (m + 31) / 32);
    FinalizeManhattan<<<fnb, fbs>>>(m, d_D);                 // mirror upper -> lower
    gpuErrchk(cudaPeekAtLastError());
}

// Full Manhattan tile (no triangle): A=(n x mA), B=(n x mB) -> d_tile[col2*mA + col1].
// Diagonal tile: pass B=A, mB=mA (diagonal entries fall out to 0). Used by the batched path.
void pc_manhattan_tile_device(const float* d_A, const float* d_B,
                                     int n, int mA, int mB, float* d_tile) {
    PcKernelTimer _kt;
    dim3 bs(16, 16), nb((mA + MANH_BM - 1) / MANH_BM, (mB + MANH_BN - 1) / MANH_BN);
    Rmanhattan_reg_different_blocks<<<nb, bs>>>(d_A, d_B, n, mA, mB, d_tile);
    gpuErrchk(cudaPeekAtLastError());
}
