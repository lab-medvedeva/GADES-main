#include "matrixgpufunctions.h"


__global__ void matrix_mul_gpu(const datatype_t* a, const datatype_t* b, datatype_t* c, const int n, const int m, const int k){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < k){
    datatype_t tmp = 0;
    for(int i = 0; i < m; i++){
      tmp += a[row * m + i] * b[i * k + col];
    }
    c[row * k + col] = tmp;
  }
}


__global__ void matrix_dist_L1_gpu(const datatype_t* a, datatype_t* b, const int n, const int m){
  int col1 = blockIdx.y * blockDim.y + threadIdx.y;
  int col2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (col1 < col2 && col2 < m){
    b[col1 * m + col2] = L1distance(a, n, m, col1, col2);
    b[col2 * m + col1] = b[col1 * m + col2];
  }
}


__global__ void matrix_dist_L2_gpu(const datatype_t* a, datatype_t* b, const int n, const int m){
  int col1 = blockIdx.y * blockDim.y + threadIdx.y;
  int col2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (col1 < col2 && col2 < m){
    b[col1 * m + col2] = L2distance(a, n, m, col1, col2);
    b[col2 * m + col1] = b[col1 * m + col2];
  }
}


__global__ void matrix_dist_Linf_gpu(const datatype_t* a, datatype_t* b, const int n, const int m){
  int col1 = blockIdx.y * blockDim.y + threadIdx.y;
  int col2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (col1 < col2 && col2 < m){
    b[col1 * m + col2] = Linfdistance(a, n, m, col1, col2);
    b[col2 * m + col1] = b[col1 * m + col2];
  }
}


__global__ void matrix_dist_Kendall_gpu(const datatype_t* a, datatype_t* b, const int n, const int m, int* R){
  int col1 = blockIdx.y * blockDim.y + threadIdx.y;
  int col2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (col1 < col2 && col2 < m){

    int threads = 16;
    int blocks_in_row = (n + threads - 1) / threads;
    int blocks_in_col = (n + threads - 1) / threads;

    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks_in_row, blocks_in_col);

    Rkendall_gpu<<<BLOCKS, THREADS>>>(a, n, m, col1, col2, R);
    cudaDeviceSynchronize();

    b[col1 * m + col2] = R[col1 * m + col2] * 2.0 / n / (n - 1);
    b[col2 * m + col1] = b[col1 * m + col2];
  }
}

__global__ void matrix_dist_Kendall_gpu_naive(const datatype_t* a, datatype_t* b, const int n, const int m, int* R){
  int col1 = blockIdx.y * blockDim.y + threadIdx.y;
  int col2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (col1 < col2 && col2 < m){
    for (int row2 = 0; row2 < n; row2++){
      for (int row1 = 0; row1 < row2; row1++){
        if ((a[row1 * m + col1] - a[row2 * m + col1]) * (a[row1 * m + col2] - a[row2 * m + col2]) < 0){
          R[col1 * m + col2]++;
        }
      }
    }
    b[col1 * m + col2] = R[col1 * m + col2] * 2.0 / n / (n - 1);
    b[col2 * m + col1] = b[col1 * m + col2];
  }
}
