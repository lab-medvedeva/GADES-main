#include "dmath.h"


__device__ datatype_t L1distance(const datatype_t* a, const int n, const int m, const int col1, const int col2){
  datatype_t ans = 0;
  for(int i = 0; i < n; i++){
    ans += fabs(a[i * m + col1] - a[i * m + col2]);
  }
  return ans;
}


__device__ datatype_t L2distance(const datatype_t* a, const int n, const int m, const int col1, const int col2){
  datatype_t ans = 0;
  for(int i = 0; i < n; i++){
    ans += (a[i * m + col1] - a[i * m + col2]) * (a[i * m + col1] - a[i * m + col2]);
  }
  return sqrt(ans);
}


__device__ datatype_t Linfdistance(const datatype_t* a, const int n, const int m, const int col1, const int col2){
  datatype_t ans = 0;
  for(int i = 0; i < n; i++){
    ans = max(ans, fabs(a[i * m + col1] - a[i * m + col2]));
  }
  return ans;
}


__global__ void Rkendall_gpu(const datatype_t* a, const int n, const int m, const int col1, const int col2, int* R){
  int row1 = blockIdx.y * blockDim.y + threadIdx.y;
  int row2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (row1 < row2 && row2 < n){
    if ((a[row1 * m + col1] - a[row2 * m + col1]) * (a[row1 * m + col2] - a[row2 * m + col2]) < 0){
      atomicAdd((R + col1 * m + col2), 1);
    }
  }
}
