#include <iostream>
#include <R.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "matrixgpufunctions.h"

typedef double datatype_t;


extern "C" void matrix_multiplication(datatype_t* a, datatype_t* b, datatype_t* c, int* n1, int* m1, int* n2, int* m2){
  if (*m1 == *n2){
    thrust::device_vector<datatype_t> da(a, a + *n1 * *m1);
    thrust::device_vector<datatype_t> db(b, b + *n2 * *m2);

    thrust::device_vector<datatype_t> dc(*n1 * *m2, 0);
    datatype_t* da_ptr = thrust::raw_pointer_cast(da.data());
    datatype_t* db_ptr = thrust::raw_pointer_cast(db.data());
    datatype_t* dc_ptr = thrust::raw_pointer_cast(dc.data());

    int threads = 16;
    int blocks_in_row = (*m2 + threads - 1) / threads;
    int blocks_in_col = (*n1 + threads - 1) / threads;

    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks_in_row, blocks_in_col);

    matrix_mul_gpu<<<BLOCKS, THREADS>>>(da_ptr, db_ptr, dc_ptr, *n1, *m1, *m2);
    cudaDeviceSynchronize();

    for(int i = 0; i < *n1; i++){
      for(int j = 0; j < *m2; j++){
        c[j * *n1 + i] = dc[i * *m2 + j];
      }
    }
  }
}


extern "C" void matrix_L1_distance(datatype_t* a, datatype_t* c, int* n, int* m){
  thrust::device_vector<datatype_t> d_matrix(a, a + *n * *m);

  thrust::device_vector<datatype_t> d_dist_matrix(*m * *m, 0);
  datatype_t* dm_ptr = thrust::raw_pointer_cast(d_matrix.data());
  datatype_t* ddm_ptr = thrust::raw_pointer_cast(d_dist_matrix.data());

  int threads = 16;
  int blocks_in_row = (*m + threads - 1) / threads;
  int blocks_in_col = (*m + threads - 1) / threads;

  dim3 THREADS(threads, threads);
  dim3 BLOCKS(blocks_in_row, blocks_in_col);

  matrix_dist_L1_gpu<<<BLOCKS, THREADS>>>(dm_ptr, ddm_ptr, *n, *m);
  cudaDeviceSynchronize();

  thrust::copy(d_dist_matrix.begin(), d_dist_matrix.end(), c);

}


extern "C" void matrix_L2_distance(datatype_t* a, datatype_t* c, int* n, int* m){
  thrust::device_vector<datatype_t> d_matrix(a, a + *n * *m);

  thrust::device_vector<datatype_t> d_dist_matrix(*m * *m, 0);
  datatype_t* dm_ptr = thrust::raw_pointer_cast(d_matrix.data());
  datatype_t* ddm_ptr = thrust::raw_pointer_cast(d_dist_matrix.data());

  int threads = 16;
  int blocks_in_row = (*m + threads - 1) / threads;
  int blocks_in_col = (*m + threads - 1) / threads;

  dim3 THREADS(threads, threads);
  dim3 BLOCKS(blocks_in_row, blocks_in_col);

  matrix_dist_L2_gpu<<<BLOCKS, THREADS>>>(dm_ptr, ddm_ptr, *n, *m);
  cudaDeviceSynchronize();

  thrust::copy(d_dist_matrix.begin(), d_dist_matrix.end(), c);

}


extern "C" void matrix_Linf_distance(datatype_t* a, datatype_t* c, int* n, int* m){
  thrust::device_vector<datatype_t> d_matrix(a, a + *n * *m);

  thrust::device_vector<datatype_t> d_dist_matrix(*m * *m, 0);
  datatype_t* dm_ptr = thrust::raw_pointer_cast(d_matrix.data());
  datatype_t* ddm_ptr = thrust::raw_pointer_cast(d_dist_matrix.data());

  int threads = 16;
  int blocks_in_row = (*m + threads - 1) / threads;
  int blocks_in_col = (*m + threads - 1) / threads;

  dim3 THREADS(threads, threads);
  dim3 BLOCKS(blocks_in_row, blocks_in_col);

  matrix_dist_Linf_gpu<<<BLOCKS, THREADS>>>(dm_ptr, ddm_ptr, *n, *m);
  cudaDeviceSynchronize();

  thrust::copy(d_dist_matrix.begin(), d_dist_matrix.end(), c);

}


extern "C" void matrix_Kendall_distance(datatype_t* a, datatype_t* c, int* n, int* m){
  thrust::device_vector<datatype_t> d_matrix(a, a + *n * *m);

  thrust::device_vector<datatype_t> d_dist_matrix(*m * *m, 0);
  thrust::device_vector<int> R(*m * *m, 0);
  datatype_t* dm_ptr = thrust::raw_pointer_cast(d_matrix.data());
  datatype_t* ddm_ptr = thrust::raw_pointer_cast(d_dist_matrix.data());
  int* r_ptr = thrust::raw_pointer_cast(R.data());

  int threads = 16;
  int blocks_in_row = (*m + threads - 1) / threads;
  int blocks_in_col = (*m + threads - 1) / threads;

  dim3 THREADS(threads, threads);
  dim3 BLOCKS(blocks_in_row, blocks_in_col);

  matrix_dist_Kendall_gpu<<<BLOCKS, THREADS>>>(dm_ptr, ddm_ptr, *n, *m, r_ptr);
  cudaDeviceSynchronize();

  thrust::copy(d_dist_matrix.begin(), d_dist_matrix.end(), c);

}


extern "C" void matrix_Kendall_distance_naive(datatype_t* a, datatype_t* c, int* n, int* m){
  thrust::device_vector<datatype_t> d_matrix(a, a + *n * *m);

  thrust::device_vector<datatype_t> d_dist_matrix(*m * *m, 0);
  thrust::device_vector<int> R(*m * *m, 0);
  datatype_t* dm_ptr = thrust::raw_pointer_cast(d_matrix.data());
  datatype_t* ddm_ptr = thrust::raw_pointer_cast(d_dist_matrix.data());
  int* r_ptr = thrust::raw_pointer_cast(R.data());

  int threads = 16;
  int blocks_in_row = (*m + threads - 1) / threads;
  int blocks_in_col = (*m + threads - 1) / threads;

  dim3 THREADS(threads, threads);
  dim3 BLOCKS(blocks_in_row, blocks_in_col);

  matrix_dist_Kendall_gpu_naive<<<BLOCKS, THREADS>>>(dm_ptr, ddm_ptr, *n, *m, r_ptr);
  cudaDeviceSynchronize();

  thrust::copy(d_dist_matrix.begin(), d_dist_matrix.end(), c);

}
