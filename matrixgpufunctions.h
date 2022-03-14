#ifndef MATRIXGPUFUNCTIONS_H
#define MATRIXGPUFUNCTIONS_H

#include <thrust/device_vector.h>
#include "dmath.h"

typedef double datatype_t;


__global__ void matrix_mul_gpu(const datatype_t* a, const datatype_t* b, datatype_t* c, const int n, const int m, const int k);


__global__ void matrix_dist_L1_gpu(const datatype_t* a, datatype_t* b, const int n, const int m);


__global__ void matrix_dist_L2_gpu(const datatype_t* a, datatype_t* b, const int n, const int m);


__global__ void matrix_dist_Linf_gpu(const datatype_t* a, datatype_t* b, const int n, const int m);


__global__ void matrix_dist_Kendall_gpu(const datatype_t* a, datatype_t* b, const int n, const int m, int* R);


__global__ void matrix_dist_Kendall_gpu_naive(const datatype_t* a, datatype_t* b, const int n, const int m, int* R);

#endif
