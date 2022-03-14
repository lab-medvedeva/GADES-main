#ifndef DMATH_H
#define DMATH_H

typedef double datatype_t;


__device__ datatype_t L1distance(const datatype_t* a, const int n, const int m, const int col1, const int col2);


__device__ datatype_t L2distance(const datatype_t* a, const int n, const int m, const int col1, const int col2);


__device__ datatype_t Linfdistance(const datatype_t* a, const int n, const int m, const int col1, const int col2);


__global__ void Rkendall_gpu(const datatype_t* a, const int n, const int m, const int col1, const int col2, int* R);

#endif
