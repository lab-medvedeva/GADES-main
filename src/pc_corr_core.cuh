#ifndef PC_CORR_CORE_CUH
#define PC_CORR_CORE_CUH
#include <R.h>
#include <Rinternals.h>
#include <cuda_runtime.h>

// Shared cross-TU declarations (ADR-0002, candidate 2). Host drivers +
// kernels launched from more than one metric TU. Keep external linkage.
void pc_cosine_same_block_device(const float* d_A, int n, int m, float* d_D);
void pc_cosine_different_blocks_device(const float* d_A, const float* d_B, int n, int m, int m_b, float* d_D);
__global__ void PCEuclidean_finalize_same_kernel(int m, float* G, const float* sq);
__global__ void PCEuclidean_finalize_xy_kernel(int m, int m_b, float* G, const float* sq_a, const float* sq_b);
void pc_euclidean_same_block_device(const float* d_A, int n, int m, float* d_D);
void pc_euclidean_different_blocks_device(const float* d_A, const float* d_B, int n, int m, int m_b, float* d_D);
void pc_drive_cosine_same(double* a, double* c, int n, int m, bool center);
void pc_drive_cosine_diff(double* a, double* b, double* c, int n, int m, int m_b, bool center);
__global__ void FinalizePerCellPairFloat( int n_cells, const float* __restrict__ in, double* __restrict__ out);
__global__ void FinalizeCosine(int columns, float* results, float* x_squares, float* y_squares);
__global__ void RcosineCorr_gpu_atomic_float_same_block( float* array, const int n, const int m, float* scalar_product, float* x_norm, float* y_norm );
__global__ void RcosineCorr_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, float* scalar_prod,float *x_norm, float* y_norm );

// --- hand-maintained cross-TU decls (dispatch.cu <-> metric TUs) ---
void pc_manhattan_same_block_device(const float* d_A, int n, int m, float* d_D);
void pc_manhattan_tile_device(const float* d_A, const float* d_B, int n, int mA, int mB, float* d_tile);
void pc_kendall_same_block_device(const float* d_A, int N, int M, int* d_disc);
void pc_drive_sparse_cosine_same(int* a_index, int* a_positions, double* a_values, double* c, int n, int m, int nnz, bool center);
void pc_drive_sparse_cosine_diff(int* a_index, int* a_positions, double* a_values, int* b_index, int* b_positions, double* b_values, double* c, int n, int m, int m_b, int nnz_a, int nnz_b, bool center);
extern "C" {
void matrix_Manhattan_sparse_per_cell_pair_distance_same_block(int*,int*,double*,int*,int*,double*,double*,int*,int*,int*,int*,int*);
void matrix_Spearman_sparse_per_cell_pair_distance_same_block(int*,int*,double*,int*,int*,double*,double*,int*,int*,int*,int*,int*);
void matrix_Kendall_sparse_per_cell_pair_distance_same_block(int*,int*,double*,int*,int*,double*,double*,int*,int*,int*,int*,int*);
}

#endif
