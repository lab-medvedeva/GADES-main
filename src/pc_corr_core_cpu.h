#ifndef PC_CORR_CORE_CPU_H
#define PC_CORR_CORE_CPU_H
#include <R.h>
#include <Rinternals.h>

// Shared cross-TU declarations (ADR-0002, candidate 2). Host drivers +
// kernels launched from more than one metric TU. Keep external linkage.
void pc_fallback_cosine_diff(const float* A, const float* B, int n, int m, int m_b, float* D, const float* xn, const float* yn);
void pc_fallback_cosine_same(const float* A, int n, int m, float* D, const float* norms);
void pc_cosine_same_block_cpu(const float* A, int n, int m, float* D);
void pc_cosine_different_blocks_cpu(const float* A, const float* B, int n, int m, int m_b, float* D);
void pc_drive_cpu_cosine_same(double* a, double* c, int n, int m, bool center);
void pc_euclidean_same_block_cpu(const float* A, int n, int m, float* D);
void pc_euclidean_different_blocks_cpu(const float* A, const float* B, int n, int m, int m_b, float* D);
void pc_drive_cpu_cosine_diff(double* a, double* b, double* c, int n, int m, int m_b, bool center);
void pc_drive_cpu_sparse_cosine_same(int* a_index, int* a_positions, double* a_values, double* c, int n, int m, int nnz, bool center);
void pc_drive_cpu_sparse_cosine_diff(int* a_index, int* a_positions, double* a_values, int* b_index, int* b_positions, double* b_values, double* c, int n, int m, int m_b, int nnz_a, int nnz_b, bool center);
extern "C" void matrix_Euclidean_distance_same_block_cpu(double * a, double * b, double * c, int * n, int * m, int * m_b);
extern "C" void matrix_Kendall_distance_same_block_cpu(double * a, double * b, double * c, int * n, int * m, int * m_b);
extern "C" void matrix_Cosine_distance_same_block_cpu( double* a, double* b, double* c, int* n, int* m, int* m_b );
extern "C" void matrix_Manhattan_distance_same_block_cpu(double * a, double * b, double * c, int * n, int * m, int * m_b);
extern "C" void matrix_Spearman_distance_same_block_cpu( double* a, double* b, double* c, int* n, int* m, int* m_b );
extern "C" void matrix_Pearson_distance_same_block_cpu( double* a, double* b, double* c, int* n, int* m, int* m_b );
extern "C" void matrix_Kendall_sparse_per_cell_pair_distance_same_block_cpu( int* csc_i, int* csc_p, double* csc_x_double, int* , int* , double* , double* result, int* num_rows, int* num_columns, int* , int* num_elements_a, int* );
extern "C" void matrix_Manhattan_sparse_per_cell_pair_distance_same_block_cpu( int* csc_i, int* csc_p, double* csc_x_double, int* , int* , double* , double* result, int* num_rows, int* num_columns, int* , int* num_elements_a, int* );
extern "C" void matrix_Spearman_sparse_per_cell_pair_distance_same_block_cpu( int* csc_i, int* csc_p, double* csc_x_double, int* , int* , double* , double* result, int* num_rows, int* num_columns, int* , int* num_elements_a, int* );

#endif
