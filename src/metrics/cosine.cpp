#include <iostream>
#include <fstream>
#include <R.h>
#include <Rinternals.h>
#include <stdio.h>
#include <cstring>
#include <math.h>
#include <cassert>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <cblas.h>
#include "pc_runtime_cpu.h"
#include "pc_linalg_cpu.h"
#include "pc_corr_core_cpu.h"



// pc_sparse_to_dense_cpu moved to pc_linalg_cpu (ADR-0002).

void pc_drive_cpu_sparse_cosine_same(int* a_index, int* a_positions,
                                            double* a_values, double* c,
                                            int n, int m, int nnz,
                                            bool center) {
    float* A = pc_sparse_to_dense_cpu(a_index, a_positions, a_values, n, m,
                                      nnz);
    if (center) pc_center_columns(A, n, m);
    std::vector<float> D((size_t)m * m);
    pc_cosine_same_block_cpu(A, n, m, D.data());
    for (size_t i = 0; i < (size_t)m * m; ++i) c[i] = (double)D[i];
    delete[] A;
}


void pc_drive_cpu_sparse_cosine_diff(int* a_index, int* a_positions,
                                            double* a_values,
                                            int* b_index, int* b_positions,
                                            double* b_values,
                                            double* c, int n, int m, int m_b,
                                            int nnz_a, int nnz_b,
                                            bool center) {
    float* A = pc_sparse_to_dense_cpu(a_index, a_positions, a_values, n, m,
                                      nnz_a);
    float* B = pc_sparse_to_dense_cpu(b_index, b_positions, b_values, n, m_b,
                                      nnz_b);
    if (center) {
        pc_center_columns(A, n, m);
        pc_center_columns(B, n, m_b);
    }
    std::vector<float> D((size_t)m * m_b);
    pc_cosine_different_blocks_cpu(A, B, n, m, m_b, D.data());
    for (size_t i = 0; i < (size_t)m * m_b; ++i) c[i] = (double)D[i];
    delete[] A; delete[] B;
}


extern "C" void  matrix_Cosine_distance_same_block_cpu(
  double* a, double* b, double* c,
  int* n, int* m, int* m_b
) {
  pc_drive_cpu_cosine_same(a, c, *n, *m, /*center=*/false);
}


extern "C" void  matrix_Cosine_distance_different_blocks_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b){
  pc_drive_cpu_cosine_diff(a, b, c, *n, *m, *m_b, /*center=*/false);
}



extern "C" void matrix_Cosine_sparse_distance_same_block_cpu(
  int *a_index, int *a_positions, double *a_double_values,
  int *b_index, int *b_positions, double *b_double_values,
  double *result, int *num_rows, int *num_columns, int *num_columns_b,
  int *num_elements_a, int *num_elements_b
){
  pc_drive_cpu_sparse_cosine_same(a_index, a_positions, a_double_values,
                                  result, *num_rows, *num_columns,
                                  *num_elements_a, /*center=*/false);
}


extern "C" void  matrix_Cosine_sparse_distance_different_blocks_cpu(
  int *a_index, int *a_positions, double *a_double_values,
  int *b_index, int *b_positions, double *b_double_values,
  double *result, int *num_rows, int *num_columns, int *num_columns_b,
  int *num_elements_a, int *num_elements_b
){
  pc_drive_cpu_sparse_cosine_diff(a_index, a_positions, a_double_values,
                                  b_index, b_positions, b_double_values,
                                  result, *num_rows, *num_columns,
                                  *num_columns_b, *num_elements_a,
                                  *num_elements_b, /*center=*/false);
}


// ─── Cosine ───

static float cosine_per_cell_pair_merge(
    const int* a_i, const float* a_x, int ia, int ea, float norm_a,
    const int* b_i, const float* b_x, int ib, int eb, float norm_b)
{
  float dot = 0;
  while (ia < ea && ib < eb) {
    if (a_i[ia] < b_i[ib]) ++ia;
    else if (b_i[ib] < a_i[ia]) ++ib;
    else { dot += a_x[ia] * b_x[ib]; ++ia; ++ib; }
  }
  return 1.0f - dot / (norm_a * norm_b);
}


static float compute_norm(const float* x, int start, int end) {
  float s = 0;
  for (int k = start; k < end; ++k) s += x[k] * x[k];
  return sqrtf(s);
}


extern "C" void matrix_Cosine_sparse_per_cell_pair_distance_same_block_cpu(
    int* csc_i, int* csc_p, double* csc_x_double,
    int* /*b*/, int* /*b*/, double* /*b*/,
    double* result, int* num_rows, int* num_columns,
    int* /*num_columns_b*/, int* num_elements_a, int* /*num_elements_b*/)
{
  int n_cells = *num_columns;
  int nnz = *num_elements_a;
  float* csc_x = new float[nnz];
  for (int k = 0; k < nnz; ++k) csc_x[k] = static_cast<float>(csc_x_double[k]);

  float* norms = new float[n_cells];
  for (int c = 0; c < n_cells; ++c) norms[c] = compute_norm(csc_x, csc_p[c], csc_p[c + 1]);

  #pragma omp parallel for schedule(dynamic)
  for (int ca = 0; ca < n_cells; ++ca) {
    for (int cb = ca + 1; cb < n_cells; ++cb) {
      float d = cosine_per_cell_pair_merge(
          csc_i, csc_x, csc_p[ca], csc_p[ca + 1], norms[ca],
          csc_i, csc_x, csc_p[cb], csc_p[cb + 1], norms[cb]);
      result[ca * n_cells + cb] = d;
      result[cb * n_cells + ca] = d;
    }
    result[ca * n_cells + ca] = 0.0;
  }
  delete[] csc_x;
  delete[] norms;
}


extern "C" void matrix_Cosine_sparse_per_cell_pair_distance_different_blocks_cpu(
    int* a_csc_i, int* a_csc_p, double* a_xd,
    int* b_csc_i, int* b_csc_p, double* b_xd,
    double* result, int* num_rows, int* num_columns,
    int* num_columns_b, int* num_elements_a, int* num_elements_b)
{
  int n_cells_a = *num_columns;
  int n_cells_b = *num_columns_b;
  int nnz_a = *num_elements_a;
  int nnz_b = *num_elements_b;
  float* a_x = new float[nnz_a];
  float* b_x = new float[nnz_b];
  for (int k = 0; k < nnz_a; ++k) a_x[k] = static_cast<float>(a_xd[k]);
  for (int k = 0; k < nnz_b; ++k) b_x[k] = static_cast<float>(b_xd[k]);

  float* a_norms = new float[n_cells_a];
  float* b_norms = new float[n_cells_b];
  for (int c = 0; c < n_cells_a; ++c) a_norms[c] = compute_norm(a_x, a_csc_p[c], a_csc_p[c + 1]);
  for (int c = 0; c < n_cells_b; ++c) b_norms[c] = compute_norm(b_x, b_csc_p[c], b_csc_p[c + 1]);

  #pragma omp parallel for schedule(dynamic)
  for (int ca = 0; ca < n_cells_a; ++ca) {
    for (int cb = 0; cb < n_cells_b; ++cb) {
      float d = cosine_per_cell_pair_merge(
          a_csc_i, a_x, a_csc_p[ca], a_csc_p[ca + 1], a_norms[ca],
          b_csc_i, b_x, b_csc_p[cb], b_csc_p[cb + 1], b_norms[cb]);
      result[cb * n_cells_a + ca] = d;
    }
  }
  delete[] a_x;  delete[] b_x;
  delete[] a_norms; delete[] b_norms;
}
