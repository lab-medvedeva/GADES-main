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



static void pc_drive_cpu_euclidean_same(double* a, double* c, int n, int m) {
    size_t sz = (size_t)n * m;
    std::vector<float> h(sz);
    for (size_t i = 0; i < sz; ++i) h[i] = (float)a[i];
    std::vector<float> D((size_t)m * m);
    pc_euclidean_same_block_cpu(h.data(), n, m, D.data());
    for (size_t i = 0; i < (size_t)m * m; ++i) c[i] = (double)D[i];
}


static void pc_drive_cpu_euclidean_diff(double* a, double* b, double* c, int n,
                                        int m, int m_b) {
    size_t szA = (size_t)n * m;
    size_t szB = (size_t)n * m_b;
    std::vector<float> hA(szA), hB(szB);
    for (size_t i = 0; i < szA; ++i) hA[i] = (float)a[i];
    for (size_t i = 0; i < szB; ++i) hB[i] = (float)b[i];
    std::vector<float> D((size_t)m * m_b);
    pc_euclidean_different_blocks_cpu(hA.data(), hB.data(), n, m, m_b, D.data());
    for (size_t i = 0; i < (size_t)m * m_b; ++i) c[i] = (double)D[i];
}


static void pc_drive_cpu_sparse_euclidean_same(int* a_index, int* a_positions,
                                               double* a_values, double* c,
                                               int n, int m, int nnz) {
    float* A = pc_sparse_to_dense_cpu(a_index, a_positions, a_values, n, m,
                                      nnz);
    std::vector<float> D((size_t)m * m);
    pc_euclidean_same_block_cpu(A, n, m, D.data());
    for (size_t i = 0; i < (size_t)m * m; ++i) c[i] = (double)D[i];
    delete[] A;
}


static void pc_drive_cpu_sparse_euclidean_diff(int* a_index, int* a_positions,
                                               double* a_values,
                                               int* b_index, int* b_positions,
                                               double* b_values,
                                               double* c, int n, int m,
                                               int m_b, int nnz_a,
                                               int nnz_b) {
    float* A = pc_sparse_to_dense_cpu(a_index, a_positions, a_values, n, m,
                                      nnz_a);
    float* B = pc_sparse_to_dense_cpu(b_index, b_positions, b_values, n, m_b,
                                      nnz_b);
    std::vector<float> D((size_t)m * m_b);
    pc_euclidean_different_blocks_cpu(A, B, n, m, m_b, D.data());
    for (size_t i = 0; i < (size_t)m * m_b; ++i) c[i] = (double)D[i];
    delete[] A; delete[] B;
}

//=========================================

// parallel_for_with_id / get_num_threads / parallel_for_cols moved to pc_runtime_cpu.

//Naive Implementation of Euclidean_distance_matrix (BruteForce)
extern "C" void matrix_Euclidean_distance_same_block_cpu(double * a, double * b, double * c, int * n, int * m, int * m_b) {
  pc_drive_cpu_euclidean_same(a, c, *n, *m);
}

//======================================================

extern "C" void matrix_Euclidean_distance_different_blocks_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b) {
  pc_drive_cpu_euclidean_diff(a, b, c, *n, *m, *m_b);
}



extern "C" void  matrix_Euclidean_sparse_distance_same_block_cpu(
  int *a_index, int *a_positions, double *a_double_values,
  int *b_index, int *b_positions, double *b_double_values,
  double *result, int *num_rows, int *num_columns, int *num_columns_b,
  int *num_elements_a, int *num_elements_b
){
  pc_drive_cpu_sparse_euclidean_same(a_index, a_positions, a_double_values,
                                     result, *num_rows, *num_columns,
                                     *num_elements_a);
}


extern "C" void  matrix_Euclidean_sparse_distance_different_blocks_cpu(
  int *a_index, int *a_positions, double *a_double_values,
  int *b_index, int *b_positions, double *b_double_values,
  double *result, int *num_rows, int *num_columns, int *num_columns_b,
  int *num_elements_a, int *num_elements_b
){
  pc_drive_cpu_sparse_euclidean_diff(a_index, a_positions, a_double_values,
                                     b_index, b_positions, b_double_values,
                                     result, *num_rows, *num_columns,
                                     *num_columns_b, *num_elements_a,
                                     *num_elements_b);
}


// ==================== Per-cell-pair sparse: Euclidean / Manhattan / Cosine / Pearson ====================

// ─── Euclidean ───

static float euclidean_per_cell_pair_merge(
    const int* a_i, const float* a_x, int ia, int ea,
    const int* b_i, const float* b_x, int ib, int eb)
{
  float acc = 0;
  while (ia < ea || ib < eb) {
    if (ia < ea && (ib >= eb || a_i[ia] < b_i[ib])) {
      float v = a_x[ia]; acc += v * v; ++ia;
    } else if (ib < eb && (ia >= ea || b_i[ib] < a_i[ia])) {
      float v = b_x[ib]; acc += v * v; ++ib;
    } else {
      float d = a_x[ia] - b_x[ib]; acc += d * d; ++ia; ++ib;
    }
  }
  return sqrtf(acc);
}


extern "C" void matrix_Euclidean_sparse_per_cell_pair_distance_same_block_cpu(
    int* csc_i, int* csc_p, double* csc_x_double,
    int* /*b*/, int* /*b*/, double* /*b*/,
    double* result, int* num_rows, int* num_columns,
    int* /*num_columns_b*/, int* num_elements_a, int* /*num_elements_b*/)
{
  int n_cells = *num_columns;
  int nnz = *num_elements_a;
  float* csc_x = new float[nnz];
  for (int k = 0; k < nnz; ++k) csc_x[k] = static_cast<float>(csc_x_double[k]);

  #pragma omp parallel for schedule(dynamic)
  for (int ca = 0; ca < n_cells; ++ca) {
    for (int cb = ca + 1; cb < n_cells; ++cb) {
      float d = euclidean_per_cell_pair_merge(
          csc_i, csc_x, csc_p[ca], csc_p[ca + 1],
          csc_i, csc_x, csc_p[cb], csc_p[cb + 1]);
      result[ca * n_cells + cb] = d;
      result[cb * n_cells + ca] = d;
    }
    result[ca * n_cells + ca] = 0.0;
  }
  delete[] csc_x;
}


extern "C" void matrix_Euclidean_sparse_per_cell_pair_distance_different_blocks_cpu(
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

  #pragma omp parallel for schedule(dynamic)
  for (int ca = 0; ca < n_cells_a; ++ca) {
    for (int cb = 0; cb < n_cells_b; ++cb) {
      float d = euclidean_per_cell_pair_merge(
          a_csc_i, a_x, a_csc_p[ca], a_csc_p[ca + 1],
          b_csc_i, b_x, b_csc_p[cb], b_csc_p[cb + 1]);
      result[cb * n_cells_a + ca] = d;
    }
  }
  delete[] a_x;
  delete[] b_x;
}
