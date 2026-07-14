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



// ==================== True Pearson (centered cosine) ====================

extern "C" void matrix_Pearson_distance_same_block_cpu(
    double* a, double* b, double* c, int* n, int* m, int* m_b
) {
    pc_drive_cpu_cosine_same(a, c, *n, *m, /*center=*/true);
}


extern "C" void matrix_Pearson_distance_different_blocks_cpu(
    double* a, double* b, double* c, int* n, int* m, int* m_b
) {
    pc_drive_cpu_cosine_diff(a, b, c, *n, *m, *m_b, /*center=*/true);
}


// Pearson sparse same_block: sliding-window CSR with mean correction
// dot_centered = dot - col_sum1*col_sum2/rows
// norm_centered = sq - col_sum^2/rows
extern "C" void matrix_Pearson_sparse_distance_same_block_cpu(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
  pc_drive_cpu_sparse_cosine_same(a_index, a_positions, a_double_values, result,
                                  *num_rows, *num_columns, *num_elements_a,
                                  /*center=*/true);
}


extern "C" void matrix_Pearson_sparse_distance_different_blocks_cpu(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
  pc_drive_cpu_sparse_cosine_diff(a_index, a_positions, a_double_values,
                                  b_index, b_positions, b_double_values,
                                  result, *num_rows, *num_columns,
                                  *num_columns_b, *num_elements_a,
                                  *num_elements_b, /*center=*/true);
}


// ─── Pearson ───

static float pearson_per_cell_pair_merge(
    const int* a_i, const float* a_x, int ia, int ea, float sum_a, float sumsq_a,
    const int* b_i, const float* b_x, int ib, int eb, float sum_b, float sumsq_b,
    int n_genes)
{
  float dot = 0;
  while (ia < ea && ib < eb) {
    if (a_i[ia] < b_i[ib]) ++ia;
    else if (b_i[ib] < a_i[ia]) ++ib;
    else { dot += a_x[ia] * b_x[ib]; ++ia; ++ib; }
  }
  float n = (float)n_genes;
  float cov = dot - sum_a * sum_b / n;
  float va  = sumsq_a - sum_a * sum_a / n;
  float vb  = sumsq_b - sum_b * sum_b / n;
  return 1.0f - cov / sqrtf(va * vb);
}


static void compute_sum_sumsq(const float* x, int start, int end, float& s, float& sq) {
  s = 0; sq = 0;
  for (int k = start; k < end; ++k) { s += x[k]; sq += x[k] * x[k]; }
}


extern "C" void matrix_Pearson_sparse_per_cell_pair_distance_same_block_cpu(
    int* csc_i, int* csc_p, double* csc_x_double,
    int* /*b*/, int* /*b*/, double* /*b*/,
    double* result, int* num_rows, int* num_columns,
    int* /*num_columns_b*/, int* num_elements_a, int* /*num_elements_b*/)
{
  int n_genes = *num_rows;
  int n_cells = *num_columns;
  int nnz = *num_elements_a;
  float* csc_x = new float[nnz];
  for (int k = 0; k < nnz; ++k) csc_x[k] = static_cast<float>(csc_x_double[k]);

  float* sums = new float[n_cells];
  float* sumsqs = new float[n_cells];
  for (int c = 0; c < n_cells; ++c) compute_sum_sumsq(csc_x, csc_p[c], csc_p[c + 1], sums[c], sumsqs[c]);

  #pragma omp parallel for schedule(dynamic)
  for (int ca = 0; ca < n_cells; ++ca) {
    for (int cb = ca + 1; cb < n_cells; ++cb) {
      float d = pearson_per_cell_pair_merge(
          csc_i, csc_x, csc_p[ca], csc_p[ca + 1], sums[ca], sumsqs[ca],
          csc_i, csc_x, csc_p[cb], csc_p[cb + 1], sums[cb], sumsqs[cb],
          n_genes);
      result[ca * n_cells + cb] = d;
      result[cb * n_cells + ca] = d;
    }
    result[ca * n_cells + ca] = 0.0;
  }
  delete[] csc_x;
  delete[] sums;
  delete[] sumsqs;
}


extern "C" void matrix_Pearson_sparse_per_cell_pair_distance_different_blocks_cpu(
    int* a_csc_i, int* a_csc_p, double* a_xd,
    int* b_csc_i, int* b_csc_p, double* b_xd,
    double* result, int* num_rows, int* num_columns,
    int* num_columns_b, int* num_elements_a, int* num_elements_b)
{
  int n_genes = *num_rows;
  int n_cells_a = *num_columns;
  int n_cells_b = *num_columns_b;
  int nnz_a = *num_elements_a;
  int nnz_b = *num_elements_b;
  float* a_x = new float[nnz_a];
  float* b_x = new float[nnz_b];
  for (int k = 0; k < nnz_a; ++k) a_x[k] = static_cast<float>(a_xd[k]);
  for (int k = 0; k < nnz_b; ++k) b_x[k] = static_cast<float>(b_xd[k]);

  float* a_s = new float[n_cells_a]; float* a_q = new float[n_cells_a];
  float* b_s = new float[n_cells_b]; float* b_q = new float[n_cells_b];
  for (int c = 0; c < n_cells_a; ++c) compute_sum_sumsq(a_x, a_csc_p[c], a_csc_p[c + 1], a_s[c], a_q[c]);
  for (int c = 0; c < n_cells_b; ++c) compute_sum_sumsq(b_x, b_csc_p[c], b_csc_p[c + 1], b_s[c], b_q[c]);

  #pragma omp parallel for schedule(dynamic)
  for (int ca = 0; ca < n_cells_a; ++ca) {
    for (int cb = 0; cb < n_cells_b; ++cb) {
      float d = pearson_per_cell_pair_merge(
          a_csc_i, a_x, a_csc_p[ca], a_csc_p[ca + 1], a_s[ca], a_q[ca],
          b_csc_i, b_x, b_csc_p[cb], b_csc_p[cb + 1], b_s[cb], b_q[cb],
          n_genes);
      result[cb * n_cells_a + ca] = d;
    }
  }
  delete[] a_x;  delete[] b_x;
  delete[] a_s;  delete[] a_q;
  delete[] b_s;  delete[] b_q;
}
