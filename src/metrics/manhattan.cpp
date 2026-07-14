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



//Naive Implementation of Manhattan_distance_matrix (BruteForce)
extern "C" void matrix_Manhattan_distance_same_block_cpu(double * a, double * b, double * c, int * n, int * m, int * m_b) {
  int array_size = * n * * m;
  float * d_array = new float[array_size];
  for (int i = 0; i < array_size; ++i) {
    d_array[i] = a[i];
  }

  float * h_result = new float[( * m) * ( * m)];
  std::memset(h_result, 0, ( * m) * ( * m) * sizeof(float));

  #pragma omp parallel for schedule(dynamic)
  for (int col1_num = 0; col1_num < * m; ++col1_num) {
    for (int col2_num = col1_num + 1; col2_num < * m; ++col2_num) {
      float * col1 = d_array + * n * col1_num;
      float * col2 = d_array + * n * col2_num;
      float sum = 0.0f;
      for (int row = 0; row < * n; ++row) {
        sum += std::abs(col1[row] - col2[row]);
      }
      h_result[col1_num * * m + col2_num] = sum;
      h_result[col2_num * * m + col1_num] = sum;
    }
  }

  for (int i = 0; i < ( * m) * ( * m); ++i) {
    c[i] = h_result[i];
  }
  free(h_result);
  free(d_array);
}


extern "C" void matrix_Manhattan_distance_different_blocks_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b) {
  int array_size = * n * * m;
  float * array_new = new float[ * n * * m];
  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  int array_size2 = * n * (*m_b);
  float * array2_new = new float[array_size2];
  for (int i = 0; i < array_size2; ++i) {
    array2_new[i] = b[i];
  }

  float * d_array = new float[array_size];
  float *d_array2 = new float[array_size2];

  std::memcpy(d_array, array_new, array_size * sizeof(float));
  std::memcpy(d_array2, array2_new, array_size2 * sizeof(float));

  float * h_result = new float[( * m) * ( * m_b)];
  std::memset(h_result, 0, ( * m) * ( * m_b) * sizeof(float));

  //CPU Implementation
  parallel_for_cols(* m, [&](int col1_start, int col1_end) {
    for (int row = 0; row < * n; row++) {
      for (int col1_num = col1_start; col1_num < col1_end; ++col1_num) {
        for (int col2_num = 0; col2_num < * m_b; ++col2_num) {
          float * col1 = d_array + * n * col1_num;
          float * col2 = d_array2 + * n * col2_num;
          if (row < * n) {
            float diff = std::abs(col1[row] - col2[row]);
            h_result[col2_num * * m + col1_num] += diff;
          }
        }
      }
    }
  });

  for (int i = 0; i < ( * m) * ( * m_b); ++i) {
    c[i] = h_result[i];
  }
  free(h_result);
  free(d_array);
  free(d_array2);
}



extern "C" void  matrix_Manhattan_sparse_distance_same_block_cpu(
  int *a_index,
  int *a_positions,
  double *a_double_values,
  int *b_index,
  int *b_positions,
  double *b_double_values,
  double *result,
  int *num_rows,
  int *num_columns,
  int *num_columns_b,
  int *num_elements_a,
  int *num_elements_b
){
  int rows = *num_rows;
  int columns = *num_columns;

  float * a_values = new float[*num_elements_a];
  float * float_result = new float[columns * columns];
  std::memset(float_result, 0, columns * columns * sizeof(float));
  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = a_double_values[i];
  }

  int result_size = columns * columns;
  int nt = get_num_threads(rows);
  std::vector<float*> locals(nt);
  for (int t = 0; t < nt; ++t) {
    locals[t] = new float[result_size];
    std::memset(locals[t], 0, result_size * sizeof(float));
  }

  parallel_for_with_id(rows, nt, [&](int t, int row_start, int row_end) {
    float* local = locals[t];
    for (int row_index = row_start; row_index < row_end; ++row_index) {
      int start_column = a_positions[row_index];
      int end_column = a_positions[row_index + 1];

      for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
        for (int col2_index = col1_index; col2_index < end_column; ++col2_index) {
          int prev_col_index = col1_index - 1;
          int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;

          int next_col_index = col2_index + 1;
          int next_col = (next_col_index < end_column) ? a_index[next_col_index] : columns;

          int col1 = a_index[col1_index];
          int col2 = a_index[col2_index];

          float value1 = a_values[col1_index];
          float value2 = a_values[col2_index];

          for (int left = prev_col + 1; left < col1; ++left) {
            local[left * columns + col2] += std::abs(value2);
            local[col2 * columns + left] += std::abs(value2);
          }

          for (int right = col2 + 1; right < next_col; ++right) {
            local[right * columns + col1] += std::abs(value1);
            local[col1 * columns + right] += std::abs(value1);
          }

          local[col1 * columns + col2] += std::abs(value1 - value2);
          local[col2 * columns + col1] += std::abs(value1 - value2);
        }
      }
    }
  });

  for (int t = 0; t < nt; ++t) {
    for (int i = 0; i < result_size; ++i) float_result[i] += locals[t][i];
    delete[] locals[t];
  }

  for (int i = 0; i < result_size; ++i) {
    result[i] = float_result[i];
  }

  free(float_result);
  free(a_values);
}


extern "C" void  matrix_Manhattan_sparse_distance_different_blocks_cpu(
  int *a_index,
  int *a_positions,
  double *a_double_values,
  int *b_index,
  int *b_positions,
  double *b_double_values,
  double *result,
  int *num_rows,
  int *num_columns,
  int *num_columns_b,
  int *num_elements_a,
  int *num_elements_b
){
  int rows = *num_rows;
  int columns = *num_columns;
  int columns_b = *num_columns_b;

  float * a_values = new float[*num_elements_a];
  float * float_result = new float[columns * columns_b];
  std::memset(float_result, 0, columns * columns_b * sizeof(float));
  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = a_double_values[i];
  }

  float * b_values = new float[*num_elements_b];
  for (int i = 0; i < *num_elements_b; ++i) {
    b_values[i] = b_double_values[i];
  }

  int result_size = columns * columns_b;
  int nt = get_num_threads(rows);
  std::vector<float*> locals(nt);
  for (int t = 0; t < nt; ++t) {
    locals[t] = new float[result_size];
    std::memset(locals[t], 0, result_size * sizeof(float));
  }

  parallel_for_with_id(rows, nt, [&](int t, int row_start, int row_end) {
    float* local = locals[t];
    for (int row_index = row_start; row_index < row_end; ++row_index) {
      int start_column = a_positions[row_index];
      int end_column = a_positions[row_index + 1];

      int start_column_b = b_positions[row_index];
      int end_column_b = b_positions[row_index + 1];

      for (int col1_index = start_column; col1_index <= end_column; ++col1_index) {
        int prev_col_index = col1_index - 1;
        int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
        float value1 = (col1_index < end_column) ? a_values[col1_index] : 0.0f;

        int col1 = (col1_index < end_column) ? a_index[col1_index] : columns;

        for (int col2_index = start_column_b; col2_index <= end_column_b; ++col2_index) {
          int prev_col_b_index = col2_index - 1;
          int prev_col2 = (prev_col_b_index >= start_column_b) ? b_index[prev_col_b_index] : -1;

          int col2 = (col2_index < end_column_b) ? b_index[col2_index] : columns_b;
          float value2 = (col2_index < end_column_b) ? b_values[col2_index] : 0.0f;

          if (col2 < columns_b) {
            for (int left = prev_col + 1; left < col1; ++left) {
              local[col2 * columns + left] += std::abs(value2);
            }
          }
          if (col1 < columns) {
            for (int left = prev_col2 + 1; left < col2; ++left) {
              local[left * columns + col1] += std::abs(value1);
            }
          }

          if (col1 < columns && col2 < columns_b) {
            local[col2 * columns + col1] += std::abs(value1 - value2);
          }
        }
      }
    }
  });

  for (int t = 0; t < nt; ++t) {
    for (int i = 0; i < result_size; ++i) float_result[i] += locals[t][i];
    delete[] locals[t];
  }

  for (int i = 0; i < result_size; ++i) {
    result[i] = float_result[i];
  }

  free(float_result);
  free(a_values);
  free(b_values);
}


// ─── Manhattan ───

static float manhattan_per_cell_pair_merge(
    const int* a_i, const float* a_x, int ia, int ea,
    const int* b_i, const float* b_x, int ib, int eb)
{
  float acc = 0;
  while (ia < ea || ib < eb) {
    if (ia < ea && (ib >= eb || a_i[ia] < b_i[ib])) {
      acc += fabsf(a_x[ia]); ++ia;
    } else if (ib < eb && (ia >= ea || b_i[ib] < a_i[ia])) {
      acc += fabsf(b_x[ib]); ++ib;
    } else {
      acc += fabsf(a_x[ia] - b_x[ib]); ++ia; ++ib;
    }
  }
  return acc;
}


extern "C" void matrix_Manhattan_sparse_per_cell_pair_distance_same_block_cpu(
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
      float d = manhattan_per_cell_pair_merge(
          csc_i, csc_x, csc_p[ca], csc_p[ca + 1],
          csc_i, csc_x, csc_p[cb], csc_p[cb + 1]);
      result[ca * n_cells + cb] = d;
      result[cb * n_cells + ca] = d;
    }
    result[ca * n_cells + ca] = 0.0;
  }
  delete[] csc_x;
}


extern "C" void matrix_Manhattan_sparse_per_cell_pair_distance_different_blocks_cpu(
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
      float d = manhattan_per_cell_pair_merge(
          a_csc_i, a_x, a_csc_p[ca], a_csc_p[ca + 1],
          b_csc_i, b_x, b_csc_p[cb], b_csc_p[cb + 1]);
      result[cb * n_cells_a + ca] = d;
    }
  }
  delete[] a_x;
  delete[] b_x;
}
