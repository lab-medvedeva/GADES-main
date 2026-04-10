//=============================
#include <iostream>
#include <fstream>
#include <R.h>
#include <stdio.h>
#include <cstring>
#include <math.h>
#include <cassert>
#include <algorithm>
#include <vector>
#include <omp.h>
//=========================================

template<typename F>
static void parallel_for_with_id(int num_items, int num_threads, F&& worker) {
    if (num_threads > num_items) num_threads = num_items;
    #pragma omp parallel num_threads(num_threads)
    {
        int t = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int chunk = num_items / nt;
        int remainder = num_items % nt;
        int start = 0;
        for (int i = 0; i < t; ++i) start += chunk + (i < remainder ? 1 : 0);
        int end = start + chunk + (t < remainder ? 1 : 0);
        worker(t, start, end);
    }
}

static int get_num_threads(int num_items) {
    int nt = omp_get_max_threads();
    if (nt <= 0) nt = 1;
    if (nt > num_items) nt = num_items;
    return nt;
}

template<typename F>
static void parallel_for_cols(int num_cols, F&& worker) {
    int num_threads = omp_get_max_threads();
    if (num_threads <= 0) num_threads = 1;
    if (num_threads > num_cols) num_threads = num_cols;

    #pragma omp parallel num_threads(num_threads)
    {
        int t = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int chunk = num_cols / nt;
        int remainder = num_cols % nt;
        int start = 0;
        for (int i = 0; i < t; ++i) start += chunk + (i < remainder ? 1 : 0);
        int end = start + chunk + (t < remainder ? 1 : 0);
        worker(start, end);
    }
}

//Naive Implementation of Euclidean_distance_matrix (BruteForce)
extern "C" void matrix_Euclidean_distance_same_block_cpu(double * a, double * b, double * c, int * n, int * m, int * m_b) {
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
        float diff = col1[row] - col2[row];
        sum += diff * diff;
      }
      h_result[col1_num * * m + col2_num] = sum;
      h_result[col2_num * * m + col1_num] = sum;
    }
  }

  for (int i = 0; i < ( * m) * ( * m); ++i) {
    c[i] = std::sqrt(h_result[i]);
  }
  free(h_result);
  free(d_array);
}
//=============================================
//Naive Implementation of Kendall_distance_matrix for same block
extern "C" void matrix_Kendall_distance_same_block_cpu(double * a, double * b, double * c, int * n, int * m, int * m_b) {
  int array_size = * n * * m;
  float * d_array = new float[array_size];
  for (int i = 0; i < array_size; ++i) {
    d_array[i] = a[i];
  }

  unsigned int * h_result = new unsigned int[( * m) * ( * m)];
  std::memset(h_result, 0, ( * m) * ( * m) * sizeof(unsigned int));

  #pragma omp parallel for schedule(dynamic)
  for (int col1_num = 0; col1_num < * m; ++col1_num) {
    for (int col2_num = col1_num + 1; col2_num < * m; ++col2_num) {
      float * col1 = d_array + * n * col1_num;
      float * col2 = d_array + * n * col2_num;
      unsigned int disc = 0;
      for (int row1 = 0; row1 < * n; ++row1) {
        for (int row2 = row1 + 1; row2 < * n; ++row2) {
          if ((col1[row1] - col1[row2]) * (col2[row1] - col2[row2]) < 0) {
            disc++;
          }
        }
      }
      h_result[col1_num * * m + col2_num] = disc;
      h_result[col2_num * * m + col1_num] = disc;
    }
  }

  for (int i = 0; i < ( * m) * ( * m); ++i) {
    c[i] = h_result[i] * 2.0f / (*n) / (*n - 1);
  }
  free(h_result);
  free(d_array);
}

extern "C" void  matrix_Cosine_distance_same_block_cpu(
  double* a,
  double* b,
  double* c,
  int* n,
  int* m,
  int* m_b
) {
  int array_size = * n * * m;
  float * array_new = new float[array_size];
  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  float * h_scalar = new float[( * m) * ( * m)];
  std::memset(h_scalar, 0, ( * m) * ( * m) * sizeof(float));
  float * h_prod1 = new float[( * m) * ( * m)];
  std::memset(h_prod1, 0, ( * m) * ( * m) * sizeof(float));
  float * h_prod2 = new float[( * m) * ( * m)];
  std::memset(h_prod2, 0, ( * m) * ( * m) * sizeof(float));

  #pragma omp parallel for schedule(dynamic)
  for (int col1_num = 0; col1_num < * m; ++col1_num) {
    for (int col2_num = col1_num; col2_num < * m; ++col2_num) {
      float * col1 = array_new + * n * col1_num;
      float * col2 = array_new + * n * col2_num;
      float scalar = 0.0f, p1 = 0.0f, p2 = 0.0f;
      for (int row = 0; row < * n; ++row) {
        scalar += col1[row] * col2[row];
        p1 += col1[row] * col1[row];
        p2 += col2[row] * col2[row];
      }
      h_scalar[col1_num * * m + col2_num] = scalar;
      h_prod1[col1_num * * m + col2_num] = p1;
      h_prod2[col1_num * * m + col2_num] = p2;
      h_scalar[col2_num * * m + col1_num] = scalar;
      h_prod1[col2_num * * m + col1_num] = p1;
      h_prod2[col2_num * * m + col1_num] = p2;
    }
  }

  for (int i = 0; i < (*m) * (*m); ++i) {
    c[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);
  }

  free(h_prod1);
  free(h_prod2);
  free(h_scalar);
  free(array_new);
}

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
//======================================================

extern "C" void matrix_Euclidean_distance_different_blocks_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b) {
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
          float diff = col1[row] - col2[row];
          diff = diff * diff;
          h_result[col2_num * * m + col1_num] += diff;
        }
      }
    }
  });

  for (int i = 0; i < ( * m) * ( * m_b); ++i) {
    c[i] = std::sqrt(h_result[i]);
  }
  free(h_result);
  free(d_array);
  free(d_array2);
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

extern "C" void  matrix_Kendall_distance_different_blocks_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b){
  int array_size = * n * * m;
  float * d_array = new float[ * n * * m];
  for (int i = 0; i < array_size; ++i) {
    d_array[i] = a[i];
  }

  int array_size2 = * n * (*m_b);
  float * d_array2 = new float[array_size2];
  for (int i = 0; i < array_size2; ++i) {
    d_array2[i] = b[i];
  }

  unsigned int * h_result = new unsigned int[( * m) * ( * m_b)];
  std::memset(h_result, 0, ( * m) * ( * m_b) * sizeof(unsigned int));

  //CPU Implementation
  parallel_for_cols(* m, [&](int col1_start, int col1_end) {
    for (int row1 = 0; row1 < * n; row1++) {
      for (int row2=0;row2<*n;row2++){
        for (int col1_num = col1_start; col1_num < col1_end; ++col1_num) {
          for (int col2_num = 0; col2_num < * m_b; ++col2_num) {
            float * col1 = d_array + * n * col1_num;
            float * col2 = d_array2 + * n * col2_num;
            if (row1 < row2 && row2 < *n){
              if ((col1[row1] - col1[row2]) * (col2[row1] - col2[row2]) < 0){
                h_result[col2_num * * m + col1_num] += 1;
              }
            }
          }
        }
      }
    }
  });


  for (int i = 0; i < ( * m) * ( * m_b); ++i) {
    c[i] = h_result[i] * 2.0f / (*n) / (*n - 1);
  }
  free(h_result);
  free(d_array);
  free(d_array2);
}

extern "C" void  matrix_Cosine_distance_different_blocks_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b){
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

  float * h_scalar = new float[( * m) * ( * m_b)];
  std::memset(h_scalar, 0, ( * m) * ( * m_b) * sizeof(float));
 
  float * h_prod1 = new float[( * m) * ( * m_b)];
  std::memset(h_prod1, 0, ( * m) * ( * m_b) * sizeof(float));
 
  float * h_prod2 = new float[( * m) * ( * m_b)];
  std::memset(h_prod2, 0, ( * m) * ( * m_b) * sizeof(float));
 

  parallel_for_cols(* m, [&](int col1_start, int col1_end) {
    for (int row = 0; row < *n; row++) {
      for (int col1_num = col1_start; col1_num < col1_end; ++col1_num) {
        for (int col2_num = 0; col2_num < * m_b; ++col2_num) {
          float * col1 = d_array + * n * col1_num;
          float * col2 = d_array2 + * n * col2_num;
          float num = col1[row] * col2[row];
          float sum1 = col1[row] * col1[row];
          float sum2 = col2[row] * col2[row];
          h_scalar[col2_num * * m + col1_num] += num;
          h_prod1[col2_num * * m + col1_num] += sum1;
          h_prod2[col2_num * * m + col1_num] += sum2;
        }
      }
    }
  });
    for (int i = 0; i < (*m) * (*m_b); ++i) {
    c[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);

  }
  free(h_prod1);
  free(h_prod2);
  free(h_scalar);
  free(d_array);
  free(d_array2);
}


extern "C" void  matrix_Euclidean_sparse_distance_same_block_cpu(
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
            local[left * columns + col2] += value2 * value2;
            local[col2 * columns + left] += value2 * value2;
          }

          for (int right = col2 + 1; right < next_col; ++right) {
            local[right * columns + col1] += value1 * value1;
            local[col1 * columns + right] += value1 * value1;
          }

          local[col1 * columns + col2] += (value1 - value2) * (value1 - value2);
          local[col2 * columns + col1] += (value1 - value2) * (value1 - value2);
        }
      }
    }
  });

  for (int t = 0; t < nt; ++t) {
    for (int i = 0; i < result_size; ++i) float_result[i] += locals[t][i];
    delete[] locals[t];
  }

  for (int i = 0; i < result_size; ++i) {
    result[i] = std::sqrt(float_result[i]);
  }

  free(float_result);
  free(a_values);
}

extern "C" void  matrix_Euclidean_sparse_distance_different_blocks_cpu(
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
              local[col2 * columns + left] += value2 * value2;
            }
          }
          if (col1 < columns) {
            for (int left = prev_col2 + 1; left < col2; ++left) {
              local[left * columns + col1] += value1 * value1;
            }
          }

          if (col1 < columns && col2 < columns_b) {
            local[col2 * columns + col1] += (value1 - value2) * (value1 - value2);
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
    result[i] = std::sqrt(float_result[i]);
  }

  free(float_result);
  free(a_values);
  free(b_values);
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


extern "C" void matrix_Cosine_sparse_distance_same_block_cpu(
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

  float * squares = new float[columns];

  int result_size = columns * columns;
  int nt = get_num_threads(rows);
  std::vector<float*> local_results(nt);
  std::vector<float*> local_squares(nt);
  for (int t = 0; t < nt; ++t) {
    local_results[t] = new float[result_size];
    std::memset(local_results[t], 0, result_size * sizeof(float));
    local_squares[t] = new float[columns];
    std::memset(local_squares[t], 0, columns * sizeof(float));
  }

  parallel_for_with_id(rows, nt, [&](int t, int row_start, int row_end) {
    float* local = local_results[t];
    float* lsq = local_squares[t];
    for (int row_index = row_start; row_index < row_end; ++row_index) {
      int start_column = a_positions[row_index];
      int end_column = a_positions[row_index + 1];

      for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
        int col1 = a_index[col1_index];
        float value1 = a_values[col1_index];
        lsq[col1] += value1 * value1;

        for (int col2_index = col1_index + 1; col2_index < end_column; ++col2_index) {
          int col2 = a_index[col2_index];
          float value2 = a_values[col2_index];

          local[col1 * columns + col2] += value1 * value2;
          local[col2 * columns + col1] += value1 * value2;
        }
      }
    }
  });

  std::memset(float_result, 0, result_size * sizeof(float));
  std::memset(squares, 0, columns * sizeof(float));
  for (int t = 0; t < nt; ++t) {
    for (int i = 0; i < result_size; ++i) float_result[i] += local_results[t][i];
    for (int i = 0; i < columns; ++i) squares[i] += local_squares[t][i];
    delete[] local_results[t];
    delete[] local_squares[t];
  }

  for (int i = 0; i < result_size; ++i) {
    int row_index = i / columns;
    int column_index = i % columns;
    if (row_index != column_index) {
        result[i] = 1.0f - float_result[i] / std::sqrt(squares[row_index]) / std::sqrt(squares[column_index]);
    } else {
        result[i] = 0.0f;
    }
  }

  free(float_result);
  free(squares);
  free(a_values);
}

extern "C" void  matrix_Cosine_sparse_distance_different_blocks_cpu(
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

  for (int i = 0; i < columns * columns_b; ++i) {
    float_result[i] = 0.0f;
  }

  float * b_values = new float[*num_elements_b];
  for (int i = 0; i < *num_elements_b; ++i) {
    b_values[i] = b_double_values[i];
  }

  float * squares_a = new float[columns];
  float * squares_b = new float[columns_b];

  int result_size = columns * columns_b;
  int nt = get_num_threads(rows);
  std::vector<float*> loc_res(nt);
  std::vector<float*> loc_sqa(nt);
  std::vector<float*> loc_sqb(nt);
  for (int t = 0; t < nt; ++t) {
    loc_res[t] = new float[result_size];
    std::memset(loc_res[t], 0, result_size * sizeof(float));
    loc_sqa[t] = new float[columns];
    std::memset(loc_sqa[t], 0, columns * sizeof(float));
    loc_sqb[t] = new float[columns_b];
    std::memset(loc_sqb[t], 0, columns_b * sizeof(float));
  }

  parallel_for_with_id(rows, nt, [&](int t, int row_start, int row_end) {
    float* local = loc_res[t];
    float* lsqa = loc_sqa[t];
    float* lsqb = loc_sqb[t];
    for (int row_index = row_start; row_index < row_end; ++row_index) {
      int start_column = a_positions[row_index];
      int end_column = a_positions[row_index + 1];

      int start_column_b = b_positions[row_index];
      int end_column_b = b_positions[row_index + 1];

      for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
        float value1 = a_values[col1_index];
        int col1 = a_index[col1_index];

        for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
          int col2 = b_index[col2_index];
          float value2 = b_values[col2_index];
          local[col2 * columns + col1] += value1 * value2;
        }
      }

      for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
        float value1 = a_values[col1_index];
        int col1 = a_index[col1_index];
        lsqa[col1] += value1 * value1;
      }
      for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
        float value2 = b_values[col2_index];
        int col2 = b_index[col2_index];
        lsqb[col2] += value2 * value2;
      }
    }
  });

  std::memset(float_result, 0, result_size * sizeof(float));
  std::memset(squares_a, 0, columns * sizeof(float));
  std::memset(squares_b, 0, columns_b * sizeof(float));
  for (int t = 0; t < nt; ++t) {
    for (int i = 0; i < result_size; ++i) float_result[i] += loc_res[t][i];
    for (int i = 0; i < columns; ++i) squares_a[i] += loc_sqa[t][i];
    for (int i = 0; i < columns_b; ++i) squares_b[i] += loc_sqb[t][i];
    delete[] loc_res[t];
    delete[] loc_sqa[t];
    delete[] loc_sqb[t];
  }

  for (int i = 0; i < result_size; ++i) {
    int row_index = i / columns;
    int column_index = i % columns;
    result[i] = 1.0f - float_result[i] / std::sqrt(squares_b[row_index]) / std::sqrt(squares_a[column_index]);
  }

  free(float_result);
  free(a_values);
  free(b_values);
  free(squares_a);
  free(squares_b);
}

extern "C" void matrix_Kendall_sparse_distance_same_block_cpu(
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
) {
  int rows = *num_rows;
  int columns = *num_columns;

  float *a_values = new float[*num_elements_a];
  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = static_cast<float>(a_double_values[i]);
  }

  int result_size = columns * columns;
  int nt = get_num_threads(rows);
  std::vector<int*> locals(nt);
  for (int t = 0; t < nt; ++t) {
    locals[t] = new int[result_size];
    std::memset(locals[t], 0, result_size * sizeof(int));
  }

  // Two-pointer merge over CSR rows: for each row pair, merge nonzero
  // columns from both rows, compute diffs, count discordant column pairs.
  parallel_for_with_id(rows, nt, [&](int t, int row_start, int row_end) {
    int* local = locals[t];
    std::vector<int> pos_cols, neg_cols;

    for (int r1 = row_start; r1 < row_end; ++r1) {
      int r1_start = a_positions[r1];
      int r1_end   = a_positions[r1 + 1];

      for (int r2 = r1 + 1; r2 < rows; ++r2) {
        int r2_start = a_positions[r2];
        int r2_end   = a_positions[r2 + 1];

        pos_cols.clear();
        neg_cols.clear();

        // Two-pointer merge of R1 and R2 nonzero columns
        int i = r1_start, j = r2_start;
        while (i < r1_end && j < r2_end) {
          int c1 = a_index[i], c2 = a_index[j];
          if (c1 == c2) {
            float diff = a_values[i] - a_values[j];
            if (diff > 0) pos_cols.push_back(c1);
            else if (diff < 0) neg_cols.push_back(c1);
            ++i; ++j;
          } else if (c1 < c2) {
            // R1 nonzero, R2 zero => diff = a_values[i]
            if (a_values[i] > 0) pos_cols.push_back(c1);
            else if (a_values[i] < 0) neg_cols.push_back(c1);
            ++i;
          } else {
            // R1 zero, R2 nonzero => diff = -a_values[j]
            if (a_values[j] < 0) pos_cols.push_back(c2);
            else if (a_values[j] > 0) neg_cols.push_back(c2);
            ++j;
          }
        }
        while (i < r1_end) {
          if (a_values[i] > 0) pos_cols.push_back(a_index[i]);
          else if (a_values[i] < 0) neg_cols.push_back(a_index[i]);
          ++i;
        }
        while (j < r2_end) {
          if (a_values[j] < 0) pos_cols.push_back(a_index[j]);
          else if (a_values[j] > 0) neg_cols.push_back(a_index[j]);
          ++j;
        }

        // Each (pos, neg) pair is discordant
        for (int p : pos_cols) {
          for (int n : neg_cols) {
            local[p * columns + n] += 1;
            local[n * columns + p] += 1;
          }
        }
      }
    }
  });

  int *disconcordant = new int[result_size];
  std::memset(disconcordant, 0, result_size * sizeof(int));
  for (int t = 0; t < nt; ++t) {
    for (int i = 0; i < result_size; ++i) disconcordant[i] += locals[t][i];
    delete[] locals[t];
  }

  for (int i = 0; i < result_size; ++i) {
    result[i] = static_cast<double>(disconcordant[i]) * 2.0f / rows / (rows - 1);
  }

  delete[] disconcordant;
  delete[] a_values;
}


extern "C" void matrix_Kendall_sparse_distance_different_blocks_cpu(
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
) {
  int rows = *num_rows;
  int columns = *num_columns;
  int columns_b = *num_columns_b;
  int result_size = columns * columns_b;

  float *a_values = new float[*num_elements_a];
  float *b_values = new float[*num_elements_b];

  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = static_cast<float>(a_double_values[i]);
  }

  for (int i = 0; i < *num_elements_b; ++i) {
    b_values[i] = static_cast<float>(b_double_values[i]);
  }

  int nt = get_num_threads(rows);
  std::vector<int*> locals(nt);
  for (int t = 0; t < nt; ++t) {
    locals[t] = new int[result_size];
    std::memset(locals[t], 0, result_size * sizeof(int));
  }

  // Two-pointer merge: for each row pair, compute diffs for block A columns
  // and block B columns separately, then count cross-discordant pairs.
  parallel_for_with_id(rows, nt, [&](int tid, int row_start, int row_end) {
    int* local = locals[tid];
    // Diffs for block A columns (indexed 0..columns-1)
    std::vector<int> a_pos_cols, a_neg_cols;
    // Diffs for block B columns (indexed 0..columns_b-1)
    std::vector<int> b_pos_cols, b_neg_cols;

    for (int r1 = row_start; r1 < row_end; ++r1) {
      int r1a_start = a_positions[r1], r1a_end = a_positions[r1 + 1];
      int r1b_start = b_positions[r1], r1b_end = b_positions[r1 + 1];

      for (int r2 = r1 + 1; r2 < rows; ++r2) {
        int r2a_start = a_positions[r2], r2a_end = a_positions[r2 + 1];
        int r2b_start = b_positions[r2], r2b_end = b_positions[r2 + 1];

        // Merge block A columns for rows r1 and r2
        a_pos_cols.clear(); a_neg_cols.clear();
        {
          int i = r1a_start, j = r2a_start;
          while (i < r1a_end && j < r2a_end) {
            int c1 = a_index[i], c2 = a_index[j];
            if (c1 == c2) {
              float diff = a_values[i] - a_values[j];
              if (diff > 0) a_pos_cols.push_back(c1);
              else if (diff < 0) a_neg_cols.push_back(c1);
              ++i; ++j;
            } else if (c1 < c2) {
              if (a_values[i] > 0) a_pos_cols.push_back(c1);
              else if (a_values[i] < 0) a_neg_cols.push_back(c1);
              ++i;
            } else {
              if (a_values[j] < 0) a_pos_cols.push_back(c2);
              else if (a_values[j] > 0) a_neg_cols.push_back(c2);
              ++j;
            }
          }
          while (i < r1a_end) {
            if (a_values[i] > 0) a_pos_cols.push_back(a_index[i]);
            else if (a_values[i] < 0) a_neg_cols.push_back(a_index[i]);
            ++i;
          }
          while (j < r2a_end) {
            if (a_values[j] < 0) a_pos_cols.push_back(a_index[j]);
            else if (a_values[j] > 0) a_neg_cols.push_back(a_index[j]);
            ++j;
          }
        }

        // Merge block B columns for rows r1 and r2
        b_pos_cols.clear(); b_neg_cols.clear();
        {
          int i = r1b_start, j = r2b_start;
          while (i < r1b_end && j < r2b_end) {
            int c1 = b_index[i], c2 = b_index[j];
            if (c1 == c2) {
              float diff = b_values[i] - b_values[j];
              if (diff > 0) b_pos_cols.push_back(c1);
              else if (diff < 0) b_neg_cols.push_back(c1);
              ++i; ++j;
            } else if (c1 < c2) {
              if (b_values[i] > 0) b_pos_cols.push_back(c1);
              else if (b_values[i] < 0) b_neg_cols.push_back(c1);
              ++i;
            } else {
              if (b_values[j] < 0) b_pos_cols.push_back(c2);
              else if (b_values[j] > 0) b_neg_cols.push_back(c2);
              ++j;
            }
          }
          while (i < r1b_end) {
            if (b_values[i] > 0) b_pos_cols.push_back(b_index[i]);
            else if (b_values[i] < 0) b_neg_cols.push_back(b_index[i]);
            ++i;
          }
          while (j < r2b_end) {
            if (b_values[j] < 0) b_pos_cols.push_back(b_index[j]);
            else if (b_values[j] > 0) b_neg_cols.push_back(b_index[j]);
            ++j;
          }
        }

        // Cross-discordant: A_pos × B_neg and A_neg × B_pos
        // result layout: result[col_b * columns + col_a]
        for (int ac : a_pos_cols) {
          for (int bc : b_neg_cols) {
            local[bc * columns + ac] += 1;
          }
        }
        for (int ac : a_neg_cols) {
          for (int bc : b_pos_cols) {
            local[bc * columns + ac] += 1;
          }
        }
      }
    }
  });

  int *disconcordant = new int[result_size];
  std::memset(disconcordant, 0, result_size * sizeof(int));
  for (int t = 0; t < nt; ++t) {
    for (int i = 0; i < result_size; ++i) disconcordant[i] += locals[t][i];
    delete[] locals[t];
  }

  for (int i = 0; i < result_size; ++i) {
    result[i] = static_cast<double>(disconcordant[i]) * 2.0f / rows / (rows - 1);
  }

  delete[] disconcordant;
  delete[] a_values;
  delete[] b_values;
}

// ==================== Pearson helpers ====================

static float* csr_to_dense_cpu(int* index, int* positions, double* values, int rows, int columns) {
    float* dense = new float[rows * columns];
    std::memset(dense, 0, rows * columns * sizeof(float));
    for (int row = 0; row < rows; ++row) {
        int start = positions[row];
        int end = positions[row + 1];
        for (int j = start; j < end; ++j) {
            int col = index[j];
            dense[col * rows + row] = static_cast<float>(values[j]);
        }
    }
    return dense;
}

static void center_columns_cpu(float* array, int n, int m) {
    for (int col = 0; col < m; ++col) {
        float* col_data = array + col * n;
        float sum = 0.0f;
        for (int row = 0; row < n; ++row) {
            sum += col_data[row];
        }
        float mean = sum / n;
        for (int row = 0; row < n; ++row) {
            col_data[row] -= mean;
        }
    }
}

// ==================== Spearman ====================

// Rank columns and center (subtract mean rank = (n+1)/2) so that
// the existing Cosine logic (which computes cosine-style correlation)
// produces the correct Pearson-on-ranks = Spearman result.
static void rank_columns_cpu(float* array, int n, int m) {
    std::vector<std::pair<float, int>> col_data(n);
    float mean_rank = (n + 1) / 2.0f;
    for (int j = 0; j < m; ++j) {
        float* col = array + j * n;
        for (int i = 0; i < n; ++i) col_data[i] = {col[i], i};
        std::sort(col_data.begin(), col_data.end());
        int i = 0;
        while (i < n) {
            int end = i + 1;
            while (end < n && col_data[end].first == col_data[i].first) ++end;
            float rank = (i + 1 + end) / 2.0f - mean_rank;
            for (int k = i; k < end; ++k) col[col_data[k].second] = rank;
            i = end;
        }
    }
}

// Sweep-line ranking directly from CSR: sorts only non-zeros per column,
// inserts the implicit-zero group at the right position, assigns centered
// average-tie ranks (rank - mean_rank) so Pearson logic gives Spearman.
// Returns a dense column-major ranked array (rows x columns).
static float* csr_to_ranked_dense_cpu(int* index, int* positions, double* values, int rows, int columns) {
    float* ranked = new float[rows * columns];
    float mean_rank = (rows + 1) / 2.0f;

    // 1. Gather per-column non-zero entries (value, row_index)
    std::vector<std::vector<std::pair<float, int>>> col_entries(columns);
    for (int row = 0; row < rows; ++row) {
        for (int idx = positions[row]; idx < positions[row + 1]; ++idx) {
            col_entries[index[idx]].push_back({static_cast<float>(values[idx]), row});
        }
    }

    for (int j = 0; j < columns; ++j) {
        float* col_out = ranked + j * rows;
        auto& entries = col_entries[j];
        int nnz = static_cast<int>(entries.size());

        // 2. Sort non-zeros by value
        std::sort(entries.begin(), entries.end());

        // 3. Count negatives and explicit zeros among stored entries
        int neg_count = 0;
        int explicit_zero_count = 0;
        for (int i = 0; i < nnz; ++i) {
            if (entries[i].first < 0.0f) neg_count++;
            else if (entries[i].first == 0.0f) explicit_zero_count++;
        }
        int total_zeros = (rows - nnz) + explicit_zero_count;

        // 4. Zero-group rank (covers implicit + explicit zeros), centered
        float zero_rank = 0.0f;
        if (total_zeros > 0) {
            zero_rank = (neg_count + 1 + neg_count + total_zeros) / 2.0f - mean_rank;
        }
        // Pre-fill all rows with zero_rank (implicit zeros get it directly)
        for (int r = 0; r < rows; ++r) col_out[r] = zero_rank;

        // 5. Rank negative entries (sorted positions 0..neg_count-1)
        int global_pos = 0;
        int i = 0;
        while (i < neg_count) {
            int end = i + 1;
            while (end < neg_count && entries[end].first == entries[i].first) ++end;
            float rank = (global_pos + 1 + global_pos + (end - i)) / 2.0f - mean_rank;
            for (int k = i; k < end; ++k) col_out[entries[k].second] = rank;
            global_pos += (end - i);
            i = end;
        }

        // Explicit zeros already got zero_rank from pre-fill

        // 6. Rank positive entries (sorted positions neg_count+explicit_zero_count .. nnz-1)
        global_pos = neg_count + total_zeros;
        i = neg_count + explicit_zero_count;
        while (i < nnz) {
            int end = i + 1;
            while (end < nnz && entries[end].first == entries[i].first) ++end;
            float rank = (global_pos + 1 + global_pos + (end - i)) / 2.0f - mean_rank;
            for (int k = i; k < end; ++k) col_out[entries[k].second] = rank;
            global_pos += (end - i);
            i = end;
        }
    }
    return ranked;
}

extern "C" void matrix_Spearman_distance_same_block_cpu(
    double* a, double* b, double* c, int* n, int* m, int* m_b
) {
    int array_size = *n * *m;
    float* array_new = new float[array_size];
    for (int i = 0; i < array_size; ++i) {
        array_new[i] = static_cast<float>(a[i]);
    }
    rank_columns_cpu(array_new, *n, *m);

    float* h_scalar = new float[(*m) * (*m)];
    std::memset(h_scalar, 0, (*m) * (*m) * sizeof(float));
    float* h_prod1 = new float[(*m) * (*m)];
    std::memset(h_prod1, 0, (*m) * (*m) * sizeof(float));
    float* h_prod2 = new float[(*m) * (*m)];
    std::memset(h_prod2, 0, (*m) * (*m) * sizeof(float));

    #pragma omp parallel for schedule(dynamic)
    for (int col1_num = 0; col1_num < *m; ++col1_num) {
        for (int col2_num = col1_num; col2_num < *m; ++col2_num) {
            float* col1 = array_new + *n * col1_num;
            float* col2 = array_new + *n * col2_num;
            float scalar = 0.0f, p1 = 0.0f, p2 = 0.0f;
            for (int row = 0; row < *n; ++row) {
                scalar += col1[row] * col2[row];
                p1 += col1[row] * col1[row];
                p2 += col2[row] * col2[row];
            }
            h_scalar[col1_num * *m + col2_num] = scalar;
            h_prod1[col1_num * *m + col2_num] = p1;
            h_prod2[col1_num * *m + col2_num] = p2;
            h_scalar[col2_num * *m + col1_num] = scalar;
            h_prod1[col2_num * *m + col1_num] = p1;
            h_prod2[col2_num * *m + col1_num] = p2;
        }
    }

    for (int i = 0; i < (*m) * (*m); ++i) {
        c[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);
    }

    free(h_prod1);
    free(h_prod2);
    free(h_scalar);
    free(array_new);
}

extern "C" void matrix_Spearman_distance_different_blocks_cpu(
    double* a, double* b, double* c, int* n, int* m, int* m_b
) {
    int array_size = *n * *m;
    float* array_new = new float[array_size];
    for (int i = 0; i < array_size; ++i) {
        array_new[i] = static_cast<float>(a[i]);
    }
    rank_columns_cpu(array_new, *n, *m);

    int array2_size = *n * (*m_b);
    float* array2_new = new float[array2_size];
    for (int i = 0; i < array2_size; ++i) {
        array2_new[i] = static_cast<float>(b[i]);
    }
    rank_columns_cpu(array2_new, *n, *m_b);

    float* d_array = new float[array_size];
    float* d_array2 = new float[array2_size];
    std::memcpy(d_array, array_new, array_size * sizeof(float));
    std::memcpy(d_array2, array2_new, array2_size * sizeof(float));

    float* h_scalar = new float[(*m) * (*m_b)];
    std::memset(h_scalar, 0, (*m) * (*m_b) * sizeof(float));
    float* h_prod1 = new float[(*m) * (*m_b)];
    std::memset(h_prod1, 0, (*m) * (*m_b) * sizeof(float));
    float* h_prod2 = new float[(*m) * (*m_b)];
    std::memset(h_prod2, 0, (*m) * (*m_b) * sizeof(float));

    parallel_for_cols(*m, [&](int col1_start, int col1_end) {
        for (int row = 0; row < *n; row++) {
            for (int col1_num = col1_start; col1_num < col1_end; ++col1_num) {
                float* col1 = d_array + *n * col1_num;
                for (int col2_num = 0; col2_num < *m_b; ++col2_num) {
                    float* col2 = d_array2 + *n * col2_num;
                    float num = col1[row] * col2[row];
                    float sum1 = col1[row] * col1[row];
                    float sum2 = col2[row] * col2[row];
                    h_scalar[col2_num * *m + col1_num] += num;
                    h_prod1[col2_num * *m + col1_num] += sum1;
                    h_prod2[col2_num * *m + col1_num] += sum2;
                }
            }
        }
    });

    for (int i = 0; i < (*m) * (*m_b); ++i) {
        c[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);
    }

    delete[] h_prod1;
    delete[] h_prod2;
    delete[] h_scalar;
    delete[] d_array;
    delete[] d_array2;
    delete[] array_new;
    delete[] array2_new;
}

extern "C" void matrix_Spearman_sparse_distance_same_block_cpu(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
    int rows = *num_rows;
    int columns = *num_columns;

    float* dense = csr_to_ranked_dense_cpu(a_index, a_positions, a_double_values, rows, columns);

    float* h_scalar = new float[columns * columns];
    std::memset(h_scalar, 0, columns * columns * sizeof(float));
    float* h_prod1 = new float[columns * columns];
    std::memset(h_prod1, 0, columns * columns * sizeof(float));
    float* h_prod2 = new float[columns * columns];
    std::memset(h_prod2, 0, columns * columns * sizeof(float));

    #pragma omp parallel for schedule(dynamic)
    for (int col1_num = 0; col1_num < columns; ++col1_num) {
        for (int col2_num = col1_num; col2_num < columns; ++col2_num) {
            float* col1 = dense + rows * col1_num;
            float* col2 = dense + rows * col2_num;
            float scalar = 0.0f, p1 = 0.0f, p2 = 0.0f;
            for (int row = 0; row < rows; ++row) {
                scalar += col1[row] * col2[row];
                p1 += col1[row] * col1[row];
                p2 += col2[row] * col2[row];
            }
            h_scalar[col1_num * columns + col2_num] = scalar;
            h_prod1[col1_num * columns + col2_num] = p1;
            h_prod2[col1_num * columns + col2_num] = p2;
            h_scalar[col2_num * columns + col1_num] = scalar;
            h_prod1[col2_num * columns + col1_num] = p1;
            h_prod2[col2_num * columns + col1_num] = p2;
        }
    }

    for (int i = 0; i < columns * columns; ++i) {
        result[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);
    }

    free(h_prod1);
    free(h_prod2);
    free(h_scalar);
    delete[] dense;
}

extern "C" void matrix_Spearman_sparse_distance_different_blocks_cpu(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
    int rows = *num_rows;
    int columns = *num_columns;
    int columns_b = *num_columns_b;

    float* dense_a = csr_to_ranked_dense_cpu(a_index, a_positions, a_double_values, rows, columns);
    float* dense_b = csr_to_ranked_dense_cpu(b_index, b_positions, b_double_values, rows, columns_b);

    float* h_scalar = new float[columns * columns_b];
    std::memset(h_scalar, 0, columns * columns_b * sizeof(float));
    float* h_prod1 = new float[columns * columns_b];
    std::memset(h_prod1, 0, columns * columns_b * sizeof(float));
    float* h_prod2 = new float[columns * columns_b];
    std::memset(h_prod2, 0, columns * columns_b * sizeof(float));

    parallel_for_cols(columns, [&](int col1_start, int col1_end) {
        for (int row = 0; row < rows; row++) {
            for (int col1_num = col1_start; col1_num < col1_end; ++col1_num) {
                float* col1 = dense_a + rows * col1_num;
                for (int col2_num = 0; col2_num < columns_b; ++col2_num) {
                    float* col2 = dense_b + rows * col2_num;
                    float num = col1[row] * col2[row];
                    float sum1 = col1[row] * col1[row];
                    float sum2 = col2[row] * col2[row];
                    h_scalar[col2_num * columns + col1_num] += num;
                    h_prod1[col2_num * columns + col1_num] += sum1;
                    h_prod2[col2_num * columns + col1_num] += sum2;
                }
            }
        }
    });

    for (int i = 0; i < columns * columns_b; ++i) {
        result[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);
    }

    delete[] h_prod1;
    delete[] h_prod2;
    delete[] h_scalar;
    delete[] dense_a;
    delete[] dense_b;
}

// ==================== True Pearson (centered cosine) ====================

extern "C" void matrix_Pearson_distance_same_block_cpu(
    double* a, double* b, double* c, int* n, int* m, int* m_b
) {
    int array_size = *n * *m;
    float* array_new = new float[array_size];
    for (int i = 0; i < array_size; ++i) {
        array_new[i] = static_cast<float>(a[i]);
    }
    center_columns_cpu(array_new, *n, *m);

    float* h_scalar = new float[(*m) * (*m)];
    std::memset(h_scalar, 0, (*m) * (*m) * sizeof(float));
    float* h_prod1 = new float[(*m) * (*m)];
    std::memset(h_prod1, 0, (*m) * (*m) * sizeof(float));
    float* h_prod2 = new float[(*m) * (*m)];
    std::memset(h_prod2, 0, (*m) * (*m) * sizeof(float));

    #pragma omp parallel for schedule(dynamic)
    for (int col1_num = 0; col1_num < *m; ++col1_num) {
        for (int col2_num = col1_num; col2_num < *m; ++col2_num) {
            float* col1 = array_new + *n * col1_num;
            float* col2 = array_new + *n * col2_num;
            float scalar = 0.0f, p1 = 0.0f, p2 = 0.0f;
            for (int row = 0; row < *n; ++row) {
                scalar += col1[row] * col2[row];
                p1 += col1[row] * col1[row];
                p2 += col2[row] * col2[row];
            }
            h_scalar[col1_num * *m + col2_num] = scalar;
            h_prod1[col1_num * *m + col2_num] = p1;
            h_prod2[col1_num * *m + col2_num] = p2;
            h_scalar[col2_num * *m + col1_num] = scalar;
            h_prod1[col2_num * *m + col1_num] = p1;
            h_prod2[col2_num * *m + col1_num] = p2;
        }
    }

    for (int i = 0; i < (*m) * (*m); ++i) {
        c[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);
    }

    free(h_prod1);
    free(h_prod2);
    free(h_scalar);
    free(array_new);
}

extern "C" void matrix_Pearson_distance_different_blocks_cpu(
    double* a, double* b, double* c, int* n, int* m, int* m_b
) {
    int array_size = *n * *m;
    float* d_array = new float[array_size];
    for (int i = 0; i < array_size; ++i) {
        d_array[i] = static_cast<float>(a[i]);
    }
    center_columns_cpu(d_array, *n, *m);

    int array2_size = *n * (*m_b);
    float* d_array2 = new float[array2_size];
    for (int i = 0; i < array2_size; ++i) {
        d_array2[i] = static_cast<float>(b[i]);
    }
    center_columns_cpu(d_array2, *n, *m_b);

    float* h_scalar = new float[(*m) * (*m_b)];
    std::memset(h_scalar, 0, (*m) * (*m_b) * sizeof(float));
    float* h_prod1 = new float[(*m) * (*m_b)];
    std::memset(h_prod1, 0, (*m) * (*m_b) * sizeof(float));
    float* h_prod2 = new float[(*m) * (*m_b)];
    std::memset(h_prod2, 0, (*m) * (*m_b) * sizeof(float));

    parallel_for_cols(*m, [&](int col1_start, int col1_end) {
        for (int row = 0; row < *n; row++) {
            for (int col1_num = col1_start; col1_num < col1_end; ++col1_num) {
                float* col1 = d_array + *n * col1_num;
                for (int col2_num = 0; col2_num < *m_b; ++col2_num) {
                    float* col2 = d_array2 + *n * col2_num;
                    float num = col1[row] * col2[row];
                    float sum1 = col1[row] * col1[row];
                    float sum2 = col2[row] * col2[row];
                    h_scalar[col2_num * *m + col1_num] += num;
                    h_prod1[col2_num * *m + col1_num] += sum1;
                    h_prod2[col2_num * *m + col1_num] += sum2;
                }
            }
        }
    });

    for (int i = 0; i < (*m) * (*m_b); ++i) {
        c[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);
    }

    delete[] h_prod1;
    delete[] h_prod2;
    delete[] h_scalar;
    delete[] d_array;
    delete[] d_array2;
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
  int rows = *num_rows;
  int columns = *num_columns;

  float * a_values = new float[*num_elements_a];
  float * float_result = new float[columns * columns];
  std::memset(float_result, 0, columns * columns * sizeof(float));

  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = a_double_values[i];
  }

  float * squares = new float[columns];
  std::memset(squares, 0, columns * sizeof(float));
  float * col_sums = new float[columns];
  std::memset(col_sums, 0, columns * sizeof(float));

  int result_size = columns * columns;
  int nt = get_num_threads(rows);
  std::vector<float*> local_results(nt);
  std::vector<float*> local_squares(nt);
  std::vector<float*> local_sums(nt);
  for (int t = 0; t < nt; ++t) {
    local_results[t] = new float[result_size];
    std::memset(local_results[t], 0, result_size * sizeof(float));
    local_squares[t] = new float[columns];
    std::memset(local_squares[t], 0, columns * sizeof(float));
    local_sums[t] = new float[columns];
    std::memset(local_sums[t], 0, columns * sizeof(float));
  }

  parallel_for_with_id(rows, nt, [&](int t, int row_start, int row_end) {
    float* local = local_results[t];
    float* lsq = local_squares[t];
    float* lsum = local_sums[t];
    for (int row_index = row_start; row_index < row_end; ++row_index) {
      int start_column = a_positions[row_index];
      int end_column = a_positions[row_index + 1];

      for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
        int col1 = a_index[col1_index];
        float value1 = a_values[col1_index];
        lsq[col1] += value1 * value1;
        lsum[col1] += value1;

        for (int col2_index = col1_index + 1; col2_index < end_column; ++col2_index) {
          int col2 = a_index[col2_index];
          float value2 = a_values[col2_index];

          local[col1 * columns + col2] += value1 * value2;
          local[col2 * columns + col1] += value1 * value2;
        }
      }
    }
  });

  std::memset(float_result, 0, result_size * sizeof(float));
  std::memset(squares, 0, columns * sizeof(float));
  std::memset(col_sums, 0, columns * sizeof(float));
  for (int t = 0; t < nt; ++t) {
    for (int i = 0; i < result_size; ++i) float_result[i] += local_results[t][i];
    for (int i = 0; i < columns; ++i) squares[i] += local_squares[t][i];
    for (int i = 0; i < columns; ++i) col_sums[i] += local_sums[t][i];
    delete[] local_results[t];
    delete[] local_squares[t];
    delete[] local_sums[t];
  }

  for (int i = 0; i < result_size; ++i) {
    int row_index = i / columns;
    int column_index = i % columns;
    if (row_index != column_index) {
        float dot_centered = float_result[i] - col_sums[row_index] * col_sums[column_index] / rows;
        float norm1 = std::sqrt(squares[row_index] - col_sums[row_index] * col_sums[row_index] / rows);
        float norm2 = std::sqrt(squares[column_index] - col_sums[column_index] * col_sums[column_index] / rows);
        result[i] = 1.0f - dot_centered / norm1 / norm2;
    } else {
        result[i] = 0.0f;
    }
  }

  free(float_result);
  free(squares);
  free(col_sums);
  free(a_values);
}

// Pearson sparse different_blocks: sliding-window CSR with mean correction
extern "C" void matrix_Pearson_sparse_distance_different_blocks_cpu(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
  int rows = *num_rows;
  int columns = *num_columns;
  int columns_b = *num_columns_b;

  float * a_values = new float[*num_elements_a];
  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = a_double_values[i];
  }
  float * b_values = new float[*num_elements_b];
  for (int i = 0; i < *num_elements_b; ++i) {
    b_values[i] = b_double_values[i];
  }

  int result_size = columns * columns_b;
  int nt = get_num_threads(rows);
  std::vector<float*> local_results(nt);
  std::vector<float*> local_sqa(nt), local_sqb(nt);
  std::vector<float*> local_suma(nt), local_sumb(nt);
  for (int t = 0; t < nt; ++t) {
    local_results[t] = new float[result_size];
    std::memset(local_results[t], 0, result_size * sizeof(float));
    local_sqa[t] = new float[columns];
    std::memset(local_sqa[t], 0, columns * sizeof(float));
    local_sqb[t] = new float[columns_b];
    std::memset(local_sqb[t], 0, columns_b * sizeof(float));
    local_suma[t] = new float[columns];
    std::memset(local_suma[t], 0, columns * sizeof(float));
    local_sumb[t] = new float[columns_b];
    std::memset(local_sumb[t], 0, columns_b * sizeof(float));
  }

  parallel_for_with_id(rows, nt, [&](int t, int row_start, int row_end) {
    float* local = local_results[t];
    float* lsqa = local_sqa[t];
    float* lsqb = local_sqb[t];
    float* lsuma = local_suma[t];
    float* lsumb = local_sumb[t];

    for (int row_index = row_start; row_index < row_end; ++row_index) {
      int start_column = a_positions[row_index];
      int end_column = a_positions[row_index + 1];
      int start_column_b = b_positions[row_index];
      int end_column_b = b_positions[row_index + 1];

      for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
        float value1 = a_values[col1_index];
        int col1 = a_index[col1_index];

        for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
          int col2 = b_index[col2_index];
          float value2 = b_values[col2_index];
          local[col2 * columns + col1] += value1 * value2;
        }
      }

      for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
        float value1 = a_values[col1_index];
        int col1 = a_index[col1_index];
        lsqa[col1] += value1 * value1;
        lsuma[col1] += value1;
      }
      for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
        float value2 = b_values[col2_index];
        int col2 = b_index[col2_index];
        lsqb[col2] += value2 * value2;
        lsumb[col2] += value2;
      }
    }
  });

  float* float_result = new float[result_size];
  float* squares_a = new float[columns];
  float* squares_b = new float[columns_b];
  float* sums_a = new float[columns];
  float* sums_b = new float[columns_b];
  std::memset(float_result, 0, result_size * sizeof(float));
  std::memset(squares_a, 0, columns * sizeof(float));
  std::memset(squares_b, 0, columns_b * sizeof(float));
  std::memset(sums_a, 0, columns * sizeof(float));
  std::memset(sums_b, 0, columns_b * sizeof(float));
  for (int t = 0; t < nt; ++t) {
    for (int i = 0; i < result_size; ++i) float_result[i] += local_results[t][i];
    for (int i = 0; i < columns; ++i) squares_a[i] += local_sqa[t][i];
    for (int i = 0; i < columns_b; ++i) squares_b[i] += local_sqb[t][i];
    for (int i = 0; i < columns; ++i) sums_a[i] += local_suma[t][i];
    for (int i = 0; i < columns_b; ++i) sums_b[i] += local_sumb[t][i];
    delete[] local_results[t];
    delete[] local_sqa[t];
    delete[] local_sqb[t];
    delete[] local_suma[t];
    delete[] local_sumb[t];
  }

  for (int i = 0; i < result_size; ++i) {
    int row_index = i / columns;
    int column_index = i % columns;
    float dot_centered = float_result[i] - sums_b[row_index] * sums_a[column_index] / rows;
    float norm1 = std::sqrt(squares_b[row_index] - sums_b[row_index] * sums_b[row_index] / rows);
    float norm2 = std::sqrt(squares_a[column_index] - sums_a[column_index] * sums_a[column_index] / rows);
    result[i] = 1.0f - dot_centered / norm1 / norm2;
  }

  free(float_result);
  free(squares_a);
  free(squares_b);
  free(sums_a);
  free(sums_b);
  free(a_values);
  free(b_values);
}
