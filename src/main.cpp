//=============================
#include <iostream>
#include <fstream>
#include <R.h>
#include <stdio.h>
#include <cstring>
#include <math.h>
#include <cassert>
//=========================================

//Naive Implementation of Euclidean_distance_matrix (BruteForce)
extern "C" void matrix_Euclidean_distance_same_block_cpu(double * a, double * b, double * c, int * n, int * m, int * m_b) {
  int array_size = * n * * m;
  float * array_new = new float[ * n * * m];
  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  float * d_array = new float[array_size];

  std::memcpy(d_array, array_new, array_size * sizeof(float));

  //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  float * h_result = new float[( * m) * ( * m)];
  std::memset(h_result, 0, ( * m) * ( * m) * sizeof(float));

  //CPU Implementation
  for (int row = 0; row < * n; row++) {
    for (int col1_num = 0; col1_num < * m; ++col1_num) {
      for (int col2_num = col1_num + 1; col2_num < * m; ++col2_num) {
        float * col1 = d_array + * n * col1_num;
        float * col2 = d_array + * n * col2_num;
        if (row < * n) {
          float diff = col1[row] - col2[row];
          diff = diff * diff;
          h_result[col1_num * * m + col2_num] += diff;
          h_result[col2_num * * m + col1_num] += diff;
        }
      }
    }
  }


  for (int i = 0; i < ( * m) * ( * m); ++i) {
    c[i] = std::sqrt(h_result[i]); //Using sqrt instead of sqrtf
  }
  free(h_result);
  free(d_array);
}
//=============================================
//Naive Implementation of Kendall_distance_matrix for different blocks
extern "C" void matrix_Kendall_distance_same_block_cpu(double * a, double * b, double * c, int * n, int * m, int * m_b) {
  int array_size = * n * * m;
  float * array_new = new float[ * n * * m];
  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  float * d_array = new float[array_size];

  std::memcpy(d_array, array_new, array_size * sizeof(float));

  unsigned int * h_result = new unsigned int[( * m) * ( * m)];
  std::memset(h_result, 0, ( * m) * ( * m) * sizeof(unsigned int));

  //CPU Implementation
  for (int row1 = 0; row1 < * n; row1++) {
    for (int row2=0;row2<*n;row2++){
      for (int col1_num = 0; col1_num < * m; ++col1_num) {
        for (int col2_num = col1_num+1; col2_num < * m; ++col2_num) {
          float * col1 = d_array + * n * col1_num;
          float * col2 = d_array + * n * col2_num;
          if (row1 < row2 && row2 < *n){
            if ((col1[row1] - col1[row2]) * (col2[row1] - col2[row2]) < 0){
              h_result[col1_num * * m + col2_num] += 1;
              h_result[col2_num * * m + col1_num] += 1;
            }
          }      
        }
      }
    }
  }
  
  for (int i = 0; i < ( * m) * ( * m); ++i) {
    c[i] = h_result[i] * 2.0f / (*n) / (*n - 1);
  }
  free(h_result);
  free(d_array);
}

extern "C" void  matrix_Pearson_distance_same_block_cpu(
  double* a,
  double* b,
  double* c,
  int* n,
  int* m,
  int* m_b
) {
  int array_size = * n * * m;
  float * array_new = new float[ * n * * m];
  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }


  float * h_scalar = new float[( * m) * ( * m)];
  std::memset(h_scalar, 0, ( * m) * ( * m) * sizeof(float));
 
  float * h_prod1 = new float[( * m) * ( * m)];
  std::memset(h_prod1, 0, ( * m) * ( * m) * sizeof(float));
 
  float * h_prod2 = new float[( * m) * ( * m)];
  std::memset(h_prod2, 0, ( * m) * ( * m) * sizeof(float));
 

  for (int row=0;row<*n;row++){
    for (int col1_num = 0; col1_num < * m; ++col1_num) {
      for (int col2_num = col1_num; col2_num < * m; ++col2_num) {
        float * col1 = array_new + * n * col1_num;
        float * col2 = array_new + * n * col2_num;
        float num = (col1[row] * col2[row]);
        float sum1 = (col1[row] * col1[row]);
        float sum2 = (col2[row] * col2[row]);
        h_scalar[col1_num * * m + col2_num] += num;
        h_prod1[col1_num * * m + col2_num] += sum1;
        h_prod2[col1_num * * m + col2_num] += sum2;
        h_scalar[col2_num * * m + col1_num] += num;
        h_prod1[col2_num * * m + col1_num] += sum1;
        h_prod2[col2_num * * m + col1_num] += sum2;
      }
    }
  }

  int j=0;
  for (int i = 0; i < (*m) * (*m); ++i) {
    c[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);

  }

  free(h_prod1);
  free(h_prod2);
  free(h_scalar);
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

  //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  unsigned int * h_result = new unsigned int[( * m) * ( * m_b)];
  std::memset(h_result, 0, ( * m) * ( * m_b) * sizeof(unsigned int));

  //CPU Implementation
  for (int row = 0; row < * n; row++) {
    for (int col1_num = 0; col1_num < * m; ++col1_num) {
      for (int col2_num = 0; col2_num < * m_b; ++col2_num) {
        float * col1 = d_array + * n * col1_num;
        float * col2 = d_array + * n * col2_num;
        if (row < * n) {
          float diff = col1[row] - col2[row];
          diff = diff * diff;
          h_result[col2_num * * m + col1_num] += diff;
        }
      }
    }
  }


  for (int i = 0; i < ( * m) * ( * m_b); ++i) {
    c[i] = std::sqrt(h_result[i]); //Using sqrt instead of sqrtf
  }
  free(h_result);
  free(d_array);
  free(d_array2);
}

extern "C" void  matrix_Kendall_distance_different_blocks_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b){
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

  unsigned int * h_result = new unsigned int[( * m) * ( * m_b)];
  std::memset(h_result, 0, ( * m) * ( * m_b) * sizeof(unsigned int));

  //CPU Implementation
  for (int row1 = 0; row1 < * n; row1++) {
    for (int row2=0;row2<*n;row2++){
    for (int col1_num = 0; col1_num < * m; ++col1_num) {
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


  for (int i = 0; i < ( * m) * ( * m_b); ++i) {
    c[i] = h_result[i] * 2.0f / (*n) / (*n - 1);
  }
  free(h_result);
  free(d_array);
  free(d_array2);
}

extern "C" void  matrix_Pearson_distance_different_blocks_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b){
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
 

  for (int row=0;row<*n;row++){
    for (int col1_num = 0; col1_num < * m; ++col1_num) {
      for (int col2_num = col1_num+1; col2_num < * m_b; ++col2_num) {
        float * col1 = d_array + * n * col1_num;
        float * col2 = d_array + * n * col2_num;
        if (row < *n ) {    
	    float num = (col1[row] * col2[row]);
            float sum1 = (col1[row] * col1[row]);
            float sum2 = (col2[row] * col2[row]);
	        h_scalar[col1_num * * m_b + col2_num] += num;
            h_prod1[col1_num * * m_b + col2_num] += sum1;
            h_prod2[col1_num * * m_b + col2_num] += sum2;
            h_scalar[col2_num * * m + col1_num] += num;
            h_prod1[col2_num * * m + col1_num] += sum1;
            h_prod2[col2_num * * m + col1_num] += sum2;
                                  
       }
      }
    }
  }
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
  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = a_double_values[i];
  }

  for (int i = 0; i < columns * columns; ++i) {
    float_result[i] = 0.0f;
  }

  for (int row_index = 0; row_index < rows; ++row_index) {
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
          float_result[left * columns + col2] += value2 * value2;
          float_result[col2 * columns + left] += value2 * value2;
        }

        for (int right = col2 + 1; right < next_col; ++right) {
          float_result[right * columns + col1] += value1 * value1;
          float_result[col1 * columns + right] += value1 * value1;

        }

        float_result[col1 * columns + col2] += (value1 - value2) * (value1 - value2);
        float_result[col2 * columns + col1] += (value1 - value2) * (value1 - value2);
      }
    }
  }

  for (int i = 0; i < columns * columns; ++i) {
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


  for (int row_index = 0; row_index < rows; ++row_index) {
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
        // std::cout << col1_index << " " << start_column << " " << end_column << " " << col2_index << std::endl;
        int prev_col_b_index = col2_index - 1;
        int prev_col2 = (prev_col_b_index >= start_column_b) ? b_index[prev_col_b_index] : -1;

        int col2 = (col2_index < end_column_b) ? b_index[col2_index] : columns_b;

        
        float value2 = (col2_index < end_column_b) ? b_values[col2_index] : 0.0f;
        
        if (col2 < columns_b) {
          for (int left = prev_col + 1; left < col1; ++left) {
            float_result[col2 * columns + left] += value2 * value2;
          }
        }
        if (col1 < columns) {
          for (int left = prev_col2 + 1; left < col2; ++left) {
            float_result[left * columns + col1] += value1 * value1;
          }
        }

        if (col1 < columns && col2 < columns_b) {
          float_result[col2 * columns + col1] += (value1 - value2) * (value1 - value2);
        }
      }
    }
  }

  for (int i = 0; i < columns * columns_b; ++i) {
    result[i] = std::sqrt(float_result[i]);
  }
  
  free(float_result);
  free(a_values);
  free(b_values);
}


extern "C" void matrix_Pearson_sparse_distance_same_block_cpu(
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
  

  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = a_double_values[i];
  }

  for (int i = 0; i < columns * columns; ++i) {
    float_result[i] = 0.0f;
  }

  float * squares = new float[columns];
  for (int i = 0; i < columns; ++i) {
    squares[i] = 0.0f;
  }


  for (int row_index = 0; row_index < rows; ++row_index) {
    int start_column = a_positions[row_index];
    int end_column = a_positions[row_index + 1];

    for (int col1_index = start_column; col1_index < end_column; ++col1_index) {

      int prev_col_index = col1_index - 1;
      int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
      int col1 = a_index[col1_index];
      float value1 = a_values[col1_index];
      squares[col1] += value1 * value1;

      for (int col2_index = col1_index + 1; col2_index < end_column; ++col2_index) {
        

        int next_col_index = col2_index + 1;
        int next_col = (next_col_index < end_column) ? a_index[next_col_index] : columns;

        int col2 = a_index[col2_index];

        
        float value2 = a_values[col2_index];

        float_result[col1 * columns + col2] += value1 * value2;
        float_result[col2 * columns + col1] += value1 * value2;
      }
    }
  }
  for (int i = 0; i < columns * columns; ++i) {
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

extern "C" void  matrix_Pearson_sparse_distance_different_blocks_cpu(
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
  for (int i = 0; i < columns; ++i) {
    squares_a[i] = 0.0f;
  }

  float * squares_b = new float[columns_b];
  for (int i = 0; i < columns_b; ++i) {
    squares_b[i] = 0.0f;
  }


  for (int row_index = 0; row_index < rows; ++row_index) {
    int start_column = a_positions[row_index];
    int end_column = a_positions[row_index + 1];

    int start_column_b = b_positions[row_index];
    int end_column_b = b_positions[row_index + 1];

    for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
      float value1 = a_values[col1_index];

      int col1 = a_index[col1_index];

      for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
        // std::cout << col1_index << " " << start_column << " " << end_column << " " << col2_index << std::endl;
        int col2 = b_index[col2_index];       
        float value2 = b_values[col2_index];

        float_result[col2 * columns + col1] += value1 * value2;

      }
    }

    for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
      float value1 = a_values[col1_index];

      int col1 = a_index[col1_index];

      squares_a[col1] += value1 * value1;
    }
    for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
      float value2 = b_values[col2_index];
      int col2 = b_index[col2_index];

      squares_b[col2] += value2 * value2;
    }
  }

  for (int i = 0; i < columns * columns_b; ++i) {
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

  int *disconcordant = new int[columns * columns];
  std::memset(disconcordant, 0, columns * columns * sizeof(int));

  float *a_values = new float[*num_elements_a];
  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = static_cast<float>(a_double_values[i]);
  }

  bool* left_thresholds = new bool[columns];
  
  
  for (int row_index = 0; row_index < rows; ++row_index) {

    int start_column = a_positions[row_index];

    int end_column = a_positions[row_index + 1];
    for (int row_jndex = row_index + 1; row_jndex < rows; ++row_jndex) {  

      int start_column_b = a_positions[row_jndex];
      int end_column_b = a_positions[row_jndex + 1];
      
      for (int i = 0; i < end_column_b - start_column_b; ++i) {
        left_thresholds[i] = false;
      }
      bool left_threshold_selected = false;
      bool right_threshold_selected = false;
      int right_down1_threshold = start_column_b;
      int left_down1_threshold = start_column_b;
      int left_down2_threshold = start_column_b;
      int right_down2_threshold = start_column_b;
      bool left_activated = false;
      bool right_activated = false;
      
      for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
        int prev_col_index = col1_index - 1;
        int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
        int col1 = a_index[col1_index];
        float value1 = a_values[col1_index];
        

        while (right_down1_threshold < end_column_b && a_index[right_down1_threshold] < col1) {
          right_down1_threshold += 1;
        }

        if (right_down1_threshold < end_column_b && a_index[right_down1_threshold] == col1) {
          left_activated = true;
        }
        if (right_down1_threshold < end_column_b && a_index[right_down1_threshold] == col1) {
          left_down2_threshold = right_down1_threshold + 1;
          right_activated = true;
        } else {
          left_down2_threshold = right_down1_threshold;
        }

        right_down2_threshold = left_down2_threshold;
        for (int col2_index = col1_index; col2_index < end_column; ++col2_index) {
          int col2 = a_index[col2_index];
          float value2 = a_values[col2_index];
          int next_col_index = col2_index + 1;
          int next_col = (next_col_index < end_column) ? a_index[next_col_index] : columns;

          while (right_down2_threshold < end_column_b && a_index[right_down2_threshold] < next_col) {
              right_down2_threshold += 1;
          }
          if (left_down1_threshold < end_column_b && !left_thresholds[left_down1_threshold - start_column_b]) {
            left_thresholds[left_down1_threshold - start_column_b] = true;
            for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
                for (int right = left + 1; right < right_down1_threshold; ++right) {
                  float product = a_values[left] * a_values[right];
                  if (product < 0) {
                    disconcordant[a_index[left] * columns + a_index[right]] += 1;
                    disconcordant[a_index[right] * columns + a_index[left]] += 1;
                  } 
                }
            }
          }
          
          if (left_down2_threshold < end_column_b && !left_thresholds[left_down2_threshold - start_column_b]) {
            left_thresholds[left_down2_threshold - start_column_b] = true;
            for (int left = left_down2_threshold; left < right_down2_threshold; left++) {
                for (int right = left + 1; right < right_down2_threshold; ++right) {
                  float product = a_values[left] * a_values[right];
                  if (product < 0) {
                    disconcordant[a_index[left] * columns + a_index[right]] += 1;
                    disconcordant[a_index[right] * columns + a_index[left]] += 1;
                  } 
                }
            }
          }
          
          for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
                for (int right = left_down2_threshold; right < right_down2_threshold; ++right) {
                  float product = a_values[left] * a_values[right];
                  if (product < 0) {
                    disconcordant[a_index[left] * columns + a_index[right]] += 1;
                    disconcordant[a_index[right] * columns + a_index[left]] += 1;
                  } 
                }
            }
          
          float left_value = (left_activated) ? a_values[right_down1_threshold] : 0;
          float right_value = (right_activated) ? a_values[left_down2_threshold - 1] : 0;
          
          float left_diff = left_value - a_values[col1_index];
          float right_diff = right_value - a_values[col2_index];
          float product = left_diff * right_diff;
          if (product < 0) {
              disconcordant[col1 * columns + col2] += 1;
              disconcordant[col2 * columns + col1] += 1;
          }
          
          for (int right = left_down2_threshold; right < right_down2_threshold; ++right) {
            product = left_diff * a_values[right];
            if (product < 0) {
              disconcordant[col1 * columns + a_index[right]] += 1;
              disconcordant[a_index[right] * columns + col1] += 1;
            } 
          }
              
          for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
            product = right_diff * a_values[left];
            if (product < 0) {
                disconcordant[a_index[left] * columns + col2] += 1;
                disconcordant[col2 * columns + a_index[left]] += 1;
            } 
          }

          right_activated = false;
          while (left_down2_threshold < end_column_b && a_index[left_down2_threshold] <= next_col) {
              if (a_index[left_down2_threshold] == next_col) {
                right_activated = true;
              }
              left_down2_threshold += 1;
          }

        }
        
        while (left_down1_threshold < end_column_b && a_index[left_down1_threshold] <= col1) {
          left_down1_threshold += 1;
        }
      }
    }
  }
  for (int i = 0; i < columns * columns; ++i) {
    result[i] = static_cast<double>(disconcordant[i]) * 2.0f / rows / (rows - 1);
  }

  delete[] disconcordant;
  delete[] left_thresholds;
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

  int *disconcordant = new int[columns * columns_b];
  std::memset(disconcordant, 0, columns * columns_b * sizeof(int));
  float *a_values = new float[*num_elements_a];
  float *b_values = new float[*num_elements_b];

  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = static_cast<float>(a_double_values[i]);
  }

  for (int i = 0; i < *num_elements_b; ++i) {
    b_values[i] = static_cast<float>(b_double_values[i]);
  }

  for (int row_index = 0; row_index < rows; ++row_index) {
    for (int row_jndex = row_index + 1; row_jndex < rows; ++row_jndex) {
      int start_column = a_positions[row_index];
      int end_column = a_positions[row_index + 1];
      int start_column_down = a_positions[row_jndex];
      int end_column_down = a_positions[row_jndex + 1];

      int start_column_b = b_positions[row_index];
      int end_column_b = b_positions[row_index + 1];
      int start_column_down_b = b_positions[row_jndex];
      int end_column_down_b = b_positions[row_jndex + 1];
      bool left_threshold_selected = false;
      bool right_threshold_selected = false;
      int right_down1_threshold = start_column_down;
      int left_down1_threshold = start_column_down;
      int left_down2_threshold = start_column_down_b;
      int right_down2_threshold = start_column_down_b;
      bool left_activated = false;
      bool right_activated = false;

      for (int col1_index = start_column; col1_index <= end_column; ++col1_index) {
        int prev_col_index = col1_index - 1;
        int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
        int col1 = (col1_index < end_column) ? a_index[col1_index]: columns;
        float value1 = (col1_index < end_column) ? a_values[col1_index]: 0;
        
        while (right_down1_threshold < end_column_down && a_index[right_down1_threshold] < col1) {
          right_down1_threshold += 1;
        }

        if (right_down1_threshold < end_column_down && a_index[right_down1_threshold] == col1) {
          left_activated = true;
        }

        int left_down2_threshold = start_column_down_b;
        int right_down2_threshold = start_column_down_b;
        right_activated = false;
        for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
          int col2 = b_index[col2_index];
          float value2 = b_values[col2_index];
          int next_col_index = col2_index + 1;
          int next_col = (next_col_index < end_column_b) ? b_index[next_col_index] : columns_b;


          while (right_down2_threshold < end_column_down_b && b_index[right_down2_threshold] < next_col) {
              right_down2_threshold += 1;
          }
          
          for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
                for (int right = left_down2_threshold; right < right_down2_threshold; ++right) {
                  float product = a_values[left] * b_values[right];
                  assert(b_index[right] < columns_b);
                  if (product < 0) {
                    disconcordant[b_index[right] * columns + a_index[left]] += 1;
                  } 
                }
            }
          
          float left_value = (left_activated) ? a_values[right_down1_threshold] : 0;
          float right_value = (right_activated) ? b_values[left_down2_threshold - 1] : 0;
          
          
          float right_diff = right_value - value2;
          
          float left_diff = left_value - value1;
          float product = left_diff * right_diff;
          if (product < 0) {
              disconcordant[col2 * columns + col1] += 1;
          }
          
          for (int right = left_down2_threshold; right < right_down2_threshold; ++right) {
            product = left_diff * b_values[right];
            if (product < 0) {
              disconcordant[b_index[right] * columns + col1] += 1;
            } 
          }
              
          for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
            product = right_diff * a_values[left];
            if (product < 0) {
                disconcordant[col2 * columns + a_index[left]] += 1;
            } 
          }

          right_activated = false;
          while (left_down2_threshold < end_column_down_b && b_index[left_down2_threshold] <= next_col) {
              if (b_index[left_down2_threshold] == next_col) {
                right_activated = true;
              }
              left_down2_threshold += 1;
          }

        }
        while (left_down1_threshold < end_column_down && a_index[left_down1_threshold] <= col1) {
          left_down1_threshold += 1;
        }
      }
    }
  }
  for (int i = 0; i < columns * columns_b; ++i) {
    result[i] = static_cast<double>(disconcordant[i]) * 2.0f / rows / (rows - 1); 
  }

  delete[] disconcordant;
  delete[] a_values;
  delete[] b_values;
}

