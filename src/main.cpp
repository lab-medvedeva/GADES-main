//=============================
#include <iostream>
#include <fstream>
#include <R.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cstdint>
//=========================================

namespace internal
{

// Naive Implementation of Euclidean_distance_matrix (BruteForce)
void dist_euclid_same(DenseMatrixView<double> matr, DenseMatrixView<double> res)
{
  float* matr_float_data = new float[matr.row_num * matr.col_num];
  for (size_t i = 0; i < matr.row_num * matr.col_num; ++i)
  {
    matr_float_data[i] = matr.data[i];
  }

  DenseMatrixView<float> matr_float = {
    .data = matr_float_data, .row_num = matr.row_num, .col_num = matr.col_num};

  float* res_float_data = new float[matr.col_num * matr.col_num];
  std::memset(res_float_data, 0, matr.col_num * matr.col_num * sizeof(float));

  DenseMatrixView<float> res_float = {
    .data = res_float_data, .row_num = matr.col_num, .col_num = matr.col_num};


  // CPU Implementation
  for (size_t row = 0; row < matr.row_num; row++)
  {
    for (size_t col1 = 0; col1 < matr.col_num - 1; ++col1)
    {
      for (size_t col2 = col1 + 1; col2 < matr.col_num; ++col2)
      {
        float diff = matr_float(row, col1) - matr_float(row, col2);
        diff = diff * diff;
        res_float(col1, col2) += diff;
      }
    }
  }

  for (size_t i = 0; i < matr.col_num - 1; ++i)
  {
    for (size_t j = i + 1; j < matr.col_num; ++j)
    {
      double curr = std::sqrt(res_float(i, j));
      res(i, j) = curr;
      res(j, i) = curr;
    }
  }

  delete[] (res_float_data);
  delete[] (matr_float_data);
}

void dist_euclid_diff(
  DenseMatrixView<double> matr_a, DenseMatrixView<double> matr_b, DenseMatrixView<double> res)
{
  float* matr_a_float_data = new float[matr_a.row_num * matr_a.col_num];
  for (size_t i = 0; i < matr_a.row_num * matr_a.col_num; ++i)
  {
    matr_a_float_data[i] = matr_a.data[i];
  }

  float* matr_b_float_data = new float[matr_b.row_num * matr_b.col_num];
  for (size_t i = 0; i < matr_b.row_num * matr_b.col_num; ++i)
  {
    matr_b_float_data[i] = matr_b.data[i];
  }

  DenseMatrixView<float> matr_a_float = {
    .data = matr_a_float_data, .row_num = matr_a.row_num, .col_num = matr_a.col_num};

  DenseMatrixView<float> matr_b_float = {
    .data = matr_b_float_data, .row_num = matr_b.row_num, .col_num = matr_b.col_num};

  float* res_float_data = new float[matr_a.col_num * matr_b.col_num];
  std::memset(res_float_data, 0, matr_a.col_num * matr_b.col_num * sizeof(float));

  DenseMatrixView<float> res_float = {
    .data = res_float_data, .row_num = matr_a.col_num, .col_num = matr_b.col_num};

  // CPU Implementation
  for (size_t row = 0; row < matr_a.row_num; row++)
  {
    for (size_t col1 = 0; col1 < matr_a.col_num; ++col1)
    {
      for (size_t col2 = 0; col2 < matr_b.col_num; ++col2)
      {
        float diff = matr_a_float(row, col1) - matr_b_float(row, col2);
        diff = diff * diff;
        res_float(col1, col2) += diff;
      }
    }
  }

  for (size_t i = 0; i < matr_a.col_num * matr_b.col_num; ++i)
  {
    res.data[i] = std::sqrt(res_float.data[i]);
  }

  delete[] (res_float_data);
  delete[] (matr_a_float_data);
  delete[] (matr_b_float_data);
}

void dist_kendall_same(DenseMatrixView<double> matr, DenseMatrixView<double> res)
{
  float* matr_float_data = new float[matr.col_num * matr.row_num];
  for (size_t i = 0; i < matr.col_num * matr.row_num; ++i)
  {
    matr_float_data[i] = matr.data[i];
  }

  DenseMatrixView<float> matr_float = {
    .data = matr_float_data, .row_num = matr.row_num, .col_num = matr.col_num};

  uint32_t* res_float_data = new uint32_t[matr.col_num * matr.col_num];
  std::memset(res_float_data, 0, matr.col_num * matr.col_num * sizeof(uint32_t));

  DenseMatrixView<uint32_t> res_float = {
    .data = res_float_data, .row_num = matr.col_num, .col_num = matr.col_num};

  for (size_t row1 = 0; row1 < matr.row_num - 1; row1++)
  {
    for (size_t row2 = row1 + 1; row2 < matr.row_num; row2++)
    {
      for (size_t col1 = 0; col1 < matr.col_num - 1; ++col1)
      {
        for (size_t col2 = col1 + 1; col2 < matr.col_num; ++col2)
        {
          if (
            (matr_float(row1, col1) < matr_float(row2, col1)) ^
            (matr_float(row1, col2) < matr_float(row2, col2)))
          {
            res_float(col1, col2) += 1;
          }
        }
      }
    }
  }

  float coef = 2.0f / static_cast<float>(matr.row_num * (matr.row_num - 1));
  for (size_t i = 0; i < matr.col_num - 1; ++i)
  {
    for (size_t j = i + 1; j < matr.col_num; ++j)
    {
      double curr = res_float(i, j) / coef;
      res(i, j) = curr;
      res(j, i) = curr;
    }
  }

  delete[] (matr_float_data);
  delete[] (res_float_data);
}

// Naive Implementation of Kendall_distance_matrix for different blocks
void dist_kendall_diff(
  DenseMatrixView<double> matr_a, DenseMatrixView<double> matr_b, DenseMatrixView<double> res)
{
  float* matr_a_float_data = new float[matr_a.row_num * matr_a.col_num];
  for (size_t i = 0; i < matr_a.row_num * matr_a.col_num; ++i)
  {
    matr_a_float_data[i] = matr_a.data[i];
  }

  float* matr_b_float_data = new float[matr_b.row_num * matr_b.col_num];
  for (size_t i = 0; i < matr_b.row_num * matr_b.col_num; ++i)
  {
    matr_b_float_data[i] = matr_b.data[i];
  }

  DenseMatrixView<float> matr_a_float = {
    .data = matr_a_float_data, .row_num = matr_a.row_num, .col_num = matr_a.col_num};

  DenseMatrixView<float> matr_b_float = {
    .data = matr_b_float_data, .row_num = matr_b.row_num, .col_num = matr_b.col_num};

  uint32_t* res_uint_data = new uint32_t[matr_a.col_num * matr_b.col_num];
  std::memset(res_uint_data, 0, matr_a.col_num * matr_b.col_num * sizeof(uint32_t));

  DenseMatrixView<uint32_t> res_uint = {
    .data = res_uint_data, .row_num = matr_a.col_num, .col_num = matr_b.col_num};

  for (size_t row1 = 0; row1 < matr_a.row_num - 1; row1++)
  {
    for (size_t row2 = row1 + 1; row2 < matr_a.row_num; row2++)
    {
      for (size_t col1 = 0; col1 < matr_a.col_num; ++col1)
      {
        for (size_t col2 = 0; col2 < matr_b.col_num; ++col2)
        {
          if (
            (matr_a_float(row1, col1) < matr_a_float(row2, col1)) ^
            (matr_b_float(row1, col2) < matr_b_float(row2, col2)))
          {
            res_uint(col1, col2) += 1;
          }
        }
      }
    }
  }

  double coef = 2.0 / static_cast<double>(matr_a.row_num * (matr_a.row_num - 1));
  for (size_t i = 0; i < matr_a.col_num; ++i)
  {
    for (size_t j = 0; j < matr_b.col_num; ++j)
    {
      double curr = static_cast<double>(res_uint(i, j)) / coef;
      res(i, j) = curr;
    }
  }

  delete[] (res_uint_data);
  delete[] (matr_a_float_data);
  delete[] (matr_b_float_data);
}

void dist_pearson_same(DenseMatrixView<double> matr, DenseMatrixView<double> res)
{
  float* matr_float_data = new float[matr.col_num * matr.row_num];
  for (size_t i = 0; i < matr.col_num * matr.row_num; ++i)
  {
    matr_float_data[i] = matr.data[i];
  }

  DenseMatrixView<float> matr_float = {
    .data = matr_float_data, .row_num = matr.row_num, .col_num = matr.col_num};


  float* res_float_data = new float[matr.col_num * matr.col_num];
  std::memset(res_float_data, 0, matr.col_num * matr.col_num * sizeof(float));

  DenseMatrixView<float> res_float = {
    .data = res_float_data, .row_num = matr.col_num, .col_num = matr.col_num};


  float* norm = new float[matr.col_num];
  std::memset(norm, 0, matr.col_num * sizeof(float));

  for (size_t row = 0; row < matr.row_num; ++row)
  {
    for (size_t col1 = 0; col1 < matr.col_num; ++col1)
    {
      norm[col1] += matr_float(row, col1) * matr_float(row, col1);
      for (size_t col2 = col1 + 1; col2 < matr.col_num; ++col2)
      {
        res_float(col1, col2) += matr_float(row, col1) * matr_float(row, col2);
      }
    }
  }

  for (size_t i = 0; i < matr.col_num; ++i)
  {
    norm[i] = sqrtf(norm[i]);
  }

  for (size_t i = 0; i < matr.col_num - 1; ++i)
  {
    for (size_t j = i + 1; j < matr.col_num; ++j)
    {
      double curr = res_float(i, j) / (norm[i] * norm[j]);
      res(i, j) = curr;
      res(j, i) = curr;
    }
  }

  delete[] (norm);
  delete[] (res_float_data);
  delete[] (matr_float_data);
}

void dist_pearson_diff(
  DenseMatrixView<double> matr_a, DenseMatrixView<double> matr_b, DenseMatrixView<double> res)
{
  float* matr_a_float_data = new float[matr_a.row_num * matr_a.col_num];
  for (size_t i = 0; i < matr_a.row_num * matr_a.col_num; ++i)
  {
    matr_a_float_data[i] = matr_a.data[i];
  }

  float* matr_b_float_data = new float[matr_b.row_num * matr_b.col_num];
  for (size_t i = 0; i < matr_b.row_num * matr_b.col_num; ++i)
  {
    matr_b_float_data[i] = matr_b.data[i];
  }

  DenseMatrixView<float> matr_a_float = {
    .data = matr_a_float_data, .row_num = matr_a.row_num, .col_num = matr_a.col_num};

  DenseMatrixView<float> matr_b_float = {
    .data = matr_b_float_data, .row_num = matr_b.row_num, .col_num = matr_b.col_num};


  float* res_float_data = new float[matr_a.col_num * matr_b.col_num];
  std::memset(res_float_data, 0, matr_a.col_num * matr_b.col_num * sizeof(float));

  DenseMatrixView<float> res_float = {
    .data = res_float_data, .row_num = matr_a.col_num, .col_num = matr_b.col_num};


  float* norm_a = new float[matr_a.col_num];
  std::memset(norm_a, 0, matr_a.col_num * sizeof(float));
  float* norm_b = new float[matr_b.col_num];
  std::memset(norm_b, 0, matr_b.col_num * sizeof(float));

  for (size_t row = 0; row < matr_a.row_num; ++row)
  {
    for (size_t col1 = 0; col1 < matr_a.col_num; ++col1)
    {
      norm_a[col1] += matr_a_float(row, col1) * matr_a_float(row, col1);

      for (size_t col2 = 0; col2 < matr_b.col_num; ++col2)
      {
        res_float(col1, col2) += matr_a_float(row, col1) * matr_b_float(row, col2);
      }
    }
    for (size_t col2 = 0; col2 < matr_b.col_num; ++col2)
    {
      norm_b[col2] += matr_b_float(row, col2) * matr_b_float(row, col2);
    }
  }

  for (size_t i = 0; i < matr_a.col_num; ++i)
  {
    norm_a[i] = sqrtf(norm_a[i]);
  }

  for (size_t i = 0; i < matr_b.col_num; ++i)
  {
    norm_b[i] = sqrtf(norm_b[i]);
  }

  for (size_t i = 0; i < matr_a.col_num; ++i)
  {
    for (size_t j = 0; j < matr_b.col_num; ++j)
    {
      double curr = res_float(i, j) / (norm_a[i] * norm_b[j]);
      res(i, j) = curr;
    }
  }

  delete[] (norm_a);
  delete[] (norm_b);
  delete[] (matr_a_float_data);
  delete[] (matr_b_float_data);
  delete[] (res_float_data);
}

} // namespace internal

// bindings

extern "C" void matrix_Euclidean_distance_same_block_cpu(
  double* a, double* /*unused*/, double* c, int* n, int* m, int* /*unused*/)
{
  DenseMatrixView<double> matr = {
    .data = a, .row_num = static_cast<size_t>(*n), .col_num = static_cast<size_t>(*m)};

  DenseMatrixView<double> res = {
    .data = c, .row_num = static_cast<size_t>(*m), .col_num = static_cast<size_t>(*m)};

  internal::dist_euclid_same(matr, res);
}

//=============================================


extern "C" void matrix_Kendall_distance_same_block_cpu(
  double* a, double* /*unused*/, double* c, int* n, int* m, int* /*unused*/)
{
  DenseMatrixView<double> matr = {
    .data = a, .row_num = static_cast<size_t>(*n), .col_num = static_cast<size_t>(*m)};

  DenseMatrixView<double> res = {
    .data = c, .row_num = static_cast<size_t>(*m), .col_num = static_cast<size_t>(*m)};

  internal::dist_kendall_same(matr, res);
}

extern "C" void matrix_Pearson_distance_same_block_cpu(
  double* a, double* /*unused*/, double* c, int* n, int* m, int* /*unused*/)
{
  DenseMatrixView<double> matr = {
    .data = a, .row_num = static_cast<size_t>(*n), .col_num = static_cast<size_t>(*m)};

  DenseMatrixView<double> res = {
    .data = c, .row_num = static_cast<size_t>(*m), .col_num = static_cast<size_t>(*m)};

  internal::dist_pearson_same(matr, res);
}

//======================================================

extern "C" void matrix_Euclidean_distance_different_blocks_cpu(
  double* a, double* b, double* c, int* n, int* m_a, int* m_b)
{
  DenseMatrixView<double> matr_a = {
    .data = a, .row_num = static_cast<size_t>(*n), .col_num = static_cast<size_t>(*m_a)};

  DenseMatrixView<double> matr_b = {
    .data = b, .row_num = static_cast<size_t>(*n), .col_num = static_cast<size_t>(*m_b)};

  DenseMatrixView<double> res = {
    .data = c, .row_num = static_cast<size_t>(*m_a), .col_num = static_cast<size_t>(*m_b)};

  internal::dist_euclid_diff(matr_a, matr_b, res);
}

extern "C" void matrix_Kendall_distance_different_blocks_cpu(
  double* a, double* b, double* c, int* n, int* m_a, int* m_b)
{

  DenseMatrixView<double> matr_a = {
    .data = a, .row_num = static_cast<size_t>(*n), .col_num = static_cast<size_t>(*m_a)};

  DenseMatrixView<double> matr_b = {
    .data = b, .row_num = static_cast<size_t>(*n), .col_num = static_cast<size_t>(*m_b)};

  DenseMatrixView<double> res = {
    .data = c, .row_num = static_cast<size_t>(*m_a), .col_num = static_cast<size_t>(*m_b)};

  internal::dist_kendall_diff(matr_a, matr_b, res);
}

extern "C" void matrix_Pearson_distance_different_blocks_cpu(
  double* a, double* b, double* c, int* n, int* m_a, int* m_b)
{

  DenseMatrixView<double> matr_a = {
    .data = a, .row_num = static_cast<size_t>(*n), .col_num = static_cast<size_t>(*m_a)};

  DenseMatrixView<double> matr_b = {
    .data = b, .row_num = static_cast<size_t>(*n), .col_num = static_cast<size_t>(*m_b)};

  DenseMatrixView<double> res = {
    .data = c, .row_num = static_cast<size_t>(*m_a), .col_num = static_cast<size_t>(*m_b)};

  internal::dist_pearson_diff(matr_a, matr_b, res);
}


extern "C" void matrix_Euclidean_sparse_distance_same_block_cpu(
  int* a_index,
  int* a_positions,
  double* a_double_values,
  int* b_index,
  int* b_positions,
  double* b_double_values,
  double* result,
  int* num_rows,
  int* num_columns,
  int* num_columns_b,
  int* num_elements_a,
  int* num_elements_b)
{
  int rows = *num_rows;
  int columns = *num_columns;

  float* a_values = new float[*num_elements_a];
  float* float_result = new float[columns * columns];
  for (int i = 0; i < *num_elements_a; ++i)
  {
    a_values[i] = a_double_values[i];
  }

  for (int i = 0; i < columns * columns; ++i)
  {
    float_result[i] = 0.0f;
  }

  for (int row_index = 0; row_index < rows; ++row_index)
  {
    int start_column = a_positions[row_index];
    int end_column = a_positions[row_index + 1];


    for (int col1_index = start_column; col1_index < end_column; ++col1_index)
    {

      for (int col2_index = col1_index; col2_index < end_column; ++col2_index)
      {
        int prev_col_index = col1_index - 1;
        int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;

        int next_col_index = col2_index + 1;
        int next_col = (next_col_index < end_column) ? a_index[next_col_index] : columns;

        int col1 = a_index[col1_index];
        int col2 = a_index[col2_index];

        float value1 = a_values[col1_index];
        float value2 = a_values[col2_index];

        for (int left = prev_col + 1; left < col1; ++left)
        {
          float_result[left * columns + col2] += value2 * value2;
          float_result[col2 * columns + left] += value2 * value2;
        }

        for (int right = col2 + 1; right < next_col; ++right)
        {
          float_result[right * columns + col1] += value1 * value1;
          float_result[col1 * columns + right] += value1 * value1;
        }

        float_result[col1 * columns + col2] += (value1 - value2) * (value1 - value2);
        float_result[col2 * columns + col1] += (value1 - value2) * (value1 - value2);
      }
    }
  }

  for (int i = 0; i < columns * columns; ++i)
  {
    result[i] = std::sqrt(float_result[i]);
  }

  delete[] (float_result);
  delete[] (a_values);
}

extern "C" void matrix_Euclidean_sparse_distance_different_blocks_cpu(
  int* a_index,
  int* a_positions,
  double* a_double_values,
  int* b_index,
  int* b_positions,
  double* b_double_values,
  double* result,
  int* num_rows,
  int* num_columns,
  int* num_columns_b,
  int* num_elements_a,
  int* num_elements_b)
{
  int rows = *num_rows;
  int columns = *num_columns;
  int columns_b = *num_columns_b;

  float* a_values = new float[*num_elements_a];
  float* float_result = new float[columns * columns_b];
  for (int i = 0; i < *num_elements_a; ++i)
  {
    a_values[i] = a_double_values[i];
  }

  for (int i = 0; i < columns * columns_b; ++i)
  {
    float_result[i] = 0.0f;
  }

  float* b_values = new float[*num_elements_b];
  for (int i = 0; i < *num_elements_b; ++i)
  {
    b_values[i] = b_double_values[i];
  }


  for (int row_index = 0; row_index < rows; ++row_index)
  {
    int start_column = a_positions[row_index];
    int end_column = a_positions[row_index + 1];

    int start_column_b = b_positions[row_index];
    int end_column_b = b_positions[row_index + 1];

    for (int col1_index = start_column; col1_index <= end_column; ++col1_index)
    {
      int prev_col_index = col1_index - 1;
      int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
      float value1 = (col1_index < end_column) ? a_values[col1_index] : 0.0f;

      int col1 = (col1_index < end_column) ? a_index[col1_index] : columns;

      for (int col2_index = start_column_b; col2_index <= end_column_b; ++col2_index)
      {
        // std::cout << col1_index << " " << start_column << " " << end_column << " " << col2_index
        // << std::endl;
        int prev_col_b_index = col2_index - 1;
        int prev_col2 = (prev_col_b_index >= start_column_b) ? b_index[prev_col_b_index] : -1;

        int col2 = (col2_index < end_column_b) ? b_index[col2_index] : columns_b;


        float value2 = (col2_index < end_column_b) ? b_values[col2_index] : 0.0f;

        if (col2 < columns_b)
        {
          for (int left = prev_col + 1; left < col1; ++left)
          {
            float_result[col2 * columns + left] += value2 * value2;
          }
        }
        if (col1 < columns)
        {
          for (int left = prev_col2 + 1; left < col2; ++left)
          {
            float_result[left * columns + col1] += value1 * value1;
          }
        }

        if (col1 < columns && col2 < columns_b)
        {
          float_result[col2 * columns + col1] += (value1 - value2) * (value1 - value2);
        }
      }
    }
  }

  for (int i = 0; i < columns * columns_b; ++i)
  {
    result[i] = std::sqrt(float_result[i]);
  }

  delete[] (float_result);
  delete[] (a_values);
  delete[] (b_values);
}


extern "C" void matrix_Pearson_sparse_distance_same_block_cpu(
  int* a_index,
  int* a_positions,
  double* a_double_values,
  int* b_index,
  int* b_positions,
  double* b_double_values,
  double* result,
  int* num_rows,
  int* num_columns,
  int* num_columns_b,
  int* num_elements_a,
  int* num_elements_b)
{
  int rows = *num_rows;
  int columns = *num_columns;

  float* a_values = new float[*num_elements_a];

  float* float_result = new float[columns * columns];


  for (int i = 0; i < *num_elements_a; ++i)
  {
    a_values[i] = a_double_values[i];
  }

  for (int i = 0; i < columns * columns; ++i)
  {
    float_result[i] = 0.0f;
  }

  float* squares = new float[columns];
  for (int i = 0; i < columns; ++i)
  {
    squares[i] = 0.0f;
  }


  for (int row_index = 0; row_index < rows; ++row_index)
  {
    int start_column = a_positions[row_index];
    int end_column = a_positions[row_index + 1];

    for (int col1_index = start_column; col1_index < end_column; ++col1_index)
    {

      // int prev_col_index = col1_index - 1;
      // int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
      int col1 = a_index[col1_index];
      float value1 = a_values[col1_index];
      squares[col1] += value1 * value1;

      for (int col2_index = col1_index + 1; col2_index < end_column; ++col2_index)
      {


        // int next_col_index = col2_index + 1;
        // int next_col = (next_col_index < end_column) ? a_index[next_col_index] : columns;

        int col2 = a_index[col2_index];


        float value2 = a_values[col2_index];

        float_result[col1 * columns + col2] += value1 * value2;
        float_result[col2 * columns + col1] += value1 * value2;
      }
    }
  }
  for (int i = 0; i < columns * columns; ++i)
  {
    int row_index = i / columns;
    int column_index = i % columns;
    if (row_index != column_index)
    {
      result[i] =
        1.0f - float_result[i] / std::sqrt(squares[row_index]) / std::sqrt(squares[column_index]);
    }
    else
    {
      result[i] = 0.0f;
    }
  }

  delete[] (float_result);
  delete[] (squares);
  delete[] (a_values);
}

extern "C" void matrix_Pearson_sparse_distance_different_blocks_cpu(
  int* a_index,
  int* a_positions,
  double* a_double_values,
  int* b_index,
  int* b_positions,
  double* b_double_values,
  double* result,
  int* num_rows,
  int* num_columns,
  int* num_columns_b,
  int* num_elements_a,
  int* num_elements_b)
{
  int rows = *num_rows;
  int columns = *num_columns;
  int columns_b = *num_columns_b;

  float* a_values = new float[*num_elements_a];
  float* float_result = new float[columns * columns_b];
  for (int i = 0; i < *num_elements_a; ++i)
  {
    a_values[i] = a_double_values[i];
  }

  for (int i = 0; i < columns * columns_b; ++i)
  {
    float_result[i] = 0.0f;
  }

  float* b_values = new float[*num_elements_b];
  for (int i = 0; i < *num_elements_b; ++i)
  {
    b_values[i] = b_double_values[i];
  }

  float* squares_a = new float[columns];
  for (int i = 0; i < columns; ++i)
  {
    squares_a[i] = 0.0f;
  }

  float* squares_b = new float[columns_b];
  for (int i = 0; i < columns_b; ++i)
  {
    squares_b[i] = 0.0f;
  }


  for (int row_index = 0; row_index < rows; ++row_index)
  {
    int start_column = a_positions[row_index];
    int end_column = a_positions[row_index + 1];

    int start_column_b = b_positions[row_index];
    int end_column_b = b_positions[row_index + 1];

    for (int col1_index = start_column; col1_index < end_column; ++col1_index)
    {
      float value1 = a_values[col1_index];

      int col1 = a_index[col1_index];

      for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index)
      {
        // std::cout << col1_index << " " << start_column << " " << end_column << " " << col2_index
        // << std::endl;
        int col2 = b_index[col2_index];
        float value2 = b_values[col2_index];

        float_result[col2 * columns + col1] += value1 * value2;
      }
    }

    for (int col1_index = start_column; col1_index < end_column; ++col1_index)
    {
      float value1 = a_values[col1_index];

      int col1 = a_index[col1_index];

      squares_a[col1] += value1 * value1;
    }
    for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index)
    {
      float value2 = b_values[col2_index];
      int col2 = b_index[col2_index];

      squares_b[col2] += value2 * value2;
    }
  }

  for (int i = 0; i < columns * columns_b; ++i)
  {
    int row_index = i / columns;
    int column_index = i % columns;
    result[i] =
      1.0f - float_result[i] / std::sqrt(squares_b[row_index]) / std::sqrt(squares_a[column_index]);
  }

  delete[] (float_result);
  delete[] (a_values);
  delete[] (b_values);
  delete[] (squares_a);
  delete[] (squares_b);
}

extern "C" void matrix_Kendall_sparse_distance_same_block_cpu(
  int* a_index,
  int* a_positions,
  double* a_double_values,
  int* b_index,
  int* b_positions,
  double* b_double_values,
  double* result,
  int* num_rows,
  int* num_columns,
  int* num_columns_b,
  int* num_elements_a,
  int* num_elements_b)
{
  int rows = *num_rows;
  int columns = *num_columns;

  int* disconcordant = new int[columns * columns];
  std::memset(disconcordant, 0, columns * columns * sizeof(int));

  float* a_values = new float[*num_elements_a];
  for (int i = 0; i < *num_elements_a; ++i)
  {
    a_values[i] = static_cast<float>(a_double_values[i]);
  }

  bool* left_thresholds = new bool[columns];


  for (int row_index = 0; row_index < rows; ++row_index)
  {

    int start_column = a_positions[row_index];

    int end_column = a_positions[row_index + 1];
    for (int row_jndex = row_index + 1; row_jndex < rows; ++row_jndex)
    {

      int start_column_b = a_positions[row_jndex];
      int end_column_b = a_positions[row_jndex + 1];

      for (int i = 0; i < end_column_b - start_column_b; ++i)
      {
        left_thresholds[i] = false;
      }
      // bool left_threshold_selected = false;
      // bool right_threshold_selected = false;
      int right_down1_threshold = start_column_b;
      int left_down1_threshold = start_column_b;
      int left_down2_threshold;
      int right_down2_threshold;
      bool left_activated = false;
      bool right_activated = false;

      for (int col1_index = start_column; col1_index < end_column; ++col1_index)
      {
        // int prev_col_index = col1_index - 1;
        // int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
        int col1 = a_index[col1_index];
        // float value1 = a_values[col1_index];


        while (right_down1_threshold < end_column_b && a_index[right_down1_threshold] < col1)
        {
          right_down1_threshold += 1;
        }

        if (right_down1_threshold < end_column_b && a_index[right_down1_threshold] == col1)
        {
          left_activated = true;
        }
        if (right_down1_threshold < end_column_b && a_index[right_down1_threshold] == col1)
        {
          left_down2_threshold = right_down1_threshold + 1;
          right_activated = true;
        }
        else
        {
          left_down2_threshold = right_down1_threshold;
        }

        right_down2_threshold = left_down2_threshold;
        for (int col2_index = col1_index; col2_index < end_column; ++col2_index)
        {
          int col2 = a_index[col2_index];
          // float value2 = a_values[col2_index];
          int next_col_index = col2_index + 1;
          int next_col = (next_col_index < end_column) ? a_index[next_col_index] : columns;

          while (right_down2_threshold < end_column_b && a_index[right_down2_threshold] < next_col)
          {
            right_down2_threshold += 1;
          }
          if (
            left_down1_threshold < end_column_b &&
            !left_thresholds[left_down1_threshold - start_column_b])
          {
            left_thresholds[left_down1_threshold - start_column_b] = true;
            for (int left = left_down1_threshold; left < right_down1_threshold; left++)
            {
              for (int right = left + 1; right < right_down1_threshold; ++right)
              {
                float product = a_values[left] * a_values[right];
                if (product < 0)
                {
                  disconcordant[a_index[left] * columns + a_index[right]] += 1;
                  disconcordant[a_index[right] * columns + a_index[left]] += 1;
                }
              }
            }
          }

          if (
            left_down2_threshold < end_column_b &&
            !left_thresholds[left_down2_threshold - start_column_b])
          {
            left_thresholds[left_down2_threshold - start_column_b] = true;
            for (int left = left_down2_threshold; left < right_down2_threshold; left++)
            {
              for (int right = left + 1; right < right_down2_threshold; ++right)
              {
                float product = a_values[left] * a_values[right];
                if (product < 0)
                {
                  disconcordant[a_index[left] * columns + a_index[right]] += 1;
                  disconcordant[a_index[right] * columns + a_index[left]] += 1;
                }
              }
            }
          }

          for (int left = left_down1_threshold; left < right_down1_threshold; left++)
          {
            for (int right = left_down2_threshold; right < right_down2_threshold; ++right)
            {
              float product = a_values[left] * a_values[right];
              if (product < 0)
              {
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
          if (product < 0)
          {
            disconcordant[col1 * columns + col2] += 1;
            disconcordant[col2 * columns + col1] += 1;
          }

          for (int right = left_down2_threshold; right < right_down2_threshold; ++right)
          {
            product = left_diff * a_values[right];
            if (product < 0)
            {
              disconcordant[col1 * columns + a_index[right]] += 1;
              disconcordant[a_index[right] * columns + col1] += 1;
            }
          }

          for (int left = left_down1_threshold; left < right_down1_threshold; left++)
          {
            product = right_diff * a_values[left];
            if (product < 0)
            {
              disconcordant[a_index[left] * columns + col2] += 1;
              disconcordant[col2 * columns + a_index[left]] += 1;
            }
          }

          right_activated = false;
          while (left_down2_threshold < end_column_b && a_index[left_down2_threshold] <= next_col)
          {
            if (a_index[left_down2_threshold] == next_col)
            {
              right_activated = true;
            }
            left_down2_threshold += 1;
          }
        }

        while (left_down1_threshold < end_column_b && a_index[left_down1_threshold] <= col1)
        {
          left_down1_threshold += 1;
        }
      }
    }
  }
  for (int i = 0; i < columns * columns; ++i)
  {
    result[i] = static_cast<double>(disconcordant[i]) * 2.0f / rows / (rows - 1);
  }

  delete[] disconcordant;
  delete[] left_thresholds;
  delete[] a_values;
}


extern "C" void matrix_Kendall_sparse_distance_different_blocks_cpu(
  int* a_index,
  int* a_positions,
  double* a_double_values,
  int* b_index,
  int* b_positions,
  double* b_double_values,
  double* result,
  int* num_rows,
  int* num_columns,
  int* num_columns_b,
  int* num_elements_a,
  int* num_elements_b)
{

  int rows = *num_rows;
  int columns = *num_columns;
  int columns_b = *num_columns_b;

  int* disconcordant = new int[columns * columns_b];
  std::memset(disconcordant, 0, columns * columns_b * sizeof(int));
  float* a_values = new float[*num_elements_a];
  float* b_values = new float[*num_elements_b];

  for (int i = 0; i < *num_elements_a; ++i)
  {
    a_values[i] = static_cast<float>(a_double_values[i]);
  }

  for (int i = 0; i < *num_elements_b; ++i)
  {
    b_values[i] = static_cast<float>(b_double_values[i]);
  }

  for (int row_index = 0; row_index < rows; ++row_index)
  {
    for (int row_jndex = row_index + 1; row_jndex < rows; ++row_jndex)
    {
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
      // int left_down2_threshold = start_column_down_b;
      // int right_down2_threshold = start_column_down_b;
      bool left_activated = false;
      bool right_activated = false;

      for (int col1_index = start_column; col1_index <= end_column; ++col1_index)
      {
        // int prev_col_index = col1_index - 1;
        // int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
        int col1 = (col1_index < end_column) ? a_index[col1_index] : columns;
        float value1 = (col1_index < end_column) ? a_values[col1_index] : 0;

        while (right_down1_threshold < end_column_down && a_index[right_down1_threshold] < col1)
        {
          right_down1_threshold += 1;
        }

        if (right_down1_threshold < end_column_down && a_index[right_down1_threshold] == col1)
        {
          left_activated = true;
        }

        int left_down2_threshold = start_column_down_b;
        int right_down2_threshold = start_column_down_b;
        right_activated = false;
        for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index)
        {
          int col2 = b_index[col2_index];
          float value2 = b_values[col2_index];
          int next_col_index = col2_index + 1;
          int next_col = (next_col_index < end_column_b) ? b_index[next_col_index] : columns_b;


          while (right_down2_threshold < end_column_down_b &&
                 b_index[right_down2_threshold] < next_col)
          {
            right_down2_threshold += 1;
          }

          for (int left = left_down1_threshold; left < right_down1_threshold; left++)
          {
            for (int right = left_down2_threshold; right < right_down2_threshold; ++right)
            {
              float product = a_values[left] * b_values[right];
              assert(b_index[right] < columns_b);
              if (product < 0)
              {
                disconcordant[b_index[right] * columns + a_index[left]] += 1;
              }
            }
          }

          float left_value = (left_activated) ? a_values[right_down1_threshold] : 0;
          float right_value = (right_activated) ? b_values[left_down2_threshold - 1] : 0;


          float right_diff = right_value - value2;

          float left_diff = left_value - value1;
          float product = left_diff * right_diff;
          if (product < 0)
          {
            disconcordant[col2 * columns + col1] += 1;
          }

          for (int right = left_down2_threshold; right < right_down2_threshold; ++right)
          {
            product = left_diff * b_values[right];
            if (product < 0)
            {
              disconcordant[b_index[right] * columns + col1] += 1;
            }
          }

          for (int left = left_down1_threshold; left < right_down1_threshold; left++)
          {
            product = right_diff * a_values[left];
            if (product < 0)
            {
              disconcordant[col2 * columns + a_index[left]] += 1;
            }
          }

          right_activated = false;
          while (left_down2_threshold < end_column_down_b &&
                 b_index[left_down2_threshold] <= next_col)
          {
            if (b_index[left_down2_threshold] == next_col)
            {
              right_activated = true;
            }
            left_down2_threshold += 1;
          }
        }
        while (left_down1_threshold < end_column_down && a_index[left_down1_threshold] <= col1)
        {
          left_down1_threshold += 1;
        }
      }
    }
  }
  for (int i = 0; i < columns * columns_b; ++i)
  {
    result[i] = static_cast<double>(disconcordant[i]) * 2.0f / rows / (rows - 1);
  }

  delete[] disconcordant;
  delete[] a_values;
  delete[] b_values;
}
