#pragma once

#include "utils.hpp"

class BasicMatrixWriter
{
public:
  BasicMatrixWriter(MatrixView<double> view)
    : matrix(view)
  {
  }

  void beginWrite(size_t offset, size_t batch_size)
  {
    this->offset = offset;
    this->batch_size = batch_size;
  }

  auto getInnerView()
  {
    MatrixView res = matrix;
    res.data += offset * (matrix.row_num + 1);
    return [res](size_t i, size_t j) -> double& { return res[i][j]; };
  }

  void finishInner(auto& view)
  {
    for (size_t i = 0; i < batch_size; ++i)
    {
      view(i, i) = 0.0;
    }
    for (size_t i = 1; i < batch_size; ++i)
    {
      for (size_t j = 0; j < i; ++j)
      {
        view(j, i) = view(i, j);
      }
    }
  }

  auto getOuterView()
  {
    MatrixView res = matrix;
    res.data += offset + matrix.row_num * (batch_size + offset);
    return [res](size_t i, size_t j) -> double& { return res[i][j]; };
  }
  auto finishOuter(auto&)
  {
    for (size_t i = offset + batch_size; i < matrix.col_num; ++i)
    {
      for (size_t j = offset; j < offset + batch_size; ++j)
      {
        matrix[j][i] = matrix[i][j];
      }
    }
  }

private:
  MatrixView<double> matrix;
  size_t offset;
  size_t batch_size;
};
