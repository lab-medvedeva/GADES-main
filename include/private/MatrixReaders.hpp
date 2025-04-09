#pragma once

#include "utils.hpp"

class BasicMatrixReader
{
public:
  BasicMatrixReader(MatrixView<const double> view)
    : matrix(view)
  {
  }

  auto getBatch(size_t begin, size_t size)
  {
    MatrixView<const double> batch = {
      .row_num = matrix.row_num, .col_num = size, .data = matrix[begin]};
    return batch;
  }

  auto getReadFunc(size_t begin)
  {
    size_t stride = matrix.row_num;
    const double* ptr = matrix.data + stride * begin - stride;
    auto func = [ptr, stride]() mutable { return (ptr += stride); };
    return func;
  }

private:
  MatrixView<const double> matrix;
};
