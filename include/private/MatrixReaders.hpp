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

class BasicSparseMatrixReader
{
public:
  using SMView = MatrixViewCSC<const double, const int64_t, const uint32_t>;
  using SCView = ColumnViewCSC<const double, const int64_t>;
  BasicSparseMatrixReader(SMView view)
    : matrix(view)
  {
  }

  auto getBatch(size_t begin, size_t size)
  {
    SMView batch = {
      .vals = matrix.vals,
      .rows = matrix.rows,
      .col_offsets = matrix.col_offsets + begin,
      .col_num = size,
      .row_num = matrix.row_num};
    return batch;
  }

  auto getReadFunc(size_t begin)
  {
    const uint32_t* col_ptr = matrix.col_offsets + begin;

    auto func = [this, col_ptr]() mutable {
      SCView col{
        .vals = matrix.vals + col_ptr[0],
        .rows = matrix.rows + col_ptr[0],
        .len = col_ptr[1] - col_ptr[0],
      };
      ++col_ptr;
      return col;
    };
    return func;
  }

private:
  SMView matrix;
};

class BasicSparseMatrixReaderNarrow
{
public:
  using SMView = MatrixViewCSC<const double, const int32_t, const uint32_t>;
  using SCView = ColumnViewCSC<const double, const int32_t>;
  BasicSparseMatrixReaderNarrow(SMView view)
    : matrix(view)
  {
  }

  auto getBatch(size_t begin, size_t size)
  {
    SMView batch = {
                    .vals = matrix.vals,
                    .rows = matrix.rows,
                    .col_offsets = matrix.col_offsets + begin,
                    .col_num = size,
                    .row_num = matrix.row_num};
    return batch;
  }

  auto getReadFunc(size_t begin)
  {
    const uint32_t* col_ptr = matrix.col_offsets + begin;

    auto func = [this, col_ptr]() mutable {
      SCView col{
        .vals = matrix.vals + col_ptr[0],
        .rows = matrix.rows + col_ptr[0],
        .len = col_ptr[1] - col_ptr[0],
      };
      ++col_ptr;
      return col;
    };
    return func;
  }

private:
  SMView matrix;
};
