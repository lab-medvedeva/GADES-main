#pragma once

#include <cstddef>
#include <cstdint>

template <typename T>
struct MatrixView
{
  size_t row_num;
  size_t col_num;
  T* data;

  T* operator[](size_t pos) const { return data + row_num * pos; }
};

template <typename T, typename IR, typename IC>
struct MatrixViewCSC
{
  T* vals;
  IR* rows;
  IC* col_offsets;

  size_t col_num;
  size_t row_num;
};

template <typename T, typename I>
struct ColumnViewCSC
{
  T* vals;
  I* rows;
  size_t len;
};
