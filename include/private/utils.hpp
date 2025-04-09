#pragma once

#include <cstddef>
#include <type_traits>
#include <concepts>

template <typename T>
struct MatrixView
{
  size_t row_num;
  size_t col_num;
  T* data;

  T* operator[](size_t pos) const { return data + row_num * pos; }

};
