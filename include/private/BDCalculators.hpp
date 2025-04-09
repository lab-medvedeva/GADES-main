#pragma once

#include "utils.hpp"
#include <span>
#include <cstddef>
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "lane_funcs.hpp"

namespace detail
{

namespace hn = hwy::HWY_NAMESPACE;

template <typename InnerCalculator>
class BDCalculatorAdaptor
{
public:
  void startCalculation(MatrixView<const double> batch)
  {
    calc.setBatch(batch);
    batch_size = batch.col_num;
  }

  void calcOuter(auto& write_view, auto& vec_getter, size_t outer_col_count)
  {
    for (size_t i = 0; i < outer_col_count; ++i)
    {
      calc.setVec(vec_getter());
      for (size_t j = 0; j < batch_size; ++j)
      {
        write_view(i, j) = calc.calcDistOuter(j);
      }
    }
  }

  void calcInner(auto& write_view)
  {
    for (size_t i = 1; i < batch_size; ++i)
    {
      for (size_t j = 0; j < i; ++j)
      {
        write_view(i, j) = calc.calcDistInner(i, j);
      }
    }
  }

private:
  InnerCalculator calc{};
  size_t outer_col_count;
  size_t batch_size;
};

class L1CalculatorHelper
{
public:
  void setBatch(MatrixView<const double> batch)
  {
    batch_row_num = batch.row_num;
    size_t lanes = hn::Lanes(d);
    aligned_batch.col_num = batch.col_num;

    aligned_batch.row_num = lane::RoundUpTo(batch.row_num, lanes);
    vec_data.resize(aligned_batch.row_num, 0.0);
    size_t size = aligned_batch.row_num * aligned_batch.col_num;
    batch_data.resize(size, 0.0);
    aligned_batch.data = batch_data.data();

    const double* from = batch.data;
    double* to = aligned_batch.data;

    if (aligned_batch.row_num == batch.row_num)
    {
      copyDataPerfect(from, to, size, lanes);
      return;
    }

    for (size_t col = 0; col < batch.col_num; ++col)
    {
      copyDataImperfect(from, to, batch.row_num, lanes);
      from += batch.row_num;
      to += aligned_batch.row_num;
    }
  }
  void setVec(const double* vec)
  {
    size_t lanes = hn::Lanes(d);
    if (batch_row_num == aligned_batch.row_num)
    {
      copyDataPerfect(vec, vec_data.data(), batch_row_num, lanes);
    }
    else
    {
      copyDataImperfect(vec, vec_data.data(), batch_row_num, lanes);
    }
  }
  double calcDistOuter(size_t i) { return calcDist(aligned_batch[i], vec_data.data()); }
  double calcDistInner(size_t i, size_t j) { return calcDist(aligned_batch[i], aligned_batch[j]); }

private:
  double calcDist(double* HWY_RESTRICT a, double* HWY_RESTRICT b)
  {
    size_t lanes = hn::Lanes(d);
    auto sums = hn::Zero(d);
    for (size_t len = aligned_batch.row_num; len; len -= lanes)
    {
      auto a_curr = hn::Load(d, a);
      auto b_curr = hn::Load(d, b);
      sums = hn::Add(hn::AbsDiff(a_curr, b_curr), sums);
      a += lanes;
      b += lanes;
    }
    return hn::ReduceSum(d, sums);
  }
  void copyDataPerfect(const double* from, double* to, size_t len, size_t lanes)
  {
    while (len)
    {
      hn::Store(hn::LoadU(d, from), d, to);
      from += lanes;
      to += lanes;
      len -= lanes;
    }
  }
  void copyDataImperfect(const double* from, double* to, size_t len, size_t lanes)
  {
    while (len >= lanes)
    {
      hn::Store(hn::LoadU(d, from), d, to);
      from += lanes;
      to += lanes;
      len -= lanes;
    }
    std::memcpy(to, from, len * sizeof(double));
  }
  static HWY_FULL(double) d; // HWY tag for function overloading
  std::vector<double, hwy::AlignedAllocator<double>> batch_data;
  MatrixView<double> aligned_batch;
  std::vector<double, hwy::AlignedAllocator<double>> vec_data;
  size_t batch_row_num;
};

} // namespace detail


using BDCalculatorL1 = detail::BDCalculatorAdaptor<detail::L1CalculatorHelper>;
