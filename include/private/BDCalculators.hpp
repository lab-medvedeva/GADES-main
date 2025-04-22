#pragma once

#include "utils.hpp"
#include <cstddef>
#include <cmath>
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

class CosineCalculatorHelper
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
    batch_norms.resize(batch.col_num);

    const double* from = batch.data;
    double* to = aligned_batch.data;

    if (aligned_batch.row_num == batch.row_num)
    {
      copyDataPerfect(from, to, size, lanes);
      for (size_t col = 0; col < batch.col_num; ++col)
      {
        batch_norms[col] = calcNorm(aligned_batch[col]);
      }
      return;
    }

    for (size_t col = 0; col < batch.col_num; ++col)
    {
      copyDataImperfect(from, to, batch.row_num, lanes);
      batch_norms[col] = calcNorm(to);
      from += batch.row_num;
      to += aligned_batch.row_num;
    }
  }

  void setVec(const double* HWY_RESTRICT vec)
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
    vec_norm = calcNorm(vec_data.data());
  }
  double calcDistOuter(size_t i)
  {
    return calcDist(aligned_batch[i], batch_norms[i], vec_data.data(), vec_norm);
  }
  double calcDistInner(size_t i, size_t j)
  {
    return calcDist(aligned_batch[i], batch_norms[i], aligned_batch[j], batch_norms[j]);
  }

private:
  double calcDist(double* HWY_RESTRICT a, double a_norm, double* HWY_RESTRICT b, double b_norm)
  {
    size_t lanes = hn::Lanes(d);
    auto sums = hn::Zero(d);
    for (size_t len = aligned_batch.row_num; len; len -= lanes)
    {
      auto a_curr = hn::Load(d, a);
      auto b_curr = hn::Load(d, b);
      sums = hn::MulAdd(a_curr, b_curr, sums);
      a += lanes;
      b += lanes;
    }
    return hn::ReduceSum(d, sums) / (a_norm * b_norm);
  }
  void copyDataPerfect(
    const double* HWY_RESTRICT from, double* HWY_RESTRICT to, size_t len, size_t lanes)
  {
    while (len)
    {
      hn::Store(hn::LoadU(d, from), d, to);
      from += lanes;
      to += lanes;
      len -= lanes;
    }
  }
  void copyDataImperfect(
    const double* HWY_RESTRICT from, double* HWY_RESTRICT to, size_t len, size_t lanes)
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

  double calcNorm(double* HWY_RESTRICT vec)
  {
    size_t lanes = hn::Lanes(d);
    size_t len = aligned_batch.row_num;
    auto res = hn::Zero(d);
    while (len)
    {
      auto curr = hn::Load(d, vec);
      res = hn::MulAdd(curr, curr, res);
      len -= lanes;
      vec += lanes;
    }
    return std::sqrt(hn::ReduceSum(d, res));
  }

  static HWY_FULL(double) d; // HWY tag for function overloading

  std::vector<double, hwy::AlignedAllocator<double>> batch_data;
  MatrixView<double> aligned_batch;
  std::vector<double> batch_norms;

  std::vector<double, hwy::AlignedAllocator<double>> vec_data;
  double vec_norm;

  size_t batch_row_num;
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

static uint32_t RoundToPow2(uint32_t val)
{
  val |= val >> 1;
  val |= val >> 2;
  val |= val >> 4;
  val |= val >> 8;
  val |= val >> 16;
  val += 1;
  return val;
}

static uint64_t RoundToPow2(uint64_t val)
{
  if ((val & (val - 1)) == 0)
  {
    return val;
  }
  val |= val >> 1;
  val |= val >> 2;
  val |= val >> 4;
  val |= val >> 8;
  val |= val >> 16;
  val |= val >> 32;
  val += 1;
  return val;
}

class SpearmanCalculatorHelper
{
  static HWY_FULL(double) d;

  // Overload tag for indices
  using ID = hn::RebindToSigned<decltype(d)>;
  static ID id;
  using IndexType = hn::TFromD<ID>;

  static HWY_FULL(float) fd;

public:
  void startCalculation(MatrixView<const double> batch)
  {
    setBatch(batch);
    batch_size = batch.col_num;
  }

  void calcOuter(auto& write_view, auto& vec_getter, size_t outer_col_count)
  {
    size_t lanes = hn::Lanes(d);
    const double* ptrs[hn::MaxLanes(d)];
    size_t len = lane::RoundDownTo(outer_col_count, lanes);
    for (size_t i = 0; i < len; i += lanes)
    {
      for (size_t k = 0; k < lanes; ++k)
      {
        ptrs[k] = vec_getter();
      }
      setVec(ptrs);
      for (size_t k = 0; k < lanes; ++k)
      {
        for (size_t j = 0; j < batch_size; ++j)
        {
          write_view(i + k, j) = calcDistOuter(j, k);
        }
      }
    }
    size_t offset = lane::Mod(outer_col_count, lanes);
    if (offset == 0)
    {
      return;
    }
    for (size_t k = 0; k < offset; ++k)
    {
      ptrs[k] = vec_getter();
    }
    for (size_t k = offset; k < lanes; ++k)
    {
      ptrs[k] = ptrs[k - 1];
    }
    setVec(ptrs);
    for (size_t k = 0; k < offset; ++k)
    {
      for (size_t j = 0; j < batch_size; ++j)
      {
        write_view(outer_col_count - offset + k, j) = calcDistOuter(j, k);
      }
    }
  }

  void calcInner(auto& write_view)
  {
    for (size_t i = 1; i < batch_size; ++i)
    {
      for (size_t j = 0; j < i; ++j)
      {
        write_view(i, j) = calcDistInner(i, j);
      }
    }
  }

private:
  void setBatch(MatrixView<const double> batch)
  {
    row_num = batch.row_num;
    const size_t lanes = hn::Lanes(d);
    const size_t lanes_f = hn::Lanes(fd);
    aligned_batch.col_num = batch.col_num;
    aligned_batch.row_num = lane::RoundUpTo(row_num, lanes_f);
    batch_data.resize(aligned_batch.col_num * aligned_batch.row_num, 0.0);
    aligned_batch.data = batch_data.data();

    size_t row_num_rounded = RoundToPow2(batch.row_num * lanes);
    val_staging_data.resize(row_num_rounded, std::numeric_limits<double>::infinity());
    ind_staging_data.resize(row_num_rounded, 0);

    vec_data.resize(lanes * aligned_batch.row_num);
    vec_batch.row_num = aligned_batch.row_num;
    vec_batch.col_num = lanes;
    vec_batch.data = vec_data.data();

    size_t len = lane::RoundDownTo(batch.col_num, lanes);

    for (size_t i = 0; i < len; i += lanes)
    {
      setData(batch, i);
    }
    size_t rem = lane::Mod(batch.col_num, lanes);
    if (rem != 0)
    {
      setDataPartial(batch, batch.col_num - rem, rem);
    }
  }

  void setData(MatrixView<const double> batch, size_t pos)
  {
    size_t lanes = hn::Lanes(d);
    {
      const double* ptrs[hn::MaxLanes(d)];
      for (size_t i = 0; i < lanes; ++i)
      {
        ptrs[i] = batch[pos + i];
      }
      copyInterleave(ptrs, val_staging_data.data());
    }
    setIndices(ind_staging_data.data());
    sortInterleaved();
    {
      float* ptrs[hn::MaxLanes(d)];
      for (size_t i = 0; i < lanes; ++i)
      {
        ptrs[i] = aligned_batch[pos + i];
      }
      setValues(ptrs, ind_staging_data.data());
    }
  }

  void setDataPartial(MatrixView<const double> batch, size_t pos, size_t len)
  {
    {
      const double* ptrs[hn::MaxLanes(d)];
      for (size_t i = 0; i < len; ++i)
      {
        ptrs[i] = batch[pos + i];
      }
      copyInterleavePartial(ptrs, val_staging_data.data(), len);
    }
    setIndices(ind_staging_data.data());
    sortInterleaved();
    {
      float* ptrs[hn::MaxLanes(d)];
      for (size_t i = 0; i < len; ++i)
      {
        ptrs[i] = aligned_batch[pos + i];
      }
      setValuesPartial(ptrs, ind_staging_data.data(), len);
    }
  }

  void copyInterleave(const double** HWY_RESTRICT from, double* HWY_RESTRICT to)
  {
    const size_t lanes = hn::Lanes(d);
    for (size_t i = 0; i < row_num; ++i)
    {
      for (size_t j = 0; j < lanes; ++j)
      {
        *(to++) = *(from[j]++);
      }
    }
  }

  void copyInterleavePartial(const double** HWY_RESTRICT from, double* HWY_RESTRICT to, size_t len)
  {
    const size_t lanes = hn::Lanes(d);
    const size_t rem = lanes - len;
    for (size_t i = 0; i < row_num; ++i)
    {
      for (size_t j = 0; j < len; ++j)
      {
        *(to++) = *(from[j]++);
      }
      to += rem;
    }
  }

  void setIndices(IndexType* to)
  {
    const size_t lanes = Lanes(id);
    auto step = hn::Set(id, (IndexType)1);
    for (auto i = hn::Zero(id); hn::GetLane(i) < row_num; i = hn::Add(i, step))
    {
      hn::Store(i, id, to);
      to += lanes;
    }
  }

  void sortInterleaved()
  {
    size_t pow = lane::getPow(Lanes(d)); // todo
    size_t row_num_padded = val_staging_data.size() >> pow;
    double* vals = val_staging_data.data();
    IndexType* indices = ind_staging_data.data();
    for (size_t blk = 2; blk <= row_num_padded; blk *= 2)
    {
      for (size_t step = blk / 2; step > 0; step /= 2)
      {
        for (size_t i = 0; i < row_num_padded; ++i)
        {
          if (i & step)
          {
            i += step - 1;
            continue;
          }
          size_t i_a = i << pow;
          size_t i_b = (i ^ step) << pow;
          auto a = hn::Load(d, vals + i_a);
          auto b = hn::Load(d, vals + i_b);
          // Should swap?
          auto mask = hn::Gt(a, b);
          if ((i & blk) != 0)
          {
            mask = hn::Not(mask);
          }
          // vals[i_a] = should_swap ? b : a
          hn::Store(IfThenElse(mask, b, a), d, vals + i_a);
          // vals[i_b] = should_swap ? a : b
          hn::Store(IfThenElse(mask, a, b), d, vals + i_b);

          auto a_ind = hn::Load(id, indices + i_a);
          auto b_ind = hn::Load(id, indices + i_b);
          auto mask_ind = hn::RebindMask(id, mask);

          hn::Store(IfThenElse(mask_ind, b_ind, a_ind), id, indices + i_a);
          hn::Store(IfThenElse(mask_ind, a_ind, b_ind), id, indices + i_b);
        }
      }
    }
  }

  void setValues(float** HWY_RESTRICT to, IndexType* HWY_RESTRICT from)
  {
    size_t lanes = hn::Lanes(d);
    for (uint32_t i = 0; i < row_num; ++i)
    {
      for (size_t j = 0; j < lanes; ++j)
      {
        to[j][*(from++)] = static_cast<float>(i);
      }
    }
  }
  void setValuesPartial(float** HWY_RESTRICT to, IndexType* HWY_RESTRICT from, size_t len)
  {
    const size_t lanes = hn::Lanes(d);
    const size_t rem = lanes - len;
    for (uint32_t i = 0; i < row_num; ++i)
    {
      for (size_t j = 0; j < len; ++j)
      {
        to[j][*(from++)] = static_cast<float>(i);
      }
      from += rem;
    }
  }

  void setVec(const double** from)
  {
    size_t lanes = hn::Lanes(d);
    copyInterleave(from, val_staging_data.data());
    setIndices(ind_staging_data.data());
    sortInterleaved();
    {
      float* ptrs[hn::MaxLanes(d)];
      for (size_t i = 0; i < lanes; ++i)
      {
        ptrs[i] = vec_batch[i];
      }
      setValues(ptrs, ind_staging_data.data());
    }
  }

  double calcDistInner(size_t i, size_t j) { return calcDist(aligned_batch[i], aligned_batch[j]); }
  double calcDistOuter(size_t i, size_t vec_pos)
  {
    return calcDist(aligned_batch[i], vec_batch[vec_pos]);
  }
  double calcDist(float* HWY_RESTRICT a, float* HWY_RESTRICT b)
  {
    size_t lanes = hn::Lanes(fd);
    auto res_vec = hn::Zero(d);
    for (size_t len = aligned_batch.row_num; len > 0; len -= lanes)
    {
      auto a_val = hn::Load(fd, a);
      auto b_val = hn::Load(fd, b);
      auto diff = hn::Sub(a_val, b_val);
      auto curr_d = hn::PromoteLowerTo(d, diff);
      res_vec = hn::MulAdd(curr_d, curr_d, res_vec);
      curr_d = hn::PromoteUpperTo(d, diff);
      res_vec = hn::MulAdd(curr_d, curr_d, res_vec);
      a += lanes;
      b += lanes;
    }
    double res = hn::ReduceSum(d, res_vec);
    double n = static_cast<double>(row_num);
    double coef = 6.0 / (n * (n * n - 1));
    res = 1.0 - coef * res;
    return res;
  }


  size_t row_num;

  std::vector<double, hwy::AlignedAllocator<double>> val_staging_data;
  std::vector<IndexType, hwy::AlignedAllocator<IndexType>> ind_staging_data;

  std::vector<float, hwy::AlignedAllocator<float>> batch_data;
  MatrixView<float> aligned_batch;

  std::vector<float, hwy::AlignedAllocator<float>> vec_data;
  MatrixView<float> vec_batch;

  size_t outer_col_count;
  size_t batch_size;
};

} // namespace detail


using BDCalculatorL1 = detail::BDCalculatorAdaptor<detail::L1CalculatorHelper>;
using BDCalculatorCosine = detail::BDCalculatorAdaptor<detail::CosineCalculatorHelper>;
using BDCalculatorSpearman = detail::SpearmanCalculatorHelper;
