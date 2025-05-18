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

struct SparseBDL1Calc
{
  static constexpr size_t SLICE_SIZE = 4096;
  static constexpr HWY_FULL(double) d{};
  static constexpr HWY_FULL(int64_t) id{};

  size_t batch_col_num;
  size_t batch_aligned;
  size_t slice_num;
  size_t row_num;

  static auto alignPtrUp(auto ptr)
  {
    const size_t lanes = hn::Lanes(d);
    size_t offset = ptr - (decltype(ptr))nullptr;
    size_t add = lane::RoundUpTo(offset, lanes) - offset;
    return ptr + add;
  }

  static auto alignPtrDown(auto ptr)
  {
    const size_t lanes = hn::Lanes(d);
    size_t offset = ptr - (decltype(ptr))nullptr;
    size_t sub = offset - lane::RoundDownTo(offset, lanes);
    return ptr - sub;
  }

  // Slice_i = Batch[i * SLICE_SIZE : (i + 1) * SLICE_SIZE]

  std::vector<double, hwy::AlignedAllocator<double>> aligned_sliced_vals_vector;
  std::vector<int64_t, hwy::AlignedAllocator<int64_t>> aligned_sliced_rows_vector;
  std::vector<uint32_t, hwy::AlignedAllocator<uint32_t>> sliced_col_offsets_vector;

  std::vector<double, hwy::AlignedAllocator<double>> aligned_tmp_res_vector;
  std::vector<uint32_t, hwy::AlignedAllocator<uint32_t>> tmp_ptrs_vector;
  std::vector<double, hwy::AlignedAllocator<double>> aligned_sliced_norms;

  HWY_ALIGN double dense_slice[SLICE_SIZE + 1] = {};

  void startCalculation(MatrixViewCSC<const double, const int64_t, const uint32_t> batch)
  {
    allocate(batch.col_num, batch.row_num, batch.col_offsets[batch.col_num] - batch.col_offsets[0]);

    double* val_out = aligned_sliced_vals_vector.data();
    int64_t* row_out = aligned_sliced_rows_vector.data();
    uint32_t* col_out = sliced_col_offsets_vector.data();
    *col_out = 0;
    ++col_out;

    memcpy(tmp_ptrs_vector.data(), batch.col_offsets, batch_col_num * sizeof(uint32_t));

    for (size_t curr_slice = 0; curr_slice != slice_num; ++curr_slice)
    {
      // *ptr is in [*batch.col_offsets, *batch.col_offsets+1), i.e. the current row
      uint32_t* HWY_RESTRICT ptr = tmp_ptrs_vector.data();
      uint32_t* HWY_RESTRICT end = ptr + batch_col_num;
      const uint32_t* HWY_RESTRICT row_end_ptr = batch.col_offsets + 1;
      for (; ptr != end; ++ptr, ++col_out, ++row_end_ptr)
      {
        const int64_t* HWY_RESTRICT row_ptr = batch.rows + *ptr;
        const int64_t* HWY_RESTRICT row_end = batch.rows + *row_end_ptr;
        const double* HWY_RESTRICT val_ptr = batch.vals + *ptr;

        while (row_ptr != row_end && *row_ptr / SLICE_SIZE == curr_slice)
        {
          *(val_out++) = *(val_ptr++);
          *(row_out++) = *(row_ptr++) % SLICE_SIZE;
        }
        int64_t* HWY_RESTRICT row_out_nxt = alignPtrUp(row_out);
        *col_out = row_out_nxt - aligned_sliced_rows_vector.data();
        *ptr = row_ptr - batch.rows;

        std::fill(row_out, row_out_nxt, SLICE_SIZE);
        std::memset(val_out, 0, (row_out_nxt - row_out) * sizeof(double));
        val_out += row_out_nxt - row_out;
        row_out = row_out_nxt;
      }
    }
    calcNorms();
  }

  void calcNorms()
  {
    const size_t lanes = hn::Lanes(d);
    {
      const HWY_RESTRICT uint32_t* col_ptr = sliced_col_offsets_vector.data();
      const HWY_RESTRICT uint32_t* col_end = col_ptr + batch_col_num * slice_num;
      double* HWY_RESTRICT out = aligned_sliced_norms.data() + batch_aligned;
      std::memset(aligned_sliced_norms.data(), 0, batch_aligned * sizeof(double));
      const size_t diff = batch_aligned - batch_col_num;
      while (col_ptr != col_end)
      {
        double* HWY_RESTRICT out_end = out + batch_col_num;
        for (; out != out_end; ++col_ptr, ++out)
        {
          const double* HWY_RESTRICT val_ptr = aligned_sliced_vals_vector.data() + col_ptr[0];
          const double* HWY_RESTRICT val_end = aligned_sliced_vals_vector.data() + col_ptr[1];
          auto res = hn::Zero(d);
          for (; val_ptr != val_end; val_ptr += lanes)
          {
            res = hn::Add(res, hn::Abs(hn::Load(d, val_ptr)));
          }
          *out = hn::ReduceSum(d, res);
        }
        out += diff;
      }
    }
    double* HWY_RESTRICT ptr = aligned_sliced_norms.data() + batch_aligned * 2;
    double* HWY_RESTRICT end = aligned_sliced_norms.data() + batch_aligned * (slice_num + 1);
    for (; ptr != end; ptr += lanes)
    {
      auto curr = hn::Load(d, ptr);
      auto prev = hn::Load(d, ptr - batch_aligned);
      hn::Store(hn::Add(curr, prev), d, ptr);
    }
  }

  void allocate(size_t col_num, size_t row_num, size_t nnz)
  {
    batch_col_num = col_num;
    this->row_num = row_num;
    slice_num = (row_num + SLICE_SIZE - 1) / SLICE_SIZE;
    const size_t lanes = hn::Lanes(d);
    batch_aligned = lane::RoundUpTo(batch_col_num, lanes);

    size_t bound1 = col_num * slice_num * (lanes - 1) + nnz;
    size_t bound2 = nnz * lanes;
    size_t aligned_val_bound = std::min(bound1, bound2);
    aligned_val_bound = std::max(aligned_val_bound, aligned_sliced_vals_vector.size());
    aligned_sliced_vals_vector.resize(aligned_val_bound);
    aligned_sliced_rows_vector.resize(aligned_val_bound);
    sliced_col_offsets_vector.resize(
      std::max(sliced_col_offsets_vector.size(), batch_col_num * slice_num + 1));
    aligned_tmp_res_vector.resize(std::max(aligned_tmp_res_vector.size(), batch_aligned));
    tmp_ptrs_vector.resize(std::max(tmp_ptrs_vector.size(), batch_col_num));
    aligned_sliced_norms.resize(
      std::max(aligned_sliced_norms.size(), batch_aligned * (slice_num + 1)));
  }

  void calcInner(auto& write_view)
  {
    for (size_t i = 1; i < batch_col_num; ++i)
    {
      calcDistI(i);
      for (size_t j = 0; j < i; ++j)
      {
        write_view(i, j) = aligned_tmp_res_vector[j];
      }
    }
  }

  void calcDistI(size_t col_pos)
  {
    const size_t lanes = hn::Lanes(d);
    setATmpRes(aligned_sliced_norms[col_pos + batch_aligned * slice_num], col_pos);
    uint32_t* HWY_RESTRICT col_ptr = sliced_col_offsets_vector.data();
    uint32_t* HWY_RESTRICT col_end = sliced_col_offsets_vector.data() + batch_col_num * slice_num;
    for (; col_ptr != col_end; col_ptr += batch_col_num)
    {
      // Set dense_slice
      {
        double* HWY_RESTRICT val_ptr = aligned_sliced_vals_vector.data() + col_ptr[col_pos];
        int64_t* HWY_RESTRICT row_ptr = aligned_sliced_rows_vector.data() + col_ptr[col_pos];
        int64_t* HWY_RESTRICT row_end = aligned_sliced_rows_vector.data() + col_ptr[col_pos + 1];
        for (; row_ptr != row_end; val_ptr += lanes, row_ptr += lanes)
        {
          auto val = hn::Load(d, val_ptr);
          auto row = hn::Load(id, row_ptr);
          hn::ScatterIndex(val, d, dense_slice, row);
        }
      }
      calcDistSStoDS(MatrixViewCSC<const double, const int64_t, const uint32_t>{
        .vals = aligned_sliced_vals_vector.data(),
        .rows = aligned_sliced_rows_vector.data(),
        .col_offsets = col_ptr,
        .col_num = col_pos,
      });
      {
        int64_t* HWY_RESTRICT row_ptr = aligned_sliced_rows_vector.data() + col_ptr[col_pos];
        int64_t* HWY_RESTRICT row_end = aligned_sliced_rows_vector.data() + col_ptr[col_pos + 1];
        if (row_end - row_ptr >= SLICE_SIZE / lanes)
        {
          resetDenseSlice();
        }
        else
        {
          auto zero = hn::Zero(d);
          for (; row_ptr != row_end; row_ptr += lanes)
          {
            hn::ScatterIndex(zero, d, dense_slice, hn::Load(id, row_ptr));
          }
        }
      }
    }
  }

  void calcOuter(auto& write_view, auto& vec_getter, size_t outer_col_count)
  {
    for (size_t i = 0; i < outer_col_count; ++i)
    {
      calcDistOToB(vec_getter());
      for (size_t j = 0; j < batch_col_num; ++j)
      {
        write_view(i, j) = aligned_tmp_res_vector[j];
      }
    }
  }

  // Outer vector to batch
  void calcDistOToB(ColumnViewCSC<const double, const int64_t> ovec)
  {
    setATmpRes(calcSVecNorm(ovec), batch_col_num);
    const double* HWY_RESTRICT val_ptr = ovec.vals;
    const int64_t* HWY_RESTRICT row_ptr = ovec.rows;
    const int64_t* HWY_RESTRICT end_ptr = ovec.rows + ovec.len;
    size_t prev_slice = 0;
    while (row_ptr != end_ptr)
    {
      const int64_t* HWY_RESTRICT old = row_ptr;
      size_t curr_slice = *row_ptr / SLICE_SIZE;
      addPartialNorms(prev_slice, curr_slice);
      prev_slice = curr_slice + 1;

      for (; (row_ptr != end_ptr) && ((*row_ptr / SLICE_SIZE) == curr_slice); ++row_ptr, ++val_ptr)
      {
        dense_slice[*row_ptr % SLICE_SIZE] = *val_ptr;
      }

      calcDistSStoDS(MatrixViewCSC<const double, const int64_t, const uint32_t>{
        .vals = aligned_sliced_vals_vector.data(),
        .rows = aligned_sliced_rows_vector.data(),
        .col_offsets = sliced_col_offsets_vector.data() + curr_slice * batch_col_num,
        .col_num = batch_col_num,
      });
      // If vector is dense, reset entire slice, otherwise, only necessary cells
      if (row_ptr - old >= SLICE_SIZE / hn::Lanes(d))
      {
        resetDenseSlice();
      }
      else
      {
        for (; old != row_ptr; ++old)
        {
          dense_slice[*old % SLICE_SIZE] = 0.0;
        }
      }
    }
    addPartialNorms(prev_slice, slice_num);
  }

  void addPartialNorms(size_t from, size_t to)
  {
    if (from == to)
    {
      return;
    }
    const size_t lanes = hn::Lanes(d);
    const double* HWY_RESTRICT norm_ptr_a = aligned_sliced_norms.data() + batch_aligned * from;
    const double* HWY_RESTRICT norm_ptr_b = aligned_sliced_norms.data() + batch_aligned * to;
    const double* HWY_RESTRICT norm_end_a = norm_ptr_a + batch_aligned;
    double* HWY_RESTRICT out = aligned_tmp_res_vector.data();
    for (; norm_ptr_a != norm_end_a; norm_ptr_a += lanes, norm_ptr_b += lanes, out += lanes)
    {
      auto a = hn::Load(d, norm_ptr_a);
      auto b = hn::Load(d, norm_ptr_b);
      auto res = hn::Load(d, out);
      res = hn::Add(res, hn::Sub(b, a));
      hn::Store(res, d, out);
    }
  }

  double calcSVecNorm(ColumnViewCSC<const double, const int64_t> svec)
  {
    const double* HWY_RESTRICT ptr = svec.vals;
    const double* HWY_RESTRICT end = ptr + svec.len;

    double res = 0.0;
    // I expect the compiler to optimize this
    for (; ptr != end; ++ptr)
    {
      res += std::abs(*ptr);
    }
    return res;
  }

  // resets aligned_tmp_res_vector.data()
  void setATmpRes(double val, size_t len)
  {
    const size_t lanes = hn::Lanes(d);
    double* HWY_RESTRICT ptr = aligned_tmp_res_vector.data();
    double* HWY_RESTRICT end = ptr + lane::RoundUpTo(len, lanes);
    auto val_v = hn::Set(d, val);
    for (; ptr != end; ptr += lanes)
    {
      hn::Store(val_v, d, ptr);
    }
  }

  void resetDenseSlice()
  {
    const size_t lanes = hn::Lanes(d);
    double* HWY_RESTRICT ptr = dense_slice;
    double* HWY_RESTRICT end = ptr + SLICE_SIZE;
    auto zero = hn::Zero(d);
    for (; ptr != end; ptr += lanes)
    {
      hn::Store(zero, d, ptr);
    }
  }

  // Aligned sparse slice to dense slice
  void calcDistSStoDS(MatrixViewCSC<const double, const int64_t, const uint32_t> s_slice)
  {
    const uint32_t* HWY_RESTRICT col_offset_ptr = s_slice.col_offsets;
    // col_offset_ptr[0] isn't guaranteed to be zero
    const double* HWY_RESTRICT val_ptr = s_slice.vals + *col_offset_ptr;
    const int64_t* HWY_RESTRICT row_ptr = s_slice.rows + *col_offset_ptr;
    double* HWY_RESTRICT res_ptr = aligned_tmp_res_vector.data();
    double* HWY_RESTRICT end_ptr = aligned_tmp_res_vector.data() + s_slice.col_num;

    for (; res_ptr != end_ptr; ++col_offset_ptr, ++res_ptr)
    {
      size_t curr_col_len = col_offset_ptr[1] - col_offset_ptr[0];
      if (curr_col_len == 0)
      {
        continue;
      }
      *res_ptr += calcDistCToDS(ColumnViewCSC<const double, const int64_t>{
        .vals = val_ptr, .rows = row_ptr, .len = curr_col_len});
      val_ptr += curr_col_len;
      row_ptr += curr_col_len;
    }
  }

  // Aligned sparse column to dense slice
  double calcDistCToDS(ColumnViewCSC<const double, const int64_t> col)
  {
    size_t lanes = hn::Lanes(d);
    const double* HWY_RESTRICT ptr = col.vals;
    const double* HWY_RESTRICT end = ptr + col.len;
    const int64_t* HWY_RESTRICT rows = col.rows;
    auto res = hn::Zero(d);
    for (; ptr != end; ptr += lanes, rows += lanes)
    {
      auto ind = hn::Load(id, rows);
      auto a = hn::Load(d, ptr);
      auto b = hn::GatherIndex(d, dense_slice, ind);
      res = hn::Add(res, hn::AbsDiff(a, b));
      res = hn::Sub(res, hn::Abs(b));
    }
    return hn::ReduceSum(d, res);
  }
};

struct SparseBDCosineCalc
{
  static constexpr size_t SLICE_SIZE = 4096;
  static constexpr HWY_FULL(double) d{};
  static constexpr HWY_FULL(int64_t) id{};

  size_t batch_col_num;
  size_t batch_aligned;
  size_t slice_num;
  size_t row_num;

  static auto alignPtrUp(auto ptr)
  {
    const size_t lanes = hn::Lanes(d);
    size_t offset = ptr - (decltype(ptr))nullptr;
    size_t add = lane::RoundUpTo(offset, lanes) - offset;
    return ptr + add;
  }

  static auto alignPtrDown(auto ptr)
  {
    const size_t lanes = hn::Lanes(d);
    size_t offset = ptr - (decltype(ptr))nullptr;
    size_t sub = offset - lane::RoundDownTo(offset, lanes);
    return ptr - sub;
  }

  std::vector<double, hwy::AlignedAllocator<double>> aligned_sliced_vals_vector;
  std::vector<int64_t, hwy::AlignedAllocator<int64_t>> aligned_sliced_rows_vector;
  std::vector<uint32_t, hwy::AlignedAllocator<uint32_t>> sliced_col_offsets_vector;

  std::vector<double, hwy::AlignedAllocator<double>> aligned_tmp_res_vector;
  std::vector<uint32_t, hwy::AlignedAllocator<uint32_t>> tmp_ptrs_vector;
  std::vector<double, hwy::AlignedAllocator<double>> aligned_norms;

  HWY_ALIGN double dense_slice[SLICE_SIZE + 1] = {};

  void startCalculation(MatrixViewCSC<const double, const int64_t, const uint32_t> batch)
  {
    allocate(batch.col_num, batch.row_num, batch.col_offsets[batch.col_num] - batch.col_offsets[0]);

    double* val_out = aligned_sliced_vals_vector.data();
    int64_t* row_out = aligned_sliced_rows_vector.data();
    uint32_t* col_out = sliced_col_offsets_vector.data();
    *col_out = 0;
    ++col_out;

    memcpy(tmp_ptrs_vector.data(), batch.col_offsets, batch_col_num * sizeof(uint32_t));

    for (size_t curr_slice = 0; curr_slice != slice_num; ++curr_slice)
    {
      // *ptr is in [*batch.col_offsets, *batch.col_offsets+1), i.e. the current row
      uint32_t* HWY_RESTRICT ptr = tmp_ptrs_vector.data();
      uint32_t* HWY_RESTRICT end = ptr + batch_col_num;
      const uint32_t* HWY_RESTRICT row_end_ptr = batch.col_offsets + 1;
      for (; ptr != end; ++ptr, ++col_out, ++row_end_ptr)
      {
        const int64_t* HWY_RESTRICT row_ptr = batch.rows + *ptr;
        const int64_t* HWY_RESTRICT row_end = batch.rows + *row_end_ptr;
        const double* HWY_RESTRICT val_ptr = batch.vals + *ptr;

        while (row_ptr != row_end && *row_ptr / SLICE_SIZE == curr_slice)
        {
          *(val_out++) = *(val_ptr++);
          *(row_out++) = *(row_ptr++) % SLICE_SIZE;
        }
        int64_t* HWY_RESTRICT row_out_nxt = alignPtrUp(row_out);
        *col_out = row_out_nxt - aligned_sliced_rows_vector.data();
        *ptr = row_ptr - batch.rows;

        std::fill(row_out, row_out_nxt, SLICE_SIZE);
        std::memset(val_out, 0, (row_out_nxt - row_out) * sizeof(double));
        val_out += row_out_nxt - row_out;
        row_out = row_out_nxt;
      }
    }
    calcNorms();
  }

  void calcNorms()
  {
    const size_t lanes = hn::Lanes(d);
    {
      const HWY_RESTRICT uint32_t* col_ptr = sliced_col_offsets_vector.data();
      const HWY_RESTRICT uint32_t* col_end = col_ptr + batch_col_num * slice_num;
      std::memset(aligned_norms.data(), 0, batch_col_num * sizeof(double));
      while (col_ptr != col_end)
      {
        double* HWY_RESTRICT out = aligned_norms.data();
        double* HWY_RESTRICT out_end = out + batch_col_num;
        for (; out != out_end; ++col_ptr, ++out)
        {
          const double* HWY_RESTRICT val_ptr = aligned_sliced_vals_vector.data() + col_ptr[0];
          const double* HWY_RESTRICT val_end = aligned_sliced_vals_vector.data() + col_ptr[1];
          auto res = hn::Zero(d);
          for (; val_ptr != val_end; val_ptr += lanes)
          {
            auto curr = hn::Load(d, val_ptr);
            res = hn::MulAdd(curr, curr, res);
          }
          *out += hn::ReduceSum(d, res);
        }
      }
    }
    double* HWY_RESTRICT ptr = aligned_norms.data();
    double* HWY_RESTRICT end = aligned_norms.data() + batch_aligned;
    for (; ptr != end; ptr += lanes)
    {
      auto curr = hn::Load(d, ptr);
      curr = hn::Sqrt(curr);
      hn::Store(curr, d, ptr);
    }
  }

  void allocate(size_t col_num, size_t row_num, size_t nnz)
  {
    batch_col_num = col_num;
    this->row_num = row_num;
    slice_num = (row_num + SLICE_SIZE - 1) / SLICE_SIZE;
    const size_t lanes = hn::Lanes(d);
    batch_aligned = lane::RoundUpTo(batch_col_num, lanes);

    size_t bound1 = col_num * slice_num * (lanes - 1) + nnz;
    size_t bound2 = nnz * lanes;
    size_t aligned_val_bound = std::min(bound1, bound2);
    aligned_val_bound = std::max(aligned_val_bound, aligned_sliced_vals_vector.size());
    aligned_sliced_vals_vector.resize(aligned_val_bound);
    aligned_sliced_rows_vector.resize(aligned_val_bound);
    sliced_col_offsets_vector.resize(
      std::max(sliced_col_offsets_vector.size(), batch_col_num * slice_num + 1));
    aligned_tmp_res_vector.resize(std::max(aligned_tmp_res_vector.size(), batch_aligned));
    tmp_ptrs_vector.resize(std::max(tmp_ptrs_vector.size(), batch_col_num));
    aligned_norms.resize(std::max(aligned_norms.size(), batch_aligned));
  }

  void calcInner(auto& write_view)
  {
    for (size_t i = 1; i < batch_col_num; ++i)
    {
      calcDistI(i);
      for (size_t j = 0; j < i; ++j)
      {
        write_view(i, j) = aligned_tmp_res_vector[j];
      }
    }
  }

  void calcDistI(size_t col_pos)
  {
    const size_t lanes = hn::Lanes(d);
    setATmpRes(col_pos);
    uint32_t* HWY_RESTRICT col_ptr = sliced_col_offsets_vector.data();
    uint32_t* HWY_RESTRICT col_end = sliced_col_offsets_vector.data() + batch_col_num * slice_num;
    for (; col_ptr != col_end; col_ptr += batch_col_num)
    {
      // Set dense_slice
      {
        double* HWY_RESTRICT val_ptr = aligned_sliced_vals_vector.data() + col_ptr[col_pos];
        int64_t* HWY_RESTRICT row_ptr = aligned_sliced_rows_vector.data() + col_ptr[col_pos];
        int64_t* HWY_RESTRICT row_end = aligned_sliced_rows_vector.data() + col_ptr[col_pos + 1];
        for (; row_ptr != row_end; val_ptr += lanes, row_ptr += lanes)
        {
          auto val = hn::Load(d, val_ptr);
          auto row = hn::Load(id, row_ptr);
          hn::ScatterIndex(val, d, dense_slice, row);
        }
      }
      calcDistSStoDS(MatrixViewCSC<const double, const int64_t, const uint32_t>{
        .vals = aligned_sliced_vals_vector.data(),
        .rows = aligned_sliced_rows_vector.data(),
        .col_offsets = col_ptr,
        .col_num = col_pos,
      });
      {
        int64_t* HWY_RESTRICT row_ptr = aligned_sliced_rows_vector.data() + col_ptr[col_pos];
        int64_t* HWY_RESTRICT row_end = aligned_sliced_rows_vector.data() + col_ptr[col_pos + 1];
        if (row_end - row_ptr >= SLICE_SIZE / lanes)
        {
          resetDenseSlice();
        }
        else
        {
          auto zero = hn::Zero(d);
          for (; row_ptr != row_end; row_ptr += lanes)
          {
            hn::ScatterIndex(zero, d, dense_slice, hn::Load(id, row_ptr));
          }
        }
      }
    }
    normalizeRes(col_pos, aligned_norms[col_pos]);
  }

  void calcOuter(auto& write_view, auto& vec_getter, size_t outer_col_count)
  {
    for (size_t i = 0; i < outer_col_count; ++i)
    {
      calcDistOToB(vec_getter());
      for (size_t j = 0; j < batch_col_num; ++j)
      {
        write_view(i, j) = aligned_tmp_res_vector[j];
      }
    }
  }

  // Outer vector to batch
  void calcDistOToB(ColumnViewCSC<const double, const int64_t> ovec)
  {
    setATmpRes(batch_col_num);
    double norm = calcSVecNorm(ovec);
    const double* HWY_RESTRICT val_ptr = ovec.vals;
    const int64_t* HWY_RESTRICT row_ptr = ovec.rows;
    const int64_t* HWY_RESTRICT end_ptr = ovec.rows + ovec.len;
    while (row_ptr != end_ptr)
    {
      const int64_t* HWY_RESTRICT old = row_ptr;
      size_t curr_slice = *row_ptr / SLICE_SIZE;

      for (; (row_ptr != end_ptr) && ((*row_ptr / SLICE_SIZE) == curr_slice); ++row_ptr, ++val_ptr)
      {
        dense_slice[*row_ptr % SLICE_SIZE] = *val_ptr;
      }

      calcDistSStoDS(MatrixViewCSC<const double, const int64_t, const uint32_t>{
        .vals = aligned_sliced_vals_vector.data(),
        .rows = aligned_sliced_rows_vector.data(),
        .col_offsets = sliced_col_offsets_vector.data() + curr_slice * batch_col_num,
        .col_num = batch_col_num,
      });
      // If vector is dense, reset entire slice, otherwise, only necessary cells
      if (row_ptr - old >= SLICE_SIZE / hn::Lanes(d))
      {
        resetDenseSlice();
      }
      else
      {
        for (; old != row_ptr; ++old)
        {
          dense_slice[*old % SLICE_SIZE] = 0.0;
        }
      }
    }
    normalizeRes(batch_col_num, norm);
  }

  double calcSVecNorm(ColumnViewCSC<const double, const int64_t> svec)
  {
    const double* HWY_RESTRICT ptr = svec.vals;
    const double* HWY_RESTRICT end = ptr + svec.len;

    double res = 0.0;
    // I expect the compiler to optimize this
    for (; ptr != end; ++ptr)
    {
      double curr = *ptr;
      res += curr * curr;
    }
    return std::sqrt(res);
  }

  // resets aligned_tmp_res_vector.data()
  void setATmpRes(size_t len)
  {
    const size_t lanes = hn::Lanes(d);
    double* HWY_RESTRICT ptr = aligned_tmp_res_vector.data();
    double* HWY_RESTRICT end = ptr + lane::RoundUpTo(len, lanes);
    auto val_v = hn::Zero(d);
    for (; ptr != end; ptr += lanes)
    {
      hn::Store(val_v, d, ptr);
    }
  }

  void resetDenseSlice()
  {
    const size_t lanes = hn::Lanes(d);
    double* HWY_RESTRICT ptr = dense_slice;
    double* HWY_RESTRICT end = ptr + SLICE_SIZE;
    auto zero = hn::Zero(d);
    for (; ptr != end; ptr += lanes)
    {
      hn::Store(zero, d, ptr);
    }
  }

  // Aligned sparse slice to dense slice
  void calcDistSStoDS(MatrixViewCSC<const double, const int64_t, const uint32_t> s_slice)
  {
    const uint32_t* HWY_RESTRICT col_offset_ptr = s_slice.col_offsets;
    // col_offset_ptr[0] isn't guaranteed to be zero
    const double* HWY_RESTRICT val_ptr = s_slice.vals + *col_offset_ptr;
    const int64_t* HWY_RESTRICT row_ptr = s_slice.rows + *col_offset_ptr;
    double* HWY_RESTRICT res_ptr = aligned_tmp_res_vector.data();
    double* HWY_RESTRICT end_ptr = aligned_tmp_res_vector.data() + s_slice.col_num;

    for (; res_ptr != end_ptr; ++col_offset_ptr, ++res_ptr)
    {
      size_t curr_col_len = col_offset_ptr[1] - col_offset_ptr[0];
      if (curr_col_len == 0)
      {
        continue;
      }
      *res_ptr += calcDistCToDS(ColumnViewCSC<const double, const int64_t>{
        .vals = val_ptr, .rows = row_ptr, .len = curr_col_len});
      val_ptr += curr_col_len;
      row_ptr += curr_col_len;
    }
  }

  // Aligned sparse column to dense slice
  double calcDistCToDS(ColumnViewCSC<const double, const int64_t> col)
  {
    size_t lanes = hn::Lanes(d);
    const double* HWY_RESTRICT ptr = col.vals;
    const double* HWY_RESTRICT end = ptr + col.len;
    const int64_t* HWY_RESTRICT rows = col.rows;
    auto res = hn::Zero(d);
    for (; ptr != end; ptr += lanes, rows += lanes)
    {
      auto ind = hn::Load(id, rows);
      auto a = hn::Load(d, ptr);
      auto b = hn::GatherIndex(d, dense_slice, ind);
      res = hn::MulAdd(a, b, res);
    }
    return hn::ReduceSum(d, res);
  }

  void normalizeRes(size_t len, double norm)
  {
    const size_t lanes = hn::Lanes(d);
    double* HWY_RESTRICT ptr = aligned_tmp_res_vector.data();
    double* HWY_RESTRICT end = ptr + lane::RoundUpTo(len, lanes);
    double* HWY_RESTRICT norm_ptr = aligned_norms.data();
    auto norm_a = hn::Set(d, norm);
    for (; ptr != end; ptr += lanes, norm_ptr += lanes)
    {
      auto curr = hn::Load(d, ptr);
      auto norm_b = hn::Load(d, norm_ptr);
      curr = hn::Div(curr, norm_a);
      curr = hn::Div(curr, norm_b);
      hn::Store(curr, d, ptr);
    }
  }
};

struct SparseBDSpearmanCalc
{

  SparseBDSpearmanCalc() { resetDenseSlice(); }

  using InputMatr = MatrixViewCSC<const double, const int32_t, const uint32_t>;
  using InputCol = ColumnViewCSC<const double, const int32_t>;
  using RankMatr = MatrixViewCSC<const float, const int32_t, const uint32_t>;
  using RankCol = ColumnViewCSC<const float, const int32_t>;

  static constexpr size_t SLICE_SIZE = 8192;
  static constexpr HWY_FULL(float) f32{};
  static constexpr HWY_FULL(double) f64{};
  static constexpr HWY_FULL(int32_t) i32{};
  static constexpr HWY_FULL(uint32_t) u32{};
  static constexpr HWY_FULL(uint64_t) u64{};

  size_t batch_col_num;
  size_t batch_aligned;
  size_t slice_num;
  size_t row_num;

  static auto alignPtrUp(auto ptr, auto tag)
  {
    const size_t lanes = hn::Lanes(tag);
    size_t offset = ptr - (decltype(ptr))nullptr;
    size_t add = lane::RoundUpTo(offset, lanes) - offset;
    return ptr + add;
  }

  static auto alignPtrDown(auto ptr, auto tag)
  {
    const size_t lanes = hn::Lanes(tag);
    size_t offset = ptr - (decltype(ptr))nullptr;
    size_t sub = offset - lane::RoundDownTo(offset, lanes);
    return ptr - sub;
  }

  std::vector<float> outer_vec_buffer;
  std::vector<float, hwy::AlignedAllocator<float>> aligned_sliced_vals_vector;
  std::vector<int32_t, hwy::AlignedAllocator<int32_t>> aligned_sliced_rows_vector;
  std::vector<uint32_t, hwy::AlignedAllocator<uint32_t>> sliced_col_offsets_vector;

  std::vector<double, hwy::AlignedAllocator<double>> aligned_tmp_res_vector;
  std::vector<uint32_t, hwy::AlignedAllocator<uint32_t>> tmp_ptrs_vector;
  std::vector<double, hwy::AlignedAllocator<double>> aligned_sliced_zero_dist;
  std::vector<double, hwy::AlignedAllocator<double>> aligned_zero_avgs;
  std::vector<double, hwy::AlignedAllocator<double>> aligned_std;
  double total_rank_sum;
  double diff;

  std::vector<float> tmp_ranks;
  std::vector<uint32_t> sorting_buffer;

  HWY_ALIGN float dense_slice[SLICE_SIZE + 1] = {};
  // todo check alignment
  void startCalculation(InputMatr batch)
  {
    allocate(
      batch.col_num,
      batch.row_num,
      /*nnz*/ batch.col_offsets[batch.col_num] - batch.col_offsets[0]);

    calcAux(batch);
    RankMatr matr = getRanksMatr(batch);

    setRankMatr(matr);
    calcZeroDist();
  }

  RankMatr getRanksMatr(InputMatr batch)
  {
    const uint32_t* HWY_RESTRICT col_ptr = batch.col_offsets;
    float* base_ptr = tmp_ranks.data() - *col_ptr;
    const uint32_t* HWY_RESTRICT col_end = col_ptr + batch.col_num;
    for (; col_ptr != col_end; ++col_ptr)
    {
      getRanksCol(batch.vals + col_ptr[0], batch.vals + col_ptr[1], base_ptr + col_ptr[0]);
    }
    return RankMatr{
      .vals = base_ptr,
      .rows = batch.rows,
      .col_offsets = batch.col_offsets,
      .col_num = batch.col_num,
      .row_num = batch.row_num,
    };
  }

  void getRanksCol(
    const double* HWY_RESTRICT ptr, const double* HWY_RESTRICT end, float* HWY_RESTRICT out)
  {
    size_t nnz = end - ptr;
    sorting_buffer.resize(std::max(nnz, sorting_buffer.size()));
    for (uint32_t i = 0; i < nnz; ++i)
    {
      sorting_buffer[i] = i;
    }
    std::sort(sorting_buffer.begin(), sorting_buffer.begin() + nnz, [&](auto a, auto b) {
      return ptr[a] < ptr[b];
    });
    for (uint32_t i = 0; i < nnz; ++i)
    {
      double val = ptr[sorting_buffer[i]];
      uint32_t res = val < 0.0 ? i : (i + row_num - nnz);
      out[sorting_buffer[i]] = static_cast<float>(res);
    }
  }

  void setRankMatr(RankMatr batch)
  {
    float* val_out = aligned_sliced_vals_vector.data();
    int32_t* row_out = aligned_sliced_rows_vector.data();
    uint32_t* col_out = sliced_col_offsets_vector.data();
    *col_out = 0;
    ++col_out;

    memcpy(tmp_ptrs_vector.data(), batch.col_offsets, batch_col_num * sizeof(uint32_t));

    for (size_t curr_slice = 0; curr_slice != slice_num; ++curr_slice)
    {
      // *ptr is in [*batch.col_offsets, *batch.col_offsets+1), i.e. the current row
      uint32_t* HWY_RESTRICT ptr = tmp_ptrs_vector.data();
      uint32_t* HWY_RESTRICT end = ptr + batch_col_num;
      const uint32_t* HWY_RESTRICT row_end_ptr = batch.col_offsets + 1;
      for (; ptr != end; ++ptr, ++col_out, ++row_end_ptr)
      {
        const int32_t* HWY_RESTRICT row_ptr = batch.rows + *ptr;
        const int32_t* HWY_RESTRICT row_end = batch.rows + *row_end_ptr;
        const float* HWY_RESTRICT val_ptr = batch.vals + *ptr;

        while (row_ptr != row_end && *row_ptr / SLICE_SIZE == curr_slice)
        {
          *(val_out++) = *(val_ptr++);
          *(row_out++) = *(row_ptr++) % SLICE_SIZE;
        }
        int32_t* HWY_RESTRICT row_out_nxt = alignPtrUp(row_out, i32);
        *col_out = row_out_nxt - aligned_sliced_rows_vector.data();
        *ptr = row_ptr - batch.rows;

        std::fill(row_out, row_out_nxt, SLICE_SIZE);
        std::fill(val_out, val_out + (row_out_nxt - row_out), 0.0);
        val_out += row_out_nxt - row_out;
        row_out = row_out_nxt;
      }
    }
  }

  void calcAux(InputMatr batch)
  {
    double n = static_cast<double>(row_num);
    total_rank_sum = n * (n - 1.0) / 2.0;
    diff = n * (n - 1.0) * (n - 1.0) / 4.0;

    const uint32_t* HWY_RESTRICT col_ptr = batch.col_offsets;
    const uint32_t* HWY_RESTRICT col_end = col_ptr + batch.col_num;
    double* HWY_RESTRICT std_ptr = aligned_std.data();
    double* HWY_RESTRICT z_avg_ptr = aligned_zero_avgs.data();
    const double* HWY_RESTRICT vals_ptr = batch.vals + col_ptr[0];
    for (; col_ptr != col_end; ++col_ptr, ++std_ptr, ++z_avg_ptr)
    {
      uint32_t nnz = col_ptr[1] - col_ptr[0];
      const double* vals_end = vals_ptr + nnz;
      uint32_t nc = calcNegativeVals(vals_ptr, vals_end);
      uint32_t zc = static_cast<uint32_t>(row_num) - nnz;
      uint32_t pc = nnz - nc;
      double z_avg = calcZeroAverage(nc, zc);
      *z_avg_ptr = z_avg;
      *std_ptr = calcStd(nc, zc, pc, z_avg);
      vals_ptr = vals_end;
    }
  }

  double calcStd(uint32_t nc_, uint32_t zc_, uint32_t pc_, double z_avg)
  {
    double avg = total_rank_sum / static_cast<double>(row_num);
    double nc = static_cast<double>(nc_);
    double zc = static_cast<double>(zc_);
    double pc = static_cast<double>(pc_);
    double res = 0.0;
    res += nc * (nc - 0.5) * (nc - 1.0) / 3.0;
    res -= nc * (nc - 1.0) * avg;
    res += nc * avg * avg;
    res += zc * (avg - z_avg) * (avg - z_avg);
    res += pc * (pc - 0.5) * (pc - 1.0) / 3.0;
    res -= pc * (pc - 1.0) * (avg - nc - zc);
    res += pc * (avg - nc - zc) * (avg - nc - zc);

    return std::sqrt(res);
  }

  double calcZeroAverage(uint32_t nc_, uint32_t zc_)
  {
    double nc = static_cast<double>(nc_);
    double zc = static_cast<double>(zc_);
    return nc + (zc - 1.0) / 2.0;
  }

  static uint32_t calcNegativeVals(const double* HWY_RESTRICT ptr, const double* HWY_RESTRICT end)
  {
    uint32_t res = 0;
    for (; ptr != end; ++ptr)
    {
      res += *ptr < 0 ? 1 : 0;
    }
    return res;
  }

  void calcZeroDist()
  {
    {
      const size_t lanes = hn::Lanes(f32);
      HWY_RESTRICT uint32_t* col_ptr = sliced_col_offsets_vector.data();
      HWY_RESTRICT uint32_t* col_end = col_ptr + batch_col_num * slice_num;
      float* HWY_RESTRICT val_ptr = aligned_sliced_vals_vector.data();
      double* HWY_RESTRICT out = aligned_sliced_zero_dist.data() + batch_aligned;
      setZeroAligned(f64, aligned_sliced_zero_dist.data(), out, lanes);
      size_t diff = batch_aligned - batch_col_num;

      while (col_ptr != col_end)
      {
        double* HWY_RESTRICT out_end = out + batch_col_num;
        double* HWY_RESTRICT avg_zero_ptr = aligned_zero_avgs.data();
        for (; out != out_end; ++col_ptr, ++out, ++avg_zero_ptr)
        {
          uint32_t len = col_ptr[1] - col_ptr[0];
          float* HWY_RESTRICT val_end = val_ptr + len;
          auto res = hn::Zero(f64);
          for (; val_ptr != val_end; val_ptr += lanes)
          {
            auto curr = hn::Load(f32, val_ptr);
            res = hn::Add(res, hn::PromoteUpperTo(f64, curr));
            res = hn::Add(res, hn::PromoteLowerTo(f64, curr));
          }
          int32_t* HWY_RESTRICT row_ptr = aligned_sliced_rows_vector.data() + col_ptr[1] - 1;
          for (; len && *row_ptr == SLICE_SIZE; --row_ptr, --len)
            ;
          *out = hn::ReduceSum(f64, res) - *avg_zero_ptr * static_cast<double>(len);
        }
        out += diff;
      }
    }
    {
      const size_t lanes = hn::Lanes(f64);
      double* HWY_RESTRICT ptr = aligned_sliced_zero_dist.data() + batch_aligned * 2;
      double* HWY_RESTRICT end = aligned_sliced_zero_dist.data() + batch_aligned * (slice_num + 1);
      for (; ptr != end; ptr += lanes)
      {
        auto curr = hn::Load(f64, ptr);
        auto prev = hn::Load(f64, ptr - batch_aligned);
        hn::Store(hn::Add(curr, prev), f64, ptr);
      }
    }
  }

  void setZeroAligned(auto tag, auto ptr, auto end, size_t lanes)
  {
    auto val = hn::Zero(tag);
    for (; ptr != end; ptr += lanes)
    {
      hn::Store(val, tag, ptr);
    }
  }

  // todo
  void allocate(size_t col_num, size_t row_num, size_t nnz)
  {
    batch_col_num = col_num;
    this->row_num = row_num;
    slice_num = (row_num + SLICE_SIZE - 1) / SLICE_SIZE;
    const size_t lanes = hn::Lanes(f32);
    batch_aligned = lane::RoundUpTo(batch_col_num, lanes);

    size_t bound1 = col_num * slice_num * (lanes - 1) + nnz;
    size_t bound2 = nnz * lanes;
    size_t aligned_val_bound = std::min(bound1, bound2);
    aligned_val_bound = std::max(aligned_val_bound, aligned_sliced_vals_vector.size());
    aligned_sliced_vals_vector.resize(aligned_val_bound);
    aligned_sliced_rows_vector.resize(aligned_val_bound);
    sliced_col_offsets_vector.resize(
      std::max(sliced_col_offsets_vector.size(), batch_col_num * slice_num + 1));
    aligned_tmp_res_vector.resize(std::max(aligned_tmp_res_vector.size(), batch_aligned));
    tmp_ptrs_vector.resize(std::max(tmp_ptrs_vector.size(), batch_col_num));
    aligned_sliced_zero_dist.resize(std::max(aligned_sliced_zero_dist.size(), batch_aligned * (slice_num + 1)));
    aligned_zero_avgs.resize(std::max(aligned_zero_avgs.size(), batch_aligned));
    aligned_std.resize(std::max(aligned_std.size(), batch_aligned));
    tmp_ranks.resize(std::max(tmp_ranks.size(), nnz));
    outer_vec_buffer.reserve(SLICE_SIZE);
    sorting_buffer.reserve(SLICE_SIZE);
  }

  void calcInner(auto& write_view)
  {
    for (size_t i = 1; i < batch_col_num; ++i)
    {
      calcDistI(i);
      for (size_t j = 0; j < i; ++j)
      {
        write_view(i, j) = aligned_tmp_res_vector[j];
      }
    }
  }

  void calcDistI(size_t col_pos)
  {
    const size_t lanes = hn::Lanes(f32);
    setATmpRes(col_pos);
    uint32_t* HWY_RESTRICT col_ptr = sliced_col_offsets_vector.data();
    uint32_t* HWY_RESTRICT col_end = sliced_col_offsets_vector.data() + batch_col_num * slice_num;
    for (; col_ptr != col_end; col_ptr += batch_col_num)
    {
      // Set dense_slice
      {
        float* HWY_RESTRICT val_ptr = aligned_sliced_vals_vector.data() + col_ptr[col_pos];
        int32_t* HWY_RESTRICT row_ptr = aligned_sliced_rows_vector.data() + col_ptr[col_pos];
        int32_t* HWY_RESTRICT row_end = aligned_sliced_rows_vector.data() + col_ptr[col_pos + 1];
        for (; row_ptr != row_end; val_ptr += lanes, row_ptr += lanes)
        {
          auto val = hn::Load(f32, val_ptr);
          auto row = hn::Load(i32, row_ptr);
          hn::ScatterIndex(val, f32, dense_slice, row);
        }
      }
      calcDistSStoDS(
        RankMatr{
          .vals = aligned_sliced_vals_vector.data(),
          .rows = aligned_sliced_rows_vector.data(),
          .col_offsets = col_ptr,
          .col_num = col_pos,
        },
        aligned_zero_avgs[col_pos]);
      {
        int32_t* HWY_RESTRICT row_ptr = aligned_sliced_rows_vector.data() + col_ptr[col_pos];
        int32_t* HWY_RESTRICT row_end = aligned_sliced_rows_vector.data() + col_ptr[col_pos + 1];
        if (row_end - row_ptr >= SLICE_SIZE / lanes)
        {
          resetDenseSlice();
        }
        else
        {
          auto min_one = hn::Set(f32, -1.0f);
          for (; row_ptr != row_end; row_ptr += lanes)
          {
            hn::ScatterIndex(min_one, f32, dense_slice, hn::Load(i32, row_ptr));
          }
          dense_slice[SLICE_SIZE] = 0.0;
        }
      }
    }
    normalizeRes(col_pos, aligned_std[col_pos]);
  }

  void calcOuter(auto& write_view, auto& vec_getter, size_t outer_col_count)
  {
    for (size_t i = 0; i < outer_col_count; ++i)
    {
      calcDistOToB(vec_getter());
      for (size_t j = 0; j < batch_col_num; ++j)
      {
        write_view(i, j) = aligned_tmp_res_vector[j];
      }
    }
  }

  void calcDistOToB(InputCol ovec)
  {
    uint32_t nnz = static_cast<uint32_t>(ovec.len);
    const double* HWY_RESTRICT vals_ptr = ovec.vals;
    const double* HWY_RESTRICT vals_end = vals_ptr + ovec.len;
    uint32_t nc = calcNegativeVals(vals_ptr, vals_end);
    uint32_t zc = static_cast<uint32_t>(row_num) - nnz;
    uint32_t pc = nnz - nc;
    double z_avg = calcZeroAverage(nc, zc);
    double std = calcStd(nc, zc, pc, z_avg);
    outer_vec_buffer.resize(std::max(outer_vec_buffer.size(), ovec.len));
    getRanksCol(vals_ptr, vals_end, outer_vec_buffer.data());
    calcDistOToB(
      RankCol{.vals = outer_vec_buffer.data(), .rows = ovec.rows, .len = ovec.len}, z_avg, std);
  }
  // Outer vector to batch
  void calcDistOToB(RankCol ovec, double zero_avg, double std)
  {
    setATmpRes(batch_col_num);
    const float* HWY_RESTRICT val_ptr = ovec.vals;
    const int32_t* HWY_RESTRICT row_ptr = ovec.rows;
    const int32_t* HWY_RESTRICT end_ptr = ovec.rows + ovec.len;
    size_t prev_slice = 0;
    while (row_ptr != end_ptr)
    {
      const int32_t* HWY_RESTRICT old = row_ptr;
      size_t curr_slice = *row_ptr / SLICE_SIZE;
      addPartialDist(prev_slice, curr_slice, zero_avg);
      prev_slice = curr_slice + 1;


      for (; (row_ptr != end_ptr) && ((*row_ptr / SLICE_SIZE) == curr_slice); ++row_ptr, ++val_ptr)
      {
        dense_slice[*row_ptr % SLICE_SIZE] = *val_ptr;
      }

      calcDistSStoDS(
        MatrixViewCSC<const float, const int32_t, const uint32_t>{
          .vals = aligned_sliced_vals_vector.data(),
          .rows = aligned_sliced_rows_vector.data(),
          .col_offsets = sliced_col_offsets_vector.data() + curr_slice * batch_col_num,
          .col_num = batch_col_num,
        },
        zero_avg);
      // If vector is dense, reset entire slice, otherwise, only necessary cells
      if (row_ptr - old >= SLICE_SIZE / hn::Lanes(f32))
      {
        resetDenseSlice();
      }
      else
      {
        for (; old != row_ptr; ++old)
        {
          dense_slice[*old % SLICE_SIZE] = -1.0f;
        }
      }
    }
    addPartialDist(prev_slice, slice_num, zero_avg);
    normalizeRes(batch_col_num, std);
  }

  void addPartialDist(size_t from, size_t to, double b_zero_avg)
  {
    if (from == to)
    {
      return;
    }
    const size_t lanes = hn::Lanes(f64);
    const double* HWY_RESTRICT dist_ptr_a = aligned_sliced_zero_dist.data() + batch_aligned * from;
    const double* HWY_RESTRICT dist_ptr_b = aligned_sliced_zero_dist.data() + batch_aligned * to;
    const double* HWY_RESTRICT dist_end_a = dist_ptr_a + batch_aligned;
    double* HWY_RESTRICT out = aligned_tmp_res_vector.data();
    auto mul_v = hn::Set(f64, b_zero_avg);
    for (; dist_ptr_a != dist_end_a; dist_ptr_a += lanes, dist_ptr_b += lanes, out += lanes)
    {
      auto a = hn::Load(f64, dist_ptr_a);
      auto b = hn::Load(f64, dist_ptr_b);
      auto curr = hn::Sub(b, a);
      auto res = hn::Load(f64, out);
      res = hn::MulAdd(curr, mul_v, res);
      hn::Store(res, f64, out);
    }
  }

  // resets aligned_tmp_res_vector.data()
  void setATmpRes(size_t len)
  {
    const size_t lanes = hn::Lanes(f64);
    double* HWY_RESTRICT a_avg_zero_ptr = aligned_zero_avgs.data();
    double* HWY_RESTRICT ptr = aligned_tmp_res_vector.data();
    double* HWY_RESTRICT end = ptr + lane::RoundUpTo(len, lanes);

    auto diff_v = hn::Set(f64, -diff);
    auto mul_v = hn::Set(f64, total_rank_sum);
    for (; ptr != end; ptr += lanes, a_avg_zero_ptr += lanes)
    {
      auto curr = hn::Load(f64, a_avg_zero_ptr);
      hn::Store(hn::MulAdd(mul_v, curr, diff_v), f64, ptr);
    }
  }

  void resetDenseSlice()
  {
    const size_t lanes = hn::Lanes(f32);
    float* HWY_RESTRICT ptr = dense_slice;
    float* HWY_RESTRICT end = ptr + SLICE_SIZE;
    auto min_one = hn::Set(f32, -1.0f);
    for (; ptr != end; ptr += lanes)
    {
      hn::Store(min_one, f32, ptr);
    }
    dense_slice[SLICE_SIZE] = 0.0;
  }

  // Aligned sparse slice to dense slice
  void calcDistSStoDS(
    MatrixViewCSC<const float, const int32_t, const uint32_t> s_slice, double z_avg_b)
  {
    const uint32_t* HWY_RESTRICT col_offset_ptr = s_slice.col_offsets;
    // col_offset_ptr[0] isn't guaranteed to be zero
    const float* HWY_RESTRICT val_ptr = s_slice.vals + *col_offset_ptr;
    const int32_t* HWY_RESTRICT row_ptr = s_slice.rows + *col_offset_ptr;
    double* HWY_RESTRICT res_ptr = aligned_tmp_res_vector.data();
    double* HWY_RESTRICT end_ptr = aligned_tmp_res_vector.data() + s_slice.col_num;
    double* HWY_RESTRICT z_avg_a_ptr = aligned_zero_avgs.data();

    for (; res_ptr != end_ptr; ++col_offset_ptr, ++res_ptr, ++z_avg_a_ptr)
    {
      size_t curr_col_len = static_cast<size_t>(col_offset_ptr[1] - col_offset_ptr[0]);
      if (curr_col_len == 0)
      {
        continue;
      }

      double next = calcDistCToDS(
        RankCol{.vals = val_ptr, .rows = row_ptr, .len = curr_col_len}, *z_avg_a_ptr, z_avg_b);
      *res_ptr += next;
      val_ptr += curr_col_len;
      row_ptr += curr_col_len;
    }
  }

  // Aligned sparse column to dense slice
  double calcDistCToDS(
    ColumnViewCSC<const float, const int32_t> col, double z_avg_a, double z_avg_b)
  {
    size_t lanes = hn::Lanes(f32);
    const float* HWY_RESTRICT ptr = col.vals;
    const float* HWY_RESTRICT end = ptr + col.len;
    const int32_t* HWY_RESTRICT rows = col.rows;
    auto main_sum = hn::Zero(f64);
    auto other_sum = hn::Zero(f64);
    auto z_avg_b_v = hn::Set(f32, z_avg_b);
    for (; ptr != end; ptr += lanes, rows += lanes)
    {
      auto ind = hn::Load(i32, rows);
      auto b = hn::GatherIndex(f32, dense_slice, ind);
      b = hn::IfNegativeThenElse(b, z_avg_b_v, b);
      other_sum = hn::Add(other_sum, hn::SumsOf2(b));

      auto a = hn::Load(f32, ptr);
      main_sum = hn::MulAdd(hn::PromoteUpperTo(f64, a), hn::PromoteUpperTo(f64, b), main_sum);
      main_sum = hn::MulAdd(hn::PromoteLowerTo(f64, a), hn::PromoteLowerTo(f64, b), main_sum);
    }
    auto res = hn::MulAdd(hn::Set(f64, -z_avg_a), other_sum, main_sum);
    return hn::ReduceSum(f64, res);
  }

  void normalizeRes(size_t len, double std_b)
  {
    const size_t lanes = hn::Lanes(f64);
    double* HWY_RESTRICT ptr = aligned_tmp_res_vector.data();
    double* HWY_RESTRICT end = ptr + lane::RoundUpTo(len, lanes);
    double* HWY_RESTRICT norm_ptr = aligned_std.data();
    auto std_b_v = hn::Set(f64, std_b);
    for (; ptr != end; ptr += lanes, norm_ptr += lanes)
    {
      auto curr = hn::Load(f64, ptr);
      auto std_a_v = hn::Load(f64, norm_ptr);
      auto div = hn::Mul(std_b_v, std_a_v);
      curr = hn::Div(curr, div);
      hn::Store(curr, f64, ptr);
    }
  }
};

} // namespace detail

using SBDCalculatorL1 = detail::SparseBDL1Calc;
using SBDCalculatorCosine = detail::SparseBDCosineCalc;
using SBDCalculatorSpearman = detail::SparseBDSpearmanCalc;
