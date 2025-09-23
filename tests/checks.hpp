#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <valarray>
#include <unordered_map>
#include <cinttypes>
#include "GADES.hpp"


inline double L1Dist(size_t i, size_t j, MatrixView<const double> matr)
{
  size_t row_num = matr.row_num;
  double res = 0.0;
  for (size_t row = 0; row < row_num; ++row)
  {
    res += abs(matr[i][row] - matr[j][row]);
  }
  return res;
}

inline double CosineDist(size_t i, size_t j, MatrixView<const double> matr)
{
  size_t row_num = matr.row_num;
  double res = 0.0;
  double norm_a = 0.0;
  double norm_b = 0.0;
  for (size_t row = 0; row < row_num; ++row)
  {
    res += matr[i][row] * matr[j][row];
    norm_a += matr[i][row] * matr[i][row];
    norm_b += matr[j][row] * matr[j][row];
  }
  return res / sqrt(norm_a * norm_b);
}

inline double SpearmanDist(size_t i, size_t j, MatrixView<const double> matr)
{
  double n = static_cast<double>(matr.row_num);
  std::vector<int> a_inds(matr.row_num);
  std::iota(a_inds.begin(), a_inds.end(), 0);
  std::vector<int> b_inds = a_inds;
  std::sort(a_inds.begin(), a_inds.end(), [&](auto a, auto b) { return matr[i][a] < matr[i][b]; });
  std::sort(b_inds.begin(), b_inds.end(), [&](auto a, auto b) { return matr[j][a] < matr[j][b]; });
  std::valarray<double> ranks_a(matr.row_num);
  std::valarray<double> ranks_b(matr.row_num);
  for (size_t i = 0; i < matr.row_num; ++i)
  {
    ranks_a[a_inds[i]] = static_cast<double>(i);
    ranks_b[b_inds[i]] = static_cast<double>(i);
  }
  ranks_a -= ranks_a.sum() / n;
  ranks_a /= std::sqrt(ranks_a.apply([](double a) { return a * a; }).sum() / n);
  ranks_b -= ranks_b.sum() / n;
  ranks_b /= std::sqrt(ranks_b.apply([](double a) { return a * a; }).sum() / n);
  double res = (ranks_a * ranks_b).sum() / n;
  return res;
}

inline static uint32_t calcNegatives(ColumnViewCSC<const double, const int32_t> a)
{
  uint32_t res = 0;
  for (size_t i = 0; i < a.len; ++i)
  {
    res += a.vals[i] < 0.0 ? 1 : 0;
  }
  return res;
}

inline static auto calcRanks(ColumnViewCSC<const double, const int32_t> col, size_t row_num)
{
  std::vector<double> ranks(col.len);
  std::vector<int> inds(col.len);
  std::iota(inds.begin(), inds.end(), 0);
  std::sort(inds.begin(), inds.end(), [&](auto a, auto b) { return col.vals[a] < col.vals[b]; });
  for (size_t i = 0; i < col.len; ++i)
  {
    double val = col.vals[inds[i]];
    ranks[inds[i]] = val < 0 ? static_cast<double>(i) : row_num - col.len + i;
  }
  return ranks;
}

inline static double calcAvg(const std::vector<double>& col, size_t row_num, double zero_rank)
{
  double avg = 0.0;
  double n = static_cast<double>(row_num);
  for (size_t i = 0; i < col.size(); ++i)
  {
    avg += col[i];
  }
  avg += static_cast<double>(row_num - col.size()) * zero_rank;
  avg /= n;
  return avg;
}

inline static double calcStd(const std::vector<double>& col, size_t row_num, double zero_rank)
{
  double res = 0.0;
  double avg = calcAvg(col, row_num, zero_rank);
  double n = static_cast<double>(row_num);
  for (size_t i = 0; i < col.size(); ++i)
  {
    double curr = col[i] - avg;
    res += curr * curr;
  }
  {
    double rem = zero_rank - avg;
    res += static_cast<double>(row_num - col.size()) * rem * rem;
  }
  res /= n;
  res = std::sqrt(res);
  return res;
}

inline static double SpearmanDistSparseImpl(
  ColumnViewCSC<const double, const int32_t> a,
  ColumnViewCSC<const double, const int32_t> b,
  size_t row_num)
{
  std::vector<double> ranks_a = calcRanks(a, row_num);
  std::vector<double> ranks_b = calcRanks(b, row_num);
  double zero_rank_a;
  {
    uint32_t neg_count = calcNegatives(a);
    uint32_t zero_count = row_num - a.len;
    zero_rank_a = static_cast<double>(neg_count * 2 + zero_count - 1) / 2.0;
  }
  double zero_rank_b;
  {
    uint32_t neg_count = calcNegatives(b);
    uint32_t zero_count = row_num - b.len;
    zero_rank_b = static_cast<double>(neg_count * 2 + zero_count - 1) / 2.0;
  }
  double avg_a = calcAvg(ranks_a, row_num, zero_rank_a);
  double std_a = calcStd(ranks_a, row_num, zero_rank_a);
  double avg_b = calcAvg(ranks_b, row_num, zero_rank_b);
  double std_b = calcStd(ranks_b, row_num, zero_rank_b);
  double mul = 0.0;
  std::unordered_map<uint32_t, double> pairs;
  uint32_t total_count = 0;
  for (size_t i = 0; i < a.len; ++i)
  {
    pairs[a.rows[i]] = ranks_a[i];
  }
  for (size_t i = 0; i < b.len; ++i)
  {
    ++total_count;
    double curr = ranks_b[i];
    uint32_t curr_ind = b.rows[i];
    if (pairs.contains(curr_ind))
    {
      mul += pairs[curr_ind] * curr;
      pairs.erase(curr_ind);
    }
    else
    {
      mul += zero_rank_a * curr;
    }
  }
  for (auto [ind, val] : pairs)
  {
    mul += val * zero_rank_b;
    ++total_count;
  }
  mul += static_cast<double>(row_num - total_count) * zero_rank_a * zero_rank_b;
  mul /= static_cast<double>(row_num);
  double res = (mul - avg_a * avg_b) / (std_a * std_b);

  return res;
}

inline double SpearmanDistSparse(
  size_t i, size_t j, MatrixViewCSC<const double, const int32_t, const uint32_t> matr)
{
  ColumnViewCSC<const double, const int32_t> a{
    .vals = matr.vals + matr.col_offsets[i],
    .rows = matr.rows + matr.col_offsets[i],
    .len = matr.col_offsets[i + 1] - matr.col_offsets[i],
  };
  ColumnViewCSC<const double, const int32_t> b{
    .vals = matr.vals + matr.col_offsets[j],
    .rows = matr.rows + matr.col_offsets[j],
    .len = matr.col_offsets[j + 1] - matr.col_offsets[j],
  };
  return SpearmanDistSparseImpl(a, b, matr.row_num);
}
