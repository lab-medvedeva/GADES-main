#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <valarray>

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
