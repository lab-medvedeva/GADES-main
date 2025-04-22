#include "gtest/gtest.h"
#include "GADES.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <valarray>

struct MatrixHolder : testing::Test
{
  MatrixHolder()
  {
    std::mt19937 gen(21); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-20.0, 20.0);
    for (auto& vec : {&r20c20_data, &r10c40_data, &r40c10_data})
    {
      vec->resize(400, 42);
      for (auto& val : *vec)
      {
        val = dis(gen);
      }
    }
    r20c20 = {.row_num = 20, .col_num = 20, .data = r20c20_data.data()};
    r10c40 = {.row_num = 10, .col_num = 40, .data = r10c40_data.data()};
    r40c10 = {.row_num = 40, .col_num = 10, .data = r40c10_data.data()};
  }
  MatrixView<const double> r20c20;
  MatrixView<const double> r10c40;
  MatrixView<const double> r40c10;

  std::vector<double> r20c20_data{400};
  std::vector<double> r10c40_data{400};
  std::vector<double> r40c10_data{400};
};

double L1Dist(size_t i, size_t j, MatrixView<const double> matr)
{
  size_t row_num = matr.row_num;
  double res = 0.0;
  for (size_t row = 0; row < row_num; ++row)
  {
    res += abs(matr[i][row] - matr[j][row]);
  }
  return res;
}

double CosineDist(size_t i, size_t j, MatrixView<const double> matr)
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

double SpearmanDist(size_t i, size_t j, MatrixView<const double> matr)
{
  std::vector<int> a_inds(matr.row_num);
  std::iota(a_inds.begin(), a_inds.end(), 0);
  std::vector<int> b_inds = a_inds;
  std::sort(a_inds.begin(), a_inds.end(), [&](auto a, auto b) { return matr[i][a] > matr[i][b]; });
  std::sort(b_inds.begin(), b_inds.end(), [&](auto a, auto b) { return matr[j][a] > matr[j][b]; });
  std::valarray<double> ranks_a(matr.row_num);
  std::valarray<double> ranks_b(matr.row_num);
  for (size_t i = 0; i < matr.row_num; ++i)
  {
    ranks_a[a_inds[i]] = static_cast<double>(i);
    ranks_b[b_inds[i]] = static_cast<double>(i);
  }
  ranks_a -= ranks_b;
  ranks_a = ranks_a.apply([](double val) { return val * val; });
  double res = ranks_a.sum();
  double n = static_cast<double>(matr.row_num);
  double coef = 6.0 / (n * (n * n - 1.0));
  return 1.0 - res * coef;
}

void CheckMatrixL1(MatrixView<const double> matr, size_t thread_num, size_t batch_size)
{
  std::vector<double> res_data(matr.col_num * matr.col_num, 42.0);
  MatrixView<double> res{.row_num = matr.col_num, .col_num = matr.col_num, .data = res_data.data()};
  CalcDistanceL1(matr, res, batch_size, thread_num);
  for (size_t i = 0; i < matr.col_num; ++i)
  {
    for (size_t j = 0; j < matr.col_num; ++j)
    {
      EXPECT_DOUBLE_EQ(res[i][j], L1Dist(i, j, matr)) << "Error at (" << i << ", " << j << ") ";
    }
  }
}

void CheckMatrixCosine(MatrixView<const double> matr, size_t thread_num, size_t batch_size)
{
  std::vector<double> res_data(matr.col_num * matr.col_num, 42.0);
  MatrixView<double> res{.row_num = matr.col_num, .col_num = matr.col_num, .data = res_data.data()};
  CalcDistanceCosine(matr, res, batch_size, thread_num);
  for (size_t i = 0; i < matr.col_num; ++i)
  {
    for (size_t j = 0; j < matr.col_num; ++j)
    {
      EXPECT_NEAR(res[i][j], CosineDist(i, j, matr), 1e-15)
        << "Error at (" << i << ", " << j << ") ";
    }
  }
}

void CheckMatrixSpearman(MatrixView<const double> matr, size_t thread_num, size_t batch_size)
{
  std::vector<double> res_data(matr.col_num * matr.col_num, 42.0);
  MatrixView<double> res{.row_num = matr.col_num, .col_num = matr.col_num, .data = res_data.data()};
  CalcDistanceSpearman(matr, res, batch_size, thread_num);
  for (size_t i = 0; i < matr.col_num; ++i)
  {
    for (size_t j = 0; j < matr.col_num; ++j)
    {
      EXPECT_NEAR(res[i][j], SpearmanDist(i, j, matr), 1e-15)
        << "Error at (" << i << ", " << j << ") " << std::endl;
    }
  }
}

using L1 = MatrixHolder;

TEST_F(L1, square)
{
  CheckMatrixL1(r20c20, 1, 0);
}

TEST_F(L1, RectRgtC)
{
  CheckMatrixL1(r40c10, 1, 0);
}

TEST_F(L1, RectRltC)
{
  CheckMatrixL1(r10c40, 1, 0);
}

TEST_F(L1, OddBatch)
{
  CheckMatrixL1(r10c40, 1, 3);
  CheckMatrixL1(r40c10, 1, 3);
  CheckMatrixL1(r20c20, 1, 3);
}

TEST_F(L1, Multithread)
{
  CheckMatrixL1(r10c40, 0, 0);
  CheckMatrixL1(r40c10, 0, 0);
  CheckMatrixL1(r20c20, 0, 0);
}

using Cosine = MatrixHolder;

TEST_F(Cosine, square)
{
  CheckMatrixCosine(r20c20, 1, 0);
}

TEST_F(Cosine, RectRgtC)
{
  CheckMatrixCosine(r40c10, 1, 0);
}

TEST_F(Cosine, RectRltC)
{
  CheckMatrixCosine(r10c40, 1, 0);
}

TEST_F(Cosine, OddBatch)
{
  CheckMatrixCosine(r10c40, 1, 3);
  CheckMatrixCosine(r40c10, 1, 3);
  CheckMatrixCosine(r20c20, 1, 3);
}

TEST_F(Cosine, Multithread)
{
  CheckMatrixCosine(r10c40, 0, 0);
  CheckMatrixCosine(r40c10, 0, 0);
  CheckMatrixCosine(r20c20, 0, 0);
}

using Spearman = MatrixHolder;

TEST_F(Spearman, square)
{
  CheckMatrixSpearman(r20c20, 1, 0);
}

TEST_F(Spearman, RectRgtC)
{
  CheckMatrixSpearman(r40c10, 1, 0);
}

TEST_F(Spearman, RectRltC)
{
  CheckMatrixSpearman(r10c40, 1, 0);
}

TEST_F(Spearman, OddBatch)
{
  CheckMatrixSpearman(r10c40, 1, 3);
  CheckMatrixSpearman(r40c10, 1, 3);
  CheckMatrixSpearman(r20c20, 1, 3);
}

TEST_F(Spearman, Multithread)
{
  CheckMatrixSpearman(r10c40, 0, 0);
  CheckMatrixSpearman(r40c10, 0, 0);
  CheckMatrixSpearman(r20c20, 0, 0);
}
