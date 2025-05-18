#include "gtest/gtest.h"
#include "GADES.hpp"
#include "checks.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <valarray>

struct MatrixHolderSparse : testing::Test
{
  void ConstructMatrix(size_t row_num, size_t col_num, double sparsity)
  {
    matr_data.resize(row_num * col_num);
    std::uniform_real_distribution<> dis(-20.0, 20.0);
    std::bernoulli_distribution coin(sparsity);

    for (auto& curr : matr_data)
    {
      curr = coin(gen) ? dis(gen) : 0.0;
    }

    matr = MatrixView<const double>{
      .row_num = row_num,
      .col_num = col_num,
      .data = matr_data.data(),
    };

    vals.clear();
    rows.clear();
    col_offsets.clear();
    col_offsets.push_back(0);
    for (size_t i = 0; i < col_num; ++i)
    {
      for (int64_t j = 0; j < (int64_t)row_num; ++j)
      {
        if (matr[i][j] != 0.0)
        {
          vals.push_back(matr[i][j]);
          rows.push_back(j);
        }
      }
      col_offsets.push_back(vals.size());
    }

    matr_sparse.vals = vals.data();
    matr_sparse.rows = rows.data();
    matr_sparse.col_offsets = col_offsets.data();
    matr_sparse.row_num = row_num;
    matr_sparse.col_num = col_num;
  }

  void CheckMatrix(auto calc_func, auto check_func, size_t thread_num, size_t batch_size)
  {
    std::vector<double> res_data(matr.col_num * matr.col_num, 42.0);
    MatrixView<double> res{
      .row_num = matr.col_num, .col_num = matr.col_num, .data = res_data.data()};
    calc_func(matr_sparse, res, batch_size, thread_num);
    bool is_correct = true;
    std::string error_table;
    for (size_t i = 0; i < matr.col_num; ++i)
    {
      for (size_t j = 0; j < matr.col_num; ++j)
      {
        bool curr = std::abs(res[i][j] - check_func(i, j, matr)) <= 1e-12;
        if (curr)
        {
          error_table += '.';
        }
        else
        {
          error_table += '@';
          is_correct = false;
        }
      }
      error_table += '\n';
    }
    error_table.pop_back();
    EXPECT_TRUE(is_correct)
      << "The given result and the check funcion differ at following positions: \n"
      << error_table;
  }

  std::mt19937 gen{21};

  std::vector<double> matr_data;
  MatrixView<const double> matr;

  std::vector<double> vals;
  std::vector<int64_t> rows;
  std::vector<uint32_t> col_offsets;
  MatrixViewCSC<const double, const int64_t, const uint32_t> matr_sparse;
};


using SparseL1 = MatrixHolderSparse;

TEST_F(SparseL1, basic)
{
  ConstructMatrix(20, 20, 0.5);
  CheckMatrix(
    [](auto p1, auto p2, auto p3, auto p4) { return CalcDistanceL1(p1, p2, p3, p4); },
    L1Dist,
    1,
    0);
}

TEST_F(SparseL1, advanced)
{
  ConstructMatrix(50, 10, 0.5);
  CheckMatrix(
    [](auto p1, auto p2, auto p3, auto p4) { return CalcDistanceL1(p1, p2, p3, p4); },
    L1Dist,
    0,
    3);
}

using SparseCosine = MatrixHolderSparse;

TEST_F(SparseCosine, basic)
{
  ConstructMatrix(20, 20, 0.5);
  CheckMatrix(
    [](auto p1, auto p2, auto p3, auto p4) { return CalcDistanceCosine(p1, p2, p3, p4); },
    CosineDist,
    1,
    0);
}

TEST_F(SparseCosine, advanced)
{
  ConstructMatrix(50, 10, 0.5);
  CheckMatrix(
    [](auto p1, auto p2, auto p3, auto p4) { return CalcDistanceCosine(p1, p2, p3, p4); },
    CosineDist,
    0,
    3);
}

struct MatrixHolderSparseNarrow : testing::Test
{
  void ConstructMatrix(size_t row_num, size_t col_num, double sparsity)
  {
    matr_data.resize(row_num * col_num);
    std::uniform_real_distribution<> dis(-20.0, 20.0);
    std::bernoulli_distribution coin(sparsity);

    for (auto& curr : matr_data)
    {
      curr = coin(gen) ? dis(gen) : 0.0;
    }

    matr = MatrixView<const double>{
      .row_num = row_num,
      .col_num = col_num,
      .data = matr_data.data(),
    };

    vals.clear();
    rows.clear();
    col_offsets.clear();
    col_offsets.push_back(0);
    for (size_t i = 0; i < col_num; ++i)
    {
      for (int32_t j = 0; j < (int32_t)row_num; ++j)
      {
        if (matr[i][j] != 0.0)
        {
          vals.push_back(matr[i][j]);
          rows.push_back(j * 2048);
        }
      }
      col_offsets.push_back(vals.size());
    }

    matr_sparse.vals = vals.data();
    matr_sparse.rows = rows.data();
    matr_sparse.col_offsets = col_offsets.data();
    matr_sparse.row_num = row_num * 2048;
    matr_sparse.col_num = col_num;
  }

  void CheckMatrix(auto calc_func, auto check_func, size_t thread_num, size_t batch_size)
  {
    std::vector<double> res_data(matr.col_num * matr.col_num, 42.0);
    MatrixView<double> res{
      .row_num = matr.col_num, .col_num = matr.col_num, .data = res_data.data()};
    calc_func(matr_sparse, res, batch_size, thread_num);
    bool is_correct = true;
    std::string error_table;
    for (size_t i = 0; i < matr.col_num; ++i)
    {
      for (size_t j = 0; j < matr.col_num; ++j)
      {
        EXPECT_NEAR(res[i][j], check_func(i, j, matr_sparse), 1e-12);
        bool curr = std::abs(res[i][j] - check_func(i, j, matr_sparse)) <= 1e-12;
        if (curr)
        {
          error_table += '.';
        }
        else
        {
          error_table += '@';
          is_correct = false;
        }
      }
      error_table += '\n';
    }
    error_table.pop_back();
    EXPECT_TRUE(is_correct)
      << "The given result and the check funcion differ at following positions: \n"
      << error_table;
  }

  std::mt19937 gen{21};

  std::vector<double> matr_data;
  MatrixView<const double> matr;

  std::vector<double> vals;
  std::vector<int32_t> rows;
  std::vector<uint32_t> col_offsets;
  MatrixViewCSC<const double, const int32_t, const uint32_t> matr_sparse;
};

using SparseSpearman = MatrixHolderSparseNarrow;

TEST_F(SparseSpearman, basic)
{
  ConstructMatrix(20, 20, 0.5);
  CheckMatrix(
    [](auto p1, auto p2, auto p3, auto p4) { return CalcDistanceSpearman(p1, p2, p3, p4); },
    SpearmanDistSparse,
    1,
    0);
}

TEST_F(SparseSpearman, advanced)
{
  ConstructMatrix(50, 10, 0.5);
  CheckMatrix(
    [](auto p1, auto p2, auto p3, auto p4) { return CalcDistanceSpearman(p1, p2, p3, p4); },
    SpearmanDistSparse,
    0,
    3);
}
