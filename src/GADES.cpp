#include <thread>
#include <algorithm>

#include "GADES.hpp"
#include "MatrixReaders.hpp"
#include "MatrixWriters.hpp"
#include "BDCalculators.hpp"
#include "DistCalculator.hpp"

void CalcDistanceL1(
  MatrixView<const double> data, MatrixView<double> res, size_t batch_size, size_t thread_num)
{
  if (thread_num == 0)
  {
    thread_num = std::thread::hardware_concurrency();
    // Per the standard hardware_concurrency can return zero if the max nuber of threads is somehow
    // unknown
    thread_num = std::max(1ul, thread_num);
  }
  // Probably not the best solution
  if (thread_num > data.col_num)
  {
    thread_num = data.col_num;
  }
  if (batch_size == 0)
  {
    batch_size = data.col_num / thread_num;
  }
  DistanceCalculator<BDCalculatorL1> calc;
  BasicMatrixReader reader(data);
  BasicMatrixWriter writer(res);
  BDCalculatorL1 calcl1;
  DistanceCalculatorConfig config{
    .col_num = data.col_num,
    .row_num = data.row_num,
    .batch_size = batch_size,
    .thread_num = thread_num,
  };
  calc.setUp(config);
  calc.calcDistance([&]() { return writer; }, [&]() { return reader; }, [&]() { return calcl1; });
}

void CalcDistanceCosine(
  MatrixView<const double> data, MatrixView<double> res, size_t batch_size, size_t thread_num)
{
  if (thread_num == 0)
  {
    thread_num = std::thread::hardware_concurrency();
    // Per the standard hardware_concurrency can return zero if the max nuber of threads is somehow
    // unknown
    thread_num = std::max(1ul, thread_num);
  }
  // Probably not the best solution
  if (thread_num > data.col_num)
  {
    thread_num = data.col_num;
  }
  if (batch_size == 0)
  {
    batch_size = data.col_num / thread_num;
  }
  DistanceCalculator<BDCalculatorCosine> calc;
  BasicMatrixReader reader(data);
  BasicMatrixWriter writer(res, 1.0);
  BDCalculatorCosine calcl1;
  DistanceCalculatorConfig config{
    .col_num = data.col_num,
    .row_num = data.row_num,
    .batch_size = batch_size,
    .thread_num = thread_num,
  };
  calc.setUp(config);
  calc.calcDistance([&]() { return writer; }, [&]() { return reader; }, [&]() { return calcl1; });
}
