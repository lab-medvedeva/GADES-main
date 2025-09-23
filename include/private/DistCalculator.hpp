#ifndef DISTCALCULATOR_HPP
#define DISTCALCULATOR_HPP

#include <cstdint>
#include <cstddef>
#include <thread>
#include "utils.hpp"


// An object is considered large if it does not comfortably fit into RAM
// As such Big Data techniques (batching/streaming/compression/etc) are necessary for dealing with
// large objects

// The following Distance Calculator class accepts a matrix of size NxM and returns a matrix of size
// MxM containing the pairwise distances between columns

// It's architecture is designed to be flexible enough to (potentially) work efficiently under any
// circumstances with the following assumptions in mind:

// The calculation is executed using one or more threads
// The input matrix (NxM) may or may not be large
// The input matrix is stored in a column-major format (or some analagous format which allows for
// easy sequential reading of the columns)
// A matrix of size NxK is not large, where K is at least double the number of threads
// The output matrix (MxM) may or may not be large
// The distance function d is such that d(x, x) = 0 and d(x, y) = d(y, x) for all x, y

// 1: The distance calculation algorithm is as follows:
// 2: A batch of data is loaded into read-write memory
// 3: The data is modified as necessary and ancilliary info is computed
// 4: The pairwise distance between the columns in the batch is computed
// 5: For each column in the input matrix after the loaded batch, the distance between it and every
// column in the batch is computed
// Repeat

// In the multithreaded case, the iterations of the previous algorithm are executed in parallel

struct DistanceCalculatorConfig
{
  size_t col_num;
  size_t row_num;
  size_t batch_size;
  size_t thread_num;
};

template <typename BDCalc>
class DistanceCalculator
{
public:
  void setUp(DistanceCalculatorConfig config)
  {
    total_col_num = config.col_num;
    batch_size = config.batch_size;
    thread_num = config.thread_num;
  }
  void calcDistance(auto&& writerProvider, auto&& readerProvider, auto&& calcProvider)
  {
    std::vector<decltype(writerProvider())> writers;
    std::vector<decltype(readerProvider())> readers;
    std::vector<decltype(calcProvider())> calculators;
    for (size_t i = 0; i < thread_num; ++i)
    {
      writers.emplace_back(writerProvider());
      readers.emplace_back(readerProvider());
      calculators.emplace_back(calcProvider());
    }
    auto start_thread = [&](size_t i) {
      auto thread =
        std::thread([&, i]() { calcDistanceSharded(writers[i], readers[i], calculators[i], i); });
      return thread;
    };
    std::vector<decltype(start_thread(0))> threads;
    for (size_t i = 1; i < thread_num; ++i)
    {
      threads.emplace_back(start_thread(i));
    }
    calcDistanceSharded(writers[0], readers[0], calculators[0], 0);
    for (auto& thread : threads)
    {
      thread.join();
    }
  }

private:
  void calcDistanceSharded(auto& writer, auto& reader, auto& calculator, size_t thread_ind)
  {
    size_t begin = batch_size * thread_ind;
    size_t stride = batch_size * thread_num;
    for (size_t offset = begin; offset < total_col_num; offset += stride)
    {
      size_t curr_batch = std::min(batch_size, total_col_num - offset);
      {
        auto readView = reader.getBatch(offset, curr_batch);
        calculator.startCalculation(readView);
      }
      writer.beginWrite(offset, curr_batch);
      {
        auto writeInnerView = writer.getInnerView();
        calculator.calcInner(writeInnerView);
        writer.finishInner(writeInnerView);
      }
      {
        auto writeOuterView = writer.getOuterView();
        auto readOuterIter = reader.getReadFunc(offset + curr_batch);
        calculator.calcOuter(writeOuterView, readOuterIter, total_col_num - offset - curr_batch);
        writer.finishOuter(writeOuterView);
      }
    }
  }

  size_t total_col_num = -1;
  size_t batch_size = -1;
  size_t thread_num = -1;
};


#endif // DISTCALCULATOR_HPP
