#include "private/utils.hpp"

void CalcDistanceL1(
  MatrixView<const double> data,
  MatrixView<double> res,
  size_t batch_size = 0,
  size_t thread_num = 0);

void CalcDistanceCosine(
    MatrixView<const double> data,
    MatrixView<double> res,
    size_t batch_size = 0,
    size_t thread_num = 0);
