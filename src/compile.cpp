#include "MatrixReaders.hpp"
#include "MatrixWriters.hpp"
#include "BDCalculators.hpp"
#include "DistCalculator.hpp"

int main()
{
  DistanceCalculator<BDCalculatorL1> calc;
  BasicMatrixReader reader;
  BasicMatrixWriter writer;
  BDCalculatorL1 calcl1;
  calc.calcDistance([&]() { return writer; }, [&]() { return reader; }, [&]() { return calcl1; });
}
