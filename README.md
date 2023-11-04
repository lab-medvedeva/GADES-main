## GPU Dist

This repo provides code that calculates pairwise matrix distances for dense and sparse matrices.

## Prerequisities

* R 4.3.0+
* CMake 3.10+
* (Optional) CUDA 11+

## Installation instructions
```shell
git clone https://github.com/lab-medvedeva/GPUDist-main.git
cd GPUDist-main
R CMD INSTALL .
```
This command builds code of the library using CMake, checks GPU and install package using CPU+GPU or only CPU code.

### (Optional) How to build source code as a library for imports

```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

`mtrx.so` file will be appeared in the root folder.

## Usage

### Dense mode
```R
library(HobotnicaGPU)

matrix <- as.matrix(read.table('matrix.csv', header=T, row.names = 1, sep=","))
dist.matrix <- mtrx_distance(matrix, batch_size = 5000, metric = 'kendall', type='gpu', sparse=F)
```

### Sparse mode
```R
library(HobotnicaGPU)

matrix <- Matrix::readMM('./matrix.mtx')
dist.matrix <- mtrx_distance(matrix, batch_size = 5000, metric = 'kendall', type='gpu', sparse=T)
```


