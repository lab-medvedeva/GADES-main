## GADES - GPU-Assisted Distance Estimation Software

This repo provides code that calculates pairwise matrix distances for dense and sparse matrices.

## Prerequisities

* R 4.0.0+
* CMake 3.10+
* (Optional) CUDA 11+

## Installation instructions

### Docker image start

```shell
docker run --names gades-gpu akhtyamovpavel/gades:gpu
```

### Local installation
```shell
git clone https://github.com/lab-medvedeva/GADES-main.git
cd GADES
Rscript install.R
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

mtx<-matrix(runif(100000),nrow=100)
dense.mtx <- as.matrix(read.table(mtx, header=T, row.names = 1, sep=","))
dist.matrix <- mtrx_distance(dense.mtx, batch_size = 5000, metric = 'kendall', type='gpu', sparse=F)
```

### Sparse mode
```R
library(HobotnicaGPU)


matrix <- rsparsematrix(nrow, ncol, density)

matrix <- Matrix::readMM('./matrix.mtx')
dist.matrix <- mtrx_distance(matrix, batch_size = 5000, metric = 'kendall', type='gpu', sparse=T)
```

### Sparse mode - GPU
```R
library(HobotnicaGPU)
library(Matrix)
matrix <- Matrix::readMM(matrix.mtx')
dist.matrix <- mtrx_distance(matrix, batch_size = 5000, metric = 'kendall', type='gpu', sparse=T)
```


