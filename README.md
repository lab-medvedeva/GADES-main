## GADES - GPU-Assisted Distance Estimation Software

This repo provides code that calculates pairwise matrix distances for dense and sparse matrices.

## Prerequisities

* R 4.0.0+
* CMake 3.10+
* (Optional) CUDA 11+

## Installation instructions

### Docker image start CUDA

Please, install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) first.

```shell
docker run --gpus all --name gades-gpu akhtyamovpavel/gades:gpu
```

### Local installation
```shell
git clone https://github.com/lab-medvedeva/GADES-main.git
cd GADES-main
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
library(GADES)

mtx <- matrix(runif(100000), nrow=100)

dist.matrix <- mtrx_distance(mtx, batch_size = 5000, metric = 'kendall', type='gpu', sparse=F, write=T)
```

### Sparse mode
```R
library(GADES)

mtx <- rsparsematrix(nrow=100, ncol=1000, density=0.1)

dist.matrix <- mtrx_distance(mtx, batch_size = 5000, metric = 'kendall', type='cpu', sparse=T, write=T)
```

### Sparse mode - GPU
```R
library(GADES)
mtx <- rsparsematrix(nrow=100, ncol=1000, density=0.1)
dist.matrix <- mtrx_distance(mtx, batch_size = 5000, metric = 'kendall', type='gpu', sparse=T, write=T)
```


