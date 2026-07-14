## GADES - GPU-Assisted Distance Estimation Software

This repo provides code that calculates pairwise matrix distances for dense and sparse matrices.

GADES ships two front-ends over the same C++/CUDA core:

* an **R package** (`GADES`), and
* a **Python package** (`gades`).

## Prerequisities

* CMake 3.10+
* OpenBLAS (CPU backend)
* (Optional) CUDA 11+ and cuBLAS (GPU backend; the CPU backend builds and runs without CUDA)
* R 4.3.0+ (for the R package)
* Python 3.9+ (for the Python package)

## Installation instructions

### Docker image start CUDA

Please, install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) first.

```shell
docker run --name gades --gpus all -it akhtyamovpavel/gades-gpu
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

### Python package

```shell
git clone https://github.com/lab-medvedeva/GADES-main.git
cd GADES-main/python

# CPU-only backend (no CUDA needed)
pip install .

# CPU + GPU backend — set the CUDA arch(es) of your card(s).
# e.g. "86" for an RTX 3090, or a list "70;75;80;86;89;90".
CUDA_ARCHITECTURES="86" pip install .
```

The CPU backend is always built; the GPU backend is added automatically when a
CUDA toolkit is found at build time. Use `gades.has_gpu()` to check at runtime.

## Usage (R package)

### Dense mode
```R
library(GADES)

mtx <- matrix(runif(100000), nrow=100)

dist.matrix <- mtrx_distance(mtx, batch_size = 5000, metric = 'kendall', type='gpu', sparse=F, write=T)
```

### Sparse mode
```R
library(GADES)
library(Matrix)

mtx <- rsparsematrix(nrow=100, ncol=1000, density=0.1)

dist.matrix <- mtrx_distance(mtx, batch_size = 5000, metric = 'kendall', type='cpu', sparse=T, write=T)
```

### Sparse mode - GPU
```R
library(GADES)
library(Matrix)
mtx <- rsparsematrix(nrow=100, ncol=1000, density=0.1)
dist.matrix <- mtrx_distance(mtx, batch_size = 5000, metric = 'kendall', type='gpu', sparse=T, write=T)
```

## Usage (Python package)

Input is a `genes x cells` matrix; `distance()` returns the symmetric
`cells x cells` distance matrix. Metrics: `euclidean`, `cosine`, `pearson`,
`manhattan`, `spearman`, `kendall`. Backend is `"auto"` (GPU if available),
`"gpu"`, or `"cpu"`.

### Dense mode
```python
import numpy as np
import gades

X = np.random.randn(2000, 500)                 # 2000 genes x 500 cells
D = gades.distance(X, metric="cosine")         # (500, 500), auto backend
D_cpu = gades.distance(X, metric="kendall", backend="cpu")
```

### Sparse mode
```python
import scipy.sparse
import gades

X = scipy.sparse.random(2000, 500, density=0.1, format="csc")
D = gades.distance(X, metric="euclidean", backend="gpu")
```

### Cross-set (two matrices)
```python
import numpy as np
import gades

X = np.random.randn(2000, 500)                 # genes x cells_a
Y = np.random.randn(2000, 300)                 # genes x cells_b (same genes)
D = gades.pairwise_distance(X, Y, metric="pearson")   # (500, 300)
```

### Helpers
```python
import gades

gades.has_gpu()            # True if a CUDA device is usable
gades.SUPPORTED_METRICS    # ['euclidean', 'cosine', 'pearson', ...]
```


