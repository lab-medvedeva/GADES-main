# GADES

GPU-Accelerated Distance Evaluation for Single-cell data.

GADES computes pairwise distance matrices between columns (cells) of a
gene-expression matrix. It supports dense and sparse input, CUDA GPU and
multi-threaded CPU backends, and six distance metrics.

## Supported metrics

| Metric | Description |
|---|---|
| `euclidean` | L2 distance |
| `cosine` | Cosine distance (1 - cosine similarity) |
| `pearson` | Pearson correlation distance (1 - r) |
| `manhattan` | L1 / city-block distance |
| `spearman` | Spearman rank-correlation distance (1 - rho) |
| `kendall` | Kendall tau distance |

## Installation

### Prerequisites

- **CPU backend**: OpenBLAS (`sudo apt install libopenblas-dev`)
- **GPU backend** (optional): CUDA Toolkit 11.0+

```bash
cd python
pip install .
```

To build with specific CUDA architectures:
```bash
CUDA_ARCHITECTURES="80;86;90" pip install .
```

## Quick start

```python
import gades
import numpy as np

# Dense matrix: genes x cells
X = np.random.randn(2000, 500)
D = gades.distance(X, metric="euclidean")          # auto-selects GPU/CPU
D = gades.distance(X, metric="pearson", backend="cpu")

# Sparse input (scipy CSC/CSR)
import scipy.sparse
X_sp = scipy.sparse.random(2000, 500, density=0.1, format="csc")
D_sp = gades.distance(X_sp, metric="cosine")

# Pairwise between two matrices
D_pw = gades.pairwise_distance(X[:, :250], X[:, 250:], metric="euclidean")

# Check GPU availability
print(gades.has_gpu())
```

### Integration with scanpy / AnnData

```python
import scanpy as sc
import gades

adata = sc.read_h5ad("pbmc3k.h5ad")
X = adata.X.T                           # gades expects genes x cells
D = gades.distance(X, metric="spearman")
```

## API

### `gades.distance(X, metric="euclidean", backend="auto")`

Compute pairwise distance matrix between columns of X.

- **X**: `np.ndarray` of shape `(n_features, n_samples)` or `scipy.sparse` matrix
- **metric**: one of `euclidean`, `cosine`, `pearson`, `manhattan`, `spearman`, `kendall`
- **backend**: `"gpu"`, `"cpu"`, or `"auto"`
- **Returns**: `np.ndarray` of shape `(n_samples, n_samples)`

### `gades.pairwise_distance(X, Y, metric="euclidean", backend="auto")`

Compute distances between columns of X and columns of Y.

- **Returns**: `np.ndarray` of shape `(n_samples_X, n_samples_Y)`

### `gades.has_gpu()`

Returns `True` if a CUDA GPU is available.

## Environment variables

| Variable | Description |
|---|---|
| `GADES_LIB_DIR` | Custom search path for shared libraries |
| `HOBO_RT_LOG=1` | Enable GPU round-trip timing logs |
| `OMP_NUM_THREADS` | Control CPU parallelism |

## Testing

```bash
cd python
pip install ".[test]"
pytest tests/ -v
```

GPU tests are auto-skipped when no CUDA device is available. Run only GPU
tests with `pytest tests/ -v -m gpu`.
