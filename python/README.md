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

### Streaming large .h5ad files

For files too large to load, read cell blocks straight out of HDF5 — no
`anndata` dependency and no version coupling (`backed='r'` sparse slicing is
broken in several anndata/scipy combinations).

```python
import gades

gades.h5ad_info("atlas.h5ad")
# {'shape': (1058909, 36161), 'encoding': 'csr_matrix', 'layers': [], ...}

obs = gades.read_obs("atlas.h5ad", columns=["cell_type"])   # never touches X

with gades.H5adReader("atlas.h5ad") as reader:
    block = reader.slice_cells(500_000, 505_000)     # genes x cells, CSC
    cells = reader.take_cells(my_indices)            # scattered cells, chunked
    dense = reader.to_dense(cells=(0, 3000))         # F-ordered float64

D = gades.distance_from_h5ad("atlas.h5ad", metric="kendall", cells=(0, 3000))
```

Measured on a 9.4 GB file (1 058 909 cells × 36 161 genes, 2.15e9 nonzeros):
`h5ad_info` 0.03 s, a 5000-cell block 0.2 s, 1500 scattered cells 0.3 s, **peak
RSS 1.1 GB**. File size is not the constraint — the `n × n` output is, and
`distance_from_h5ad` refuses to allocate one over `max_output_bytes` (8 GiB by
default).

Blocks come back as **genes × cells**, the orientation `distance()` wants. This
is free: AnnData stores `X` as CSR over cells, and a CSR block's
`(data, indices, indptr)` triple read as CSC *is* its own transpose — so there
is no sparse transpose and no copy. (Same trick as `R/h5ad_reader.R`.)

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

### `gades.H5adReader(path, layer=None)`

Streaming access to an `.h5ad` matrix. Handles `csr_matrix`, `csc_matrix` and
dense encodings; requires `h5py`.

- `.slice_cells(start, stop, genes=None)` — contiguous cell block, **genes × cells**
- `.take_cells(indices, genes=None, chunk=8192)` — arbitrary cells, read in
  contiguous chunks and filtered (never row-by-row); output follows the order
  of `indices`
- `.to_dense(cells=None, genes=None)` — Fortran-ordered float64, ready for `distance()`
- `.obs(columns=None)`, `.obs_column(name)`, `.obs_names()`, `.var_names()`, `.layers`

Use as a context manager, or call `.close()`.

### `gades.h5ad_info(path, layer=None)` / `gades.read_obs(path, columns=None)`

Metadata and obs without reading the expression matrix.

### `gades.distance_from_h5ad(path, metric=..., cells=None, genes=None, layer=None, backend="auto", dense=False, max_output_bytes=8 GiB)`

Stream and compute in one call. `cells` is `None`, a `(start, stop)` tuple, or
an array of indices.

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
