"""GADES — GPU-Accelerated Distance Evaluation for Single-cell data.

Computes pairwise distance matrices between columns (cells) of a
gene-expression matrix. Supports dense and sparse input, CUDA GPU
and multi-threaded CPU backends, and six distance metrics:
euclidean, cosine, pearson, manhattan, spearman, kendall.

Quick start::

    import gades
    import numpy as np

    X = np.random.randn(2000, 500)   # 2000 genes x 500 cells
    D = gades.distance(X, metric="euclidean")

    # Sparse input (scipy CSC)
    import scipy.sparse
    X_sp = scipy.sparse.random(2000, 500, density=0.1, format="csc")
    D_sp = gades.distance(X_sp, metric="cosine", backend="cpu")
"""

from .distance import distance, pairwise_distance, METRICS
from ._backend import backend as _backend

__version__ = "2.0.0b1"

SUPPORTED_METRICS = list(METRICS.keys())


def has_gpu() -> bool:
    """Check if a CUDA GPU is available for computation."""
    return _backend.has_gpu()


__all__ = [
    "distance",
    "pairwise_distance",
    "has_gpu",
    "SUPPORTED_METRICS",
    "__version__",
]
