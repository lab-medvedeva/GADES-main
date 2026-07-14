from __future__ import annotations

import numpy as np
import scipy.sparse

from ._backend import backend as _backend

METRICS = {
    "euclidean": 0,
    "cosine": 1,
    "pearson": 2,
    "manhattan": 3,
    "spearman": 4,
    "kendall": 5,
}


def _resolve_backend(requested: str):
    if requested == "gpu":
        if not _backend.has_gpu():
            raise RuntimeError("GPU requested but no CUDA device available")
        return "gpu"
    if requested == "cpu":
        return "cpu"
    # auto
    return "gpu" if _backend.has_gpu() else "cpu"


def _metric_code(metric: str) -> int:
    m = metric.lower()
    if m not in METRICS:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Supported: {', '.join(METRICS)}"
        )
    return METRICS[m]


def _prepare_dense(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D array, got {X.ndim}-D")
    return np.asfortranarray(X)


def _prepare_sparse(X):
    if not scipy.sparse.issparse(X):
        raise TypeError("Expected a scipy sparse matrix")
    csc = scipy.sparse.csc_matrix(X)
    indices = np.ascontiguousarray(csc.indices, dtype=np.int32)
    indptr = np.ascontiguousarray(csc.indptr, dtype=np.int32)
    data = np.ascontiguousarray(csc.data, dtype=np.float64)
    return indices, indptr, data, csc.shape[0], csc.shape[1], len(csc.data)


def distance(
    X,
    metric: str = "euclidean",
    backend: str = "auto",
) -> np.ndarray:
    """Compute pairwise distance matrix between columns of X.

    Parameters
    ----------
    X : array-like of shape (n_features, n_samples) or scipy.sparse matrix
        Input data matrix. Rows are features (genes), columns are samples
        (cells). For scRNA-seq: genes x cells.
    metric : str
        One of: 'euclidean', 'cosine', 'pearson', 'manhattan', 'spearman',
        'kendall'.
    backend : str
        'gpu' (CUDA), 'cpu' (OpenBLAS+OpenMP), or 'auto' (GPU if available).

    Returns
    -------
    D : numpy.ndarray of shape (n_samples, n_samples)
        Symmetric pairwise distance matrix.
    """
    be = _resolve_backend(backend)
    mc = _metric_code(metric)

    if scipy.sparse.issparse(X):
        indices, indptr, data, n, m, nnz = _prepare_sparse(X)
        out = np.empty((m, m), dtype=np.float64, order="F")
        if be == "gpu":
            rc = _backend.gpu.gades_sparse_gpu(
                indices.ctypes.data_as(_backend.gpu.gades_sparse_gpu.argtypes[0]),
                indptr.ctypes.data_as(_backend.gpu.gades_sparse_gpu.argtypes[1]),
                data.ctypes.data_as(_backend.gpu.gades_sparse_gpu.argtypes[2]),
                out.ctypes.data_as(_backend.gpu.gades_sparse_gpu.argtypes[3]),
                n, m, nnz, mc,
            )
        else:
            rc = _backend.cpu.gades_sparse_cpu(
                indices.ctypes.data_as(_backend.cpu.gades_sparse_cpu.argtypes[0]),
                indptr.ctypes.data_as(_backend.cpu.gades_sparse_cpu.argtypes[1]),
                data.ctypes.data_as(_backend.cpu.gades_sparse_cpu.argtypes[2]),
                out.ctypes.data_as(_backend.cpu.gades_sparse_cpu.argtypes[3]),
                n, m, nnz, mc,
            )
        if rc != 0:
            raise RuntimeError(f"Backend returned error code {rc}")
        return np.ascontiguousarray(out)

    # Dense path
    X_f = _prepare_dense(X)
    n, m = X_f.shape
    out = np.empty((m, m), dtype=np.float64, order="F")
    if be == "gpu":
        rc = _backend.gpu.gades_dense_gpu(
            X_f.ctypes.data_as(_backend.gpu.gades_dense_gpu.argtypes[0]),
            out.ctypes.data_as(_backend.gpu.gades_dense_gpu.argtypes[1]),
            n, m, mc,
        )
    else:
        rc = _backend.cpu.gades_dense_cpu(
            X_f.ctypes.data_as(_backend.cpu.gades_dense_cpu.argtypes[0]),
            out.ctypes.data_as(_backend.cpu.gades_dense_cpu.argtypes[1]),
            n, m, mc,
        )
    if rc != 0:
        raise RuntimeError(f"Backend returned error code {rc}")
    return np.ascontiguousarray(out)


def pairwise_distance(
    X,
    Y,
    metric: str = "euclidean",
    backend: str = "auto",
) -> np.ndarray:
    """Compute pairwise distances between columns of X and columns of Y.

    Parameters
    ----------
    X : array-like of shape (n_features, n_samples_a) or scipy.sparse matrix
    Y : array-like of shape (n_features, n_samples_b) or scipy.sparse matrix
        Must have the same number of rows (features) as X.
    metric : str
        One of: 'euclidean', 'cosine', 'pearson', 'manhattan', 'spearman',
        'kendall'.
    backend : str
        'gpu', 'cpu', or 'auto'.

    Returns
    -------
    D : numpy.ndarray of shape (n_samples_a, n_samples_b)
        Distance matrix.
    """
    be = _resolve_backend(backend)
    mc = _metric_code(metric)

    both_sparse = scipy.sparse.issparse(X) and scipy.sparse.issparse(Y)

    if both_sparse:
        ai, ap, ax, n_a, m_a, nnz_a = _prepare_sparse(X)
        bi, bp, bx, n_b, m_b, nnz_b = _prepare_sparse(Y)
        if n_a != n_b:
            raise ValueError(
                f"X and Y must have the same number of rows: {n_a} != {n_b}"
            )
        out = np.empty((m_a, m_b), dtype=np.float64, order="F")
        if be == "gpu":
            fn = _backend.gpu.gades_sparse_pairwise_gpu
        else:
            fn = _backend.cpu.gades_sparse_pairwise_cpu
        rc = fn(
            ai.ctypes.data_as(fn.argtypes[0]),
            ap.ctypes.data_as(fn.argtypes[1]),
            ax.ctypes.data_as(fn.argtypes[2]),
            bi.ctypes.data_as(fn.argtypes[3]),
            bp.ctypes.data_as(fn.argtypes[4]),
            bx.ctypes.data_as(fn.argtypes[5]),
            out.ctypes.data_as(fn.argtypes[6]),
            n_a, m_a, m_b, nnz_a, nnz_b, mc,
        )
        if rc != 0:
            raise RuntimeError(f"Backend returned error code {rc}")
        return np.ascontiguousarray(out)

    # Dense path
    X_f = _prepare_dense(X)
    Y_f = _prepare_dense(Y)
    if X_f.shape[0] != Y_f.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of rows: "
            f"{X_f.shape[0]} != {Y_f.shape[0]}"
        )
    n = X_f.shape[0]
    m_a, m_b = X_f.shape[1], Y_f.shape[1]
    out = np.empty((m_a, m_b), dtype=np.float64, order="F")
    if be == "gpu":
        fn = _backend.gpu.gades_dense_pairwise_gpu
    else:
        fn = _backend.cpu.gades_dense_pairwise_cpu
    rc = fn(
        X_f.ctypes.data_as(fn.argtypes[0]),
        Y_f.ctypes.data_as(fn.argtypes[1]),
        out.ctypes.data_as(fn.argtypes[2]),
        n, m_a, m_b, mc,
    )
    if rc != 0:
        raise RuntimeError(f"Backend returned error code {rc}")
    return np.ascontiguousarray(out)
