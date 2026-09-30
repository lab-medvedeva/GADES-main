import numpy as np
import pytest
import scipy.sparse
import scipy.spatial.distance
from numpy.testing import assert_allclose

import gades


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def dense_matrix():
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 30))


@pytest.fixture
def sparse_matrix():
    rng = np.random.default_rng(42)
    return scipy.sparse.random(100, 30, density=0.3, format="csc", random_state=rng)


# ── Helpers ───────────────────────────────────────────────────────────────


def scipy_pdist(X, metric):
    """Reference distance matrix via scipy."""
    if metric == "euclidean":
        d = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(X.T, metric="euclidean")
        )
    elif metric == "manhattan":
        d = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(X.T, metric="cityblock")
        )
    elif metric == "cosine":
        d = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(X.T, metric="cosine")
        )
    elif metric == "pearson":
        d = 1.0 - np.corrcoef(X.T)
    elif metric == "spearman":
        from scipy.stats import spearmanr

        corr, _ = spearmanr(X, axis=0)
        if X.shape[1] == 2:
            d = np.array([[0.0, 1.0 - corr], [1.0 - corr, 0.0]])
        else:
            d = 1.0 - corr
    elif metric == "kendall":
        # GADES Kendall = discordant_pairs / total_pairs = (1 - tau) / 2
        from scipy.stats import kendalltau

        m = X.shape[1]
        d = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                tau, _ = kendalltau(X[:, i], X[:, j])
                d[i, j] = d[j, i] = (1.0 - tau) / 2.0
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return d


# ── Dense CPU tests ───────────────────────────────────────────────────────


class TestDenseCPU:
    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
    def test_basic_metrics(self, dense_matrix, metric):
        D = gades.distance(dense_matrix, metric=metric, backend="cpu")
        ref = scipy_pdist(dense_matrix, metric)
        assert D.shape == (30, 30)
        assert_allclose(D, ref, rtol=1e-4, atol=1e-5)

    def test_pearson(self, dense_matrix):
        D = gades.distance(dense_matrix, metric="pearson", backend="cpu")
        ref = scipy_pdist(dense_matrix, "pearson")
        assert_allclose(D, ref, rtol=1e-4, atol=1e-5)

    def test_spearman(self, dense_matrix):
        D = gades.distance(dense_matrix, metric="spearman", backend="cpu")
        ref = scipy_pdist(dense_matrix, "spearman")
        assert_allclose(D, ref, rtol=1e-3, atol=1e-4)

    def test_kendall(self):
        rng = np.random.default_rng(123)
        X = rng.standard_normal((20, 8))
        D = gades.distance(X, metric="kendall", backend="cpu")
        ref = scipy_pdist(X, "kendall")
        assert_allclose(D, ref, rtol=1e-3, atol=1e-3)

    def test_symmetry(self, dense_matrix):
        D = gades.distance(dense_matrix, metric="euclidean", backend="cpu")
        assert_allclose(D, D.T, atol=1e-10)

    def test_zero_diagonal(self, dense_matrix):
        D = gades.distance(dense_matrix, metric="euclidean", backend="cpu")
        assert_allclose(np.diag(D), 0.0, atol=1e-5)

    def test_pairwise(self, dense_matrix):
        X = dense_matrix[:, :15]
        Y = dense_matrix[:, 15:]
        D = gades.pairwise_distance(X, Y, metric="euclidean", backend="cpu")
        assert D.shape == (15, 15)
        ref = scipy.spatial.distance.cdist(X.T, Y.T, metric="euclidean")
        assert_allclose(D, ref, rtol=1e-4, atol=1e-5)


# ── Sparse CPU tests ──────────────────────────────────────────────────────


class TestSparseCPU:
    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
    def test_basic_metrics(self, sparse_matrix, metric):
        X_dense = sparse_matrix.toarray()
        D = gades.distance(sparse_matrix, metric=metric, backend="cpu")
        ref = scipy_pdist(X_dense, metric)
        assert D.shape == (30, 30)
        assert_allclose(D, ref, rtol=1e-3, atol=1e-3)

    def test_pearson(self, sparse_matrix):
        X_dense = sparse_matrix.toarray()
        D = gades.distance(sparse_matrix, metric="pearson", backend="cpu")
        ref = scipy_pdist(X_dense, "pearson")
        assert_allclose(D, ref, rtol=1e-3, atol=1e-3)

    def test_spearman(self, sparse_matrix):
        X_dense = sparse_matrix.toarray()
        D = gades.distance(sparse_matrix, metric="spearman", backend="cpu")
        ref = scipy_pdist(X_dense, "spearman")
        assert_allclose(D, ref, rtol=5e-2, atol=5e-2)

    def test_kendall(self):
        # Sparse Kendall uses a zero-gap-aware kernel that differs from
        # scipy's dense kendalltau (which ignores sparsity structure).
        # Validate symmetry and zero diagonal instead.
        rng = np.random.default_rng(99)
        X = scipy.sparse.random(20, 8, density=0.4, format="csc", random_state=rng)
        D = gades.distance(X, metric="kendall", backend="cpu")
        assert D.shape == (8, 8)
        assert_allclose(D, D.T, atol=1e-10)
        assert_allclose(np.diag(D), 0.0, atol=1e-5)

    def test_sparse_pairwise(self, sparse_matrix):
        X = sparse_matrix[:, :15]
        Y = sparse_matrix[:, 15:]
        D = gades.pairwise_distance(X, Y, metric="euclidean", backend="cpu")
        assert D.shape == (15, 15)


# ── GPU tests ─────────────────────────────────────────────────────────────


@pytest.mark.gpu
class TestDenseGPU:
    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine", "pearson"])
    def test_basic_metrics(self, dense_matrix, metric):
        D = gades.distance(dense_matrix, metric=metric, backend="gpu")
        ref = scipy_pdist(dense_matrix, metric)
        assert D.shape == (30, 30)
        assert_allclose(D, ref, rtol=1e-3, atol=1e-3)

    def test_spearman(self, dense_matrix):
        D = gades.distance(dense_matrix, metric="spearman", backend="gpu")
        ref = scipy_pdist(dense_matrix, "spearman")
        assert_allclose(D, ref, rtol=1e-2, atol=1e-2)

    def test_kendall(self):
        rng = np.random.default_rng(123)
        X = rng.standard_normal((20, 8))
        D = gades.distance(X, metric="kendall", backend="gpu")
        ref = scipy_pdist(X, "kendall")
        assert_allclose(D, ref, rtol=1e-3, atol=1e-3)

    def test_gpu_cpu_agree(self, dense_matrix):
        for metric in gades.SUPPORTED_METRICS:
            D_gpu = gades.distance(dense_matrix, metric=metric, backend="gpu")
            D_cpu = gades.distance(dense_matrix, metric=metric, backend="cpu")
            assert_allclose(
                D_gpu,
                D_cpu,
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"GPU/CPU mismatch for {metric}",
            )


@pytest.mark.gpu
class TestSparseGPU:
    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
    def test_basic_metrics(self, sparse_matrix, metric):
        X_dense = sparse_matrix.toarray()
        D = gades.distance(sparse_matrix, metric=metric, backend="gpu")
        ref = scipy_pdist(X_dense, metric)
        assert_allclose(D, ref, rtol=1e-3, atol=1e-3)


# ── Edge cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_invalid_metric(self):
        X = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="Unknown metric"):
            gades.distance(X, metric="hamming")

    def test_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            gades.distance(np.array([1, 2, 3]))

    def test_two_columns(self):
        X = np.random.randn(50, 2)
        D = gades.distance(X, metric="euclidean", backend="cpu")
        assert D.shape == (2, 2)
        assert_allclose(D[0, 0], 0.0, atol=1e-5)
        assert_allclose(D[1, 1], 0.0, atol=1e-5)

    def test_csr_input_converted(self):
        X = scipy.sparse.random(50, 10, density=0.3, format="csr")
        D = gades.distance(X, metric="euclidean", backend="cpu")
        assert D.shape == (10, 10)

    def test_integer_input(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
        D = gades.distance(X, metric="euclidean", backend="cpu")
        assert D.dtype == np.float64

    def test_c_order_input(self):
        X = np.ascontiguousarray(np.random.randn(50, 10))
        D = gades.distance(X, metric="euclidean", backend="cpu")
        ref = scipy_pdist(X, "euclidean")
        assert_allclose(D, ref, rtol=1e-4, atol=1e-5)


# ── sparse input must be canonicalised before it reaches the kernels ───────


def _unsorted_row_slice(seed=0, n_genes=600, n_cells=120, density=0.4):
    """A CSC matrix whose row indices are not sorted, as fancy indexing yields.

    Selecting genes by variance (or any ranking) rather than by position is the
    ordinary way to get here, and it used to produce silently wrong distances.
    """
    rng = np.random.default_rng(seed)
    base = scipy.sparse.random(n_genes, n_cells, density=density, format="csc",
                               random_state=seed)
    base.data = np.ceil(base.data * 10)
    order = rng.permutation(n_genes)[: n_genes // 2]     # ranking order, not index order
    return base[order], base[np.sort(order)], order


def test_unsorted_sparse_indices_give_the_same_answer_as_sorted():
    unsorted, sorted_same_genes, _ = _unsorted_row_slice()
    assert not unsorted.has_sorted_indices                # precondition of the bug

    dense_ref = gades.distance(
        np.asfortranarray(np.asarray(unsorted.todense(), dtype=np.float64)),
        metric="kendall",
    )
    assert_allclose(gades.distance(unsorted, metric="kendall"), dense_ref, atol=1e-9)

    # and the caller's matrix is left exactly as it was handed over
    assert not unsorted.has_sorted_indices


@pytest.mark.parametrize("metric", ["kendall", "spearman", "euclidean", "cosine"])
def test_unsorted_matches_dense_for_every_metric(metric):
    unsorted, _, _ = _unsorted_row_slice(seed=3)
    dense = np.asfortranarray(np.asarray(unsorted.todense(), dtype=np.float64))
    assert_allclose(gades.distance(unsorted, metric=metric),
                    gades.distance(dense, metric=metric), atol=1e-5)


def test_kendall_distance_never_exceeds_one():
    """The discordant-pair fraction is bounded by 1; unsorted input broke that."""
    unsorted, _, _ = _unsorted_row_slice(seed=5, density=0.5)
    d = gades.distance(unsorted, metric="kendall")
    assert d.max() <= 1.0 + 1e-12
    assert d.min() >= 0.0


def test_duplicate_entries_are_summed_not_passed_through():
    rng = np.random.default_rng(11)
    n_genes, n_cells = 200, 40
    rows = rng.integers(0, n_genes, 3000)
    cols = rng.integers(0, n_cells, 3000)
    vals = rng.random(3000) * 5
    coo = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n_genes, n_cells))
    duped = scipy.sparse.csc_matrix(coo, copy=True)
    duped.sum_duplicates = lambda: None                  # keep the duplicates in place
    reference = gades.distance(
        np.asfortranarray(np.asarray(coo.todense(), dtype=np.float64)), metric="kendall"
    )
    assert_allclose(gades.distance(scipy.sparse.csc_matrix(coo), metric="kendall"),
                    reference, atol=1e-9)


def test_pairwise_also_canonicalises():
    unsorted, _, _ = _unsorted_row_slice(seed=7)
    left, right = unsorted[:, :60], unsorted[:, 60:]
    dense = np.asfortranarray(np.asarray(unsorted.todense(), dtype=np.float64))
    reference = gades.pairwise_distance(dense[:, :60], dense[:, 60:], metric="kendall")
    assert_allclose(gades.pairwise_distance(left, right, metric="kendall"),
                    reference, atol=1e-9)


# ── the C-level backstop, reached by bypassing the Python wrapper ──────────


def _call_sparse_abi(matrix, metric_code, backend):
    """Hand the raw CSC arrays to the C ABI, skipping _prepare_sparse."""
    lib = getattr(gades._backend, backend)
    fn = lib.gades_sparse_gpu if backend == "gpu" else lib.gades_sparse_cpu
    indices = np.ascontiguousarray(matrix.indices, dtype=np.int32)
    indptr = np.ascontiguousarray(matrix.indptr, dtype=np.int32)
    data = np.ascontiguousarray(matrix.data, dtype=np.float64)
    n, m = matrix.shape
    out = np.empty((m, m), dtype=np.float64, order="F")
    rc = fn(
        indices.ctypes.data_as(fn.argtypes[0]),
        indptr.ctypes.data_as(fn.argtypes[1]),
        data.ctypes.data_as(fn.argtypes[2]),
        out.ctypes.data_as(fn.argtypes[3]),
        n, m, matrix.nnz, metric_code,
    )
    assert rc == 0
    return np.ascontiguousarray(out), indices, data


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("metric_code,metric", [(5, "kendall"), (4, "spearman"),
                                                (3, "manhattan")])
def test_c_abi_repairs_unsorted_indices(backend, metric_code, metric):
    """The wrapper canonicalises, but the C ABI must not trust its caller."""
    if backend == "gpu" and not gades.has_gpu():
        pytest.skip("No CUDA GPU available")
    unsorted, _, _ = _unsorted_row_slice(seed=13)
    assert not unsorted.has_sorted_indices

    dense = np.asfortranarray(np.asarray(unsorted.todense(), dtype=np.float64))
    reference = gades.distance(dense, metric=metric, backend=backend)

    out, _, _ = _call_sparse_abi(unsorted, metric_code, backend)
    assert_allclose(out, reference, atol=1e-5)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_c_abi_does_not_reorder_the_callers_arrays(backend):
    """Repair happens in a private copy: R SEXP and NumPy buffers are not ours."""
    if backend == "gpu" and not gades.has_gpu():
        pytest.skip("No CUDA GPU available")
    unsorted, _, _ = _unsorted_row_slice(seed=17)
    before_idx = unsorted.indices.copy()
    before_val = unsorted.data.copy()

    _, indices, data = _call_sparse_abi(unsorted, 5, backend)

    assert np.array_equal(indices, before_idx)
    assert np.array_equal(data, before_val)
    assert np.array_equal(unsorted.indices, before_idx)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_c_abi_leaves_already_sorted_input_alone(backend):
    """The common path must be a check only -- same answer, no repair notice."""
    if backend == "gpu" and not gades.has_gpu():
        pytest.skip("No CUDA GPU available")
    _, sorted_matrix, _ = _unsorted_row_slice(seed=19)
    assert sorted_matrix.has_sorted_indices
    dense = np.asfortranarray(np.asarray(sorted_matrix.todense(), dtype=np.float64))
    out, _, _ = _call_sparse_abi(sorted_matrix, 5, backend)
    assert_allclose(out, gades.distance(dense, metric="kendall", backend=backend),
                    atol=1e-9)
