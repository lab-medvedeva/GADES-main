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
