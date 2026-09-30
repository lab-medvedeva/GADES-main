import numpy as np
import pytest
import scipy.sparse

import gades

h5py = pytest.importorskip("h5py")


# ── Fixtures ──────────────────────────────────────────────────────────────


def _write_h5ad(path, X, obs, encoding, layers=None):
    """Write a minimal but spec-conformant .h5ad, without needing anndata."""
    n_cells, n_genes = X.shape

    def write_matrix(group, name, matrix):
        if encoding == "array":
            dataset = group.create_dataset(name, data=np.asarray(matrix, dtype="f8"))
            dataset.attrs["encoding-type"] = "array"
            return
        sparse = (
            scipy.sparse.csr_matrix(matrix)
            if encoding == "csr_matrix"
            else scipy.sparse.csc_matrix(matrix)
        )
        node = group.create_group(name)
        node.attrs["encoding-type"] = encoding
        node.attrs["shape"] = np.array([n_cells, n_genes], dtype="i8")
        node.create_dataset("data", data=sparse.data.astype("f4"))
        node.create_dataset("indices", data=sparse.indices.astype("i4"))
        node.create_dataset("indptr", data=sparse.indptr.astype("i4"))

    with h5py.File(path, "w") as f:
        write_matrix(f, "X", X)

        group = f.create_group("layers")
        for name, matrix in (layers or {}).items():
            write_matrix(group, name, matrix)

        obs_group = f.create_group("obs")
        obs_group.attrs["_index"] = "_index"
        obs_group.create_dataset(
            "_index", data=np.array([f"cell{i}" for i in range(n_cells)], dtype="S16")
        )
        for name, values in obs.items():
            categories, codes = np.unique(values, return_inverse=True)
            node = obs_group.create_group(name)
            node.attrs["encoding-type"] = "categorical"
            node.create_dataset("categories", data=categories.astype("S32"))
            node.create_dataset("codes", data=codes.astype("i1"))

        var_group = f.create_group("var")
        var_group.attrs["_index"] = "_index"
        var_group.create_dataset(
            "_index", data=np.array([f"gene{j}" for j in range(n_genes)], dtype="S16")
        )


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    X = rng.poisson(1.2, size=(200, 60)).astype(np.float64)
    X[X > 6] = 0
    obs = {
        "cell_type": np.array(["A"] * 80 + ["B"] * 70 + ["C"] * 50),
        "batch": np.array(["b0", "b1"] * 100),
    }
    return X, obs


@pytest.fixture
def csr_file(tmp_path, dataset):
    X, obs = dataset
    path = tmp_path / "csr.h5ad"
    _write_h5ad(path, X, obs, "csr_matrix", layers={"corrected": X * 2})
    return path, X, obs


# ── The layout trick ──────────────────────────────────────────────────────


def test_csr_block_is_the_genes_by_cells_transpose(csr_file):
    """The whole design rests on this: read as CSC, a CSR block is its own transpose."""
    path, X, _ = csr_file
    with gades.H5adReader(path) as reader:
        block = reader.slice_cells(40, 90)
    assert scipy.sparse.issparse(block)
    assert block.shape == (X.shape[1], 50)
    np.testing.assert_allclose(block.toarray(), X[40:90].T)


def test_full_range_round_trips(csr_file):
    path, X, _ = csr_file
    with gades.H5adReader(path) as reader:
        np.testing.assert_allclose(reader.slice_cells(0, X.shape[0]).toarray(), X.T)


@pytest.mark.parametrize("encoding", ["csr_matrix", "csc_matrix", "array"])
def test_every_encoding_gives_the_same_answer(tmp_path, dataset, encoding):
    X, obs = dataset
    path = tmp_path / f"{encoding}.h5ad"
    _write_h5ad(path, X, obs, encoding)
    with gades.H5adReader(path) as reader:
        assert reader.encoding == encoding
        block = reader.slice_cells(10, 60)
        dense = block.toarray() if scipy.sparse.issparse(block) else block
        np.testing.assert_allclose(dense, X[10:60].T)


# ── Selection ─────────────────────────────────────────────────────────────


def test_gene_subset(csr_file):
    path, X, _ = csr_file
    genes = np.array([0, 5, 17, 42, 59])
    with gades.H5adReader(path) as reader:
        block = reader.slice_cells(0, 30, genes=genes)
    assert block.shape == (5, 30)
    np.testing.assert_allclose(block.toarray(), X[0:30][:, genes].T)


def test_take_cells_preserves_requested_order(csr_file):
    path, X, _ = csr_file
    picks = np.array([197, 3, 55, 3, 120])
    with gades.H5adReader(path) as reader:
        block = reader.take_cells(picks)
    np.testing.assert_allclose(block.toarray(), X[picks].T)


def test_take_cells_spanning_chunks(csr_file):
    path, X, _ = csr_file
    picks = np.array([1, 40, 99, 150, 199])
    with gades.H5adReader(path) as reader:
        block = reader.take_cells(picks, chunk=16)
    np.testing.assert_allclose(block.toarray(), X[picks].T)


def test_take_cells_matches_contiguous_slice(csr_file):
    path, X, _ = csr_file
    with gades.H5adReader(path) as reader:
        by_range = reader.slice_cells(20, 45).toarray()
        by_index = reader.take_cells(np.arange(20, 45)).toarray()
    np.testing.assert_allclose(by_range, by_index)


def test_to_dense_is_fortran_ordered_float64(csr_file):
    path, X, _ = csr_file
    with gades.H5adReader(path) as reader:
        dense = reader.to_dense(cells=(0, 25))
    assert dense.dtype == np.float64
    assert dense.flags.f_contiguous
    np.testing.assert_allclose(dense, X[0:25].T)


# ── Metadata ──────────────────────────────────────────────────────────────


def test_obs_is_read_without_touching_the_matrix(csr_file):
    path, _, obs = csr_file
    frame = gades.read_obs(path)
    assert list(frame["cell_type"]) == list(obs["cell_type"])
    assert list(frame["batch"]) == list(obs["batch"])
    assert list(frame.index[:2]) == ["cell0", "cell1"]


def test_obs_column_subset(csr_file):
    path, _, obs = csr_file
    frame = gades.read_obs(path, columns=["batch"])
    assert list(frame.columns) == ["batch"]


def test_info(csr_file):
    path, X, _ = csr_file
    info = gades.h5ad_info(path)
    assert info["shape"] == X.shape
    assert info["encoding"] == "csr_matrix"
    assert info["layers"] == ["corrected"]
    assert set(info["obs_columns"]) == {"_index", "cell_type", "batch"}


def test_names(csr_file):
    path, X, _ = csr_file
    with gades.H5adReader(path) as reader:
        assert list(reader.obs_names()[:2]) == ["cell0", "cell1"]
        assert list(reader.var_names()[:2]) == ["gene0", "gene1"]


def test_layer_is_read_instead_of_x(csr_file):
    path, X, _ = csr_file
    with gades.H5adReader(path, layer="corrected") as reader:
        np.testing.assert_allclose(reader.slice_cells(0, 20).toarray(), (X * 2)[0:20].T)


def test_missing_layer_lists_what_exists(csr_file):
    path, _, _ = csr_file
    with pytest.raises(KeyError, match="corrected"):
        gades.H5adReader(path, layer="nope")


# ── Distance integration ──────────────────────────────────────────────────


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "kendall"])
def test_distance_from_h5ad_matches_in_memory(csr_file, metric):
    path, X, _ = csr_file
    streamed = gades.distance_from_h5ad(path, metric=metric, cells=(0, 40))
    in_memory = gades.distance(np.asfortranarray(X[0:40].T), metric=metric)
    np.testing.assert_allclose(streamed, in_memory, atol=1e-5)


def test_distance_from_h5ad_on_selected_cells(csr_file):
    path, X, _ = csr_file
    picks = np.array([0, 10, 20, 30, 40, 50])
    streamed = gades.distance_from_h5ad(path, metric="euclidean", cells=picks)
    in_memory = gades.distance(np.asfortranarray(X[picks].T), metric="euclidean")
    np.testing.assert_allclose(streamed, in_memory, atol=1e-5)


def test_distance_from_h5ad_with_layer_and_genes(csr_file):
    path, X, _ = csr_file
    genes = np.arange(0, 60, 3)
    streamed = gades.distance_from_h5ad(
        path, metric="euclidean", cells=(0, 30), genes=genes, layer="corrected"
    )
    in_memory = gades.distance(
        np.asfortranarray((X * 2)[0:30][:, genes].T), metric="euclidean"
    )
    np.testing.assert_allclose(streamed, in_memory, atol=1e-4)


def test_oversized_output_is_refused_before_allocating():
    from gades.h5ad import _check_output_size

    _check_output_size(30_000)                       # 6.7 GiB, under the default
    with pytest.raises(MemoryError, match="GiB"):
        _check_output_size(40_000)                   # 11.9 GiB, over it


def test_output_guard_fires_through_the_public_api(csr_file):
    path, _, _ = csr_file
    with pytest.raises(MemoryError, match="subset"):
        gades.distance_from_h5ad(
            path, metric="euclidean", cells=(0, 200), max_output_bytes=1000
        )


# ── Validation ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("start, stop", [(-1, 10), (10, 10), (0, 999), (50, 20)])
def test_invalid_cell_range(csr_file, start, stop):
    path, _, _ = csr_file
    with gades.H5adReader(path) as reader:
        with pytest.raises(ValueError, match="invalid cell range"):
            reader.slice_cells(start, stop)


def test_out_of_range_indices(csr_file):
    path, _, _ = csr_file
    with gades.H5adReader(path) as reader:
        with pytest.raises(ValueError, match="out of range"):
            reader.take_cells([0, 1, 500])
        with pytest.raises(ValueError, match="empty"):
            reader.take_cells([])


def test_context_manager_closes_the_file(csr_file):
    path, _, _ = csr_file
    reader = gades.H5adReader(path)
    with reader:
        pass
    assert reader._file is None
