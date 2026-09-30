"""Streaming reader for .h5ad files.

Reads cell blocks out of an AnnData file without loading it into memory, so the
size of the file stops being the constraint (the distance matrix becomes one).
A 9.4 GB, 1.06M-cell file streams 5000-cell blocks in ~0.2 s at well under 1 GB
of resident memory.

The reader talks to HDF5 directly rather than going through ``anndata``: it has
no version coupling (``backed='r'`` sparse slicing is broken in several
anndata/scipy combinations), and it is faster for contiguous blocks.

**The layout trick.** AnnData stores ``X`` as CSR over cells (cells x genes).
GADES wants genes x cells, and its sparse path wants CSC. A CSR block's
``(data, indices, indptr)`` triple *is already* the CSC representation of its
own transpose -- so the genes x cells matrix GADES needs comes out with no
sparse transpose and no copy. This mirrors ``R/h5ad_reader.R``.

Quick start::

    import gades

    info = gades.h5ad_info("atlas.h5ad")
    obs = gades.read_obs("atlas.h5ad", columns=["cell_type"])

    with gades.H5adReader("atlas.h5ad") as reader:
        block = reader.slice_cells(0, 5000)          # genes x cells, CSC
        D = gades.distance(block, metric="kendall")
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

__all__ = ["H5adReader", "h5ad_info", "read_obs", "distance_from_h5ad"]

_CHUNK = 8192


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - trivial
        raise ImportError(
            "reading .h5ad files requires h5py (pip install 'gades[anndata]')"
        ) from exc
    return h5py


def _decode(values):
    """Bytes -> str for HDF5 string arrays, other dtypes untouched."""
    values = np.asarray(values)
    if values.dtype.kind == "S":
        return values.astype(str)
    if values.dtype.kind == "O":
        return np.array([v.decode() if isinstance(v, bytes) else v for v in values])
    return values


def _read_dataframe_column(group, name):
    """Read one obs/var column, expanding anndata's categorical encoding."""
    node = group[name]
    if hasattr(node, "keys") and "codes" in node:            # categorical
        codes = node["codes"][:]
        categories = _decode(node["categories"][:])
        out = np.empty(len(codes), dtype=object)
        known = codes >= 0
        out[known] = categories[codes[known]]
        out[~known] = None
        return out
    return _decode(node[:])


def _index_key(group):
    key = group.attrs.get("_index", "_index")
    return key.decode() if isinstance(key, bytes) else key


class H5adReader:
    """Streaming access to the matrices of an .h5ad file.

    Parameters
    ----------
    path : str or pathlib.Path
    layer : str, optional
        Read ``layers/<layer>`` instead of ``X``.

    Notes
    -----
    Cell blocks come back as **genes x cells** -- the orientation
    :func:`gades.distance` expects, where columns are the objects being
    compared. ``indptr`` is cached on open (4-8 bytes per cell), so slicing is
    bounded by the ``data``/``indices`` reads alone.

    Usable as a context manager; otherwise call :meth:`close`.
    """

    def __init__(self, path, layer=None):
        h5py = _require_h5py()
        self.path = str(path)
        self.layer = layer
        self._file = h5py.File(self.path, "r")

        try:
            node = self._file["X"] if layer is None else self._file["layers"][layer]
        except KeyError as exc:
            available = list(self._file.get("layers", {}))
            self._file.close()
            raise KeyError(
                f"layer {layer!r} not found in {self.path}; available: {available}"
            ) from exc

        self._node = node
        self.encoding = node.attrs.get("encoding-type", "array")
        if isinstance(self.encoding, bytes):
            self.encoding = self.encoding.decode()

        if self.encoding in ("csr_matrix", "csc_matrix"):
            self.shape = tuple(int(v) for v in node.attrs["shape"])
            self._indptr = node["indptr"][:]
            self.dtype = node["data"].dtype
        else:
            self.shape = tuple(int(v) for v in node.shape)
            self._indptr = None
            self.dtype = node.dtype

        self.n_cells, self.n_genes = self.shape

    # ---------------------------------------------------------------- lifecycle

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        target = "X" if self.layer is None else f"layers/{self.layer}"
        return (
            f"H5adReader({self.path!r}, {target}, {self.n_cells} cells x "
            f"{self.n_genes} genes, {self.encoding})"
        )

    # ---------------------------------------------------------------- metadata

    @property
    def layers(self):
        """Names of the available layers."""
        return list(self._file.get("layers", {}))

    def obs_names(self):
        group = self._file["obs"]
        return _decode(group[_index_key(group)][:])

    def var_names(self):
        group = self._file["var"]
        return _decode(group[_index_key(group)][:])

    @property
    def obs_columns(self):
        return [k for k in self._file["obs"] if not k.startswith("__")]

    def obs_column(self, name):
        """One obs column as a numpy array, categoricals expanded to labels."""
        return _read_dataframe_column(self._file["obs"], name)

    def obs(self, columns=None, as_frame=True):
        """Read obs. Cheap -- it never touches the expression matrix."""
        columns = list(self.obs_columns) if columns is None else list(columns)
        data = {c: self.obs_column(c) for c in columns}
        if as_frame:
            try:
                import pandas as pd
            except ImportError:
                return data
            return pd.DataFrame(data, index=self.obs_names())
        return data

    # ---------------------------------------------------------------- reading

    def _csr_block(self, start, stop, out_dtype):
        """CSR rows [start, stop) returned as a CSC genes x cells matrix."""
        indptr = self._indptr[start : stop + 1]
        lo, hi = int(indptr[0]), int(indptr[-1])
        data = self._node["data"][lo:hi].astype(out_dtype, copy=False)
        indices = self._node["indices"][lo:hi]

        # The CSR triple of a cells x genes block is, read as CSC, exactly its
        # genes x cells transpose. No sparse transpose, no copy.
        return scipy.sparse.csc_matrix(
            (data, indices, indptr - lo), shape=(self.n_genes, stop - start)
        )

    def slice_cells(self, start, stop, genes=None, dtype=np.float64):
        """A contiguous block of cells as **genes x cells**.

        Parameters
        ----------
        start, stop : int
            Half-open cell range.
        genes : array-like, optional
            Gene indices to keep (positional).
        dtype : numpy dtype
            GADES computes in float64; leave as is unless you are only
            inspecting the data.

        Returns
        -------
        scipy.sparse.csc_matrix or numpy.ndarray, shape (n_genes, stop - start)
        """
        if not 0 <= start < stop <= self.n_cells:
            raise ValueError(
                f"invalid cell range [{start}, {stop}) for {self.n_cells} cells"
            )

        if self.encoding == "csr_matrix":
            block = self._csr_block(start, stop, dtype)
        elif self.encoding == "csc_matrix":
            # indptr runs over genes here, so cells cannot be sliced cheaply.
            full = scipy.sparse.csc_matrix(
                (
                    self._node["data"][:].astype(dtype, copy=False),
                    self._node["indices"][:],
                    self._indptr,
                ),
                shape=(self.n_cells, self.n_genes),
            )
            block = full[start:stop].T.tocsc()
        else:
            block = np.asarray(self._node[start:stop], dtype=dtype).T

        if genes is not None:
            genes = np.asarray(genes)
            block = block[genes]
        return block

    def take_cells(self, indices, genes=None, dtype=np.float64, chunk=_CHUNK):
        """Arbitrary (non-contiguous) cells as **genes x cells**.

        Reads in contiguous chunks and filters, rather than seeking per row --
        scattered single-row reads on a large CSR are orders of magnitude
        slower. Output columns follow the order of ``indices``.
        """
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            raise ValueError("indices is empty")
        if indices.min() < 0 or indices.max() >= self.n_cells:
            raise ValueError(
                f"cell indices out of range [0, {self.n_cells})"
            )

        order = np.argsort(indices, kind="stable")
        wanted = indices[order]

        pieces = []
        for lo in range(int(wanted[0]), int(wanted[-1]) + 1, chunk):
            hi = min(lo + chunk, self.n_cells)
            local = wanted[(wanted >= lo) & (wanted < hi)]
            if local.size == 0:
                continue
            block = self.slice_cells(lo, hi, genes=genes, dtype=dtype)
            pieces.append(block[:, local - lo])

        stacked = (
            scipy.sparse.hstack(pieces, format="csc")
            if scipy.sparse.issparse(pieces[0])
            else np.hstack(pieces)
        )

        # Undo the sort so the caller gets the order they asked for.
        restore = np.empty_like(order)
        restore[order] = np.arange(len(order))
        return stacked[:, restore]

    def to_dense(self, cells=None, genes=None, dtype=np.float64):
        """Dense **genes x cells**, Fortran-ordered, ready for GADES.

        ``cells`` may be ``None`` (all), a ``(start, stop)`` tuple, or an array
        of indices.
        """
        if cells is None:
            block = self.slice_cells(0, self.n_cells, genes=genes, dtype=dtype)
        elif isinstance(cells, tuple) and len(cells) == 2:
            block = self.slice_cells(cells[0], cells[1], genes=genes, dtype=dtype)
        else:
            block = self.take_cells(cells, genes=genes, dtype=dtype)

        if scipy.sparse.issparse(block):
            block = block.toarray()
        return np.asfortranarray(block, dtype=dtype)


def h5ad_info(path, layer=None):
    """Shape, encoding, layers and obs columns of an .h5ad, without reading it."""
    with H5adReader(path, layer=layer) as reader:
        return {
            "path": reader.path,
            "shape": reader.shape,
            "n_cells": reader.n_cells,
            "n_genes": reader.n_genes,
            "encoding": reader.encoding,
            "dtype": np.dtype(reader.dtype).name,
            "layers": reader.layers,
            "obs_columns": reader.obs_columns,
        }


def read_obs(path, columns=None, as_frame=True):
    """Read obs from an .h5ad without touching the expression matrix."""
    with H5adReader(path) as reader:
        return reader.obs(columns=columns, as_frame=as_frame)


_DEFAULT_MAX_OUTPUT = 8 * 1024 ** 3


def _check_output_size(n_cells, max_bytes=_DEFAULT_MAX_OUTPUT):
    """Refuse an n x n float64 output larger than ``max_bytes``.

    Streaming makes the input size a non-issue, which makes it easy to ask for
    a distance matrix that cannot possibly fit. Fail before allocating.
    """
    needed = n_cells * n_cells * 8
    if needed > max_bytes:
        raise MemoryError(
            f"the distance matrix for {n_cells} cells would need "
            f"{needed / 1024 ** 3:.1f} GiB (limit {max_bytes / 1024 ** 3:.1f} GiB); "
            "subset `cells` first, or raise max_output_bytes"
        )


def distance_from_h5ad(
    path,
    metric="euclidean",
    cells=None,
    genes=None,
    layer=None,
    backend="auto",
    dense=False,
    max_output_bytes=_DEFAULT_MAX_OUTPUT,
):
    """Stream an .h5ad and compute the pairwise distance matrix between cells.

    Parameters
    ----------
    path : str or pathlib.Path
    metric : str
        See :data:`gades.METRICS`.
    cells : None, (start, stop) tuple, or array of indices
        Which cells to compare. ``None`` uses all of them -- check the size of
        the result first, it is ``n_cells**2 * 8`` bytes.
    genes : array-like, optional
        Positional gene subset.
    layer : str, optional
        Read ``layers/<layer>`` instead of ``X``.
    backend : {'auto', 'gpu', 'cpu'}
    dense : bool
        Densify before computing. The sparse path is usually preferable; this
        is here for metrics or shapes where the dense kernels win.
    max_output_bytes : int
        Guard against asking for a distance matrix that cannot fit.

    Returns
    -------
    numpy.ndarray of shape (n_selected_cells, n_selected_cells)
    """
    from .distance import distance

    with H5adReader(path, layer=layer) as reader:
        if dense:
            matrix = reader.to_dense(cells=cells, genes=genes)
        elif cells is None:
            matrix = reader.slice_cells(0, reader.n_cells, genes=genes)
        elif isinstance(cells, tuple) and len(cells) == 2:
            matrix = reader.slice_cells(cells[0], cells[1], genes=genes)
        else:
            matrix = reader.take_cells(cells, genes=genes)

    _check_output_size(matrix.shape[1], max_bytes=max_output_bytes)
    return distance(matrix, metric=metric, backend=backend)
