#' Open h5ad file for streaming.
#'
#' Opens the file in read-only mode, caches `indptr` (small: 4 bytes × n_cells)
#' so per-batch slice operations are I/O-bounded by `data`/`indices` only.
#' Verifies that X is `csr_matrix`-encoded (cells × genes), which is the
#' standard anndata convention.
#'
#' @param path Path to .h5ad file.
#' @return A handle list — pass to `h5ad_slice_cells()`, `h5ad_obs_names()`,
#'   `h5ad_close()`.
#' @export
h5ad_open <- function(path) {
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    stop("hdf5r is required to read h5ad files")
  }
  f <- hdf5r::H5File$new(path, mode = "r")
  x <- f[["X"]]
  enc <- hdf5r::h5attr(x, "encoding-type")
  if (enc != "csr_matrix") {
    f$close_all()
    stop(sprintf("h5ad X must be 'csr_matrix' encoded; got '%s'", enc))
  }
  shape <- hdf5r::h5attr(x, "shape")
  n_cells <- as.integer(shape[1])
  n_genes <- as.integer(shape[2])
  indptr <- x[["indptr"]][1:(n_cells + 1)]
  list(
    file = f,
    n_cells = n_cells,
    n_genes = n_genes,
    indptr = indptr,
    path = path
  )
}

#' Close h5ad handle.
#' @param handle From `h5ad_open()`.
#' @export
h5ad_close <- function(handle) {
  handle$file$close_all()
  invisible(NULL)
}

#' Read cell names (obs_names) from h5ad.
#'
#' Resolves the `_index` attribute on `/obs` (anndata convention) and reads
#' the named dataset. Falls back to `obs/_index` if attribute is missing.
#'
#' @param handle From `h5ad_open()`.
#' @return Character vector of length `n_cells`.
#' @export
h5ad_obs_names <- function(handle) {
  obs <- handle$file[["obs"]]
  attrs <- hdf5r::h5attributes(obs)
  idx_key <- attrs[["_index"]]
  if (is.null(idx_key)) idx_key <- "_index"
  obs[[idx_key]][1:handle$n_cells]
}

#' Read a batch of cells from h5ad as a sparse `dgCMatrix`.
#'
#' Reads only the relevant slice of `data` / `indices` (and pre-cached
#' `indptr`). The result is a CSC matrix of shape `(n_genes, batch_size)`,
#' matching Hobotnica's `count_matrix` layout (genes × cells), suitable for
#' direct use in `process_batch()` with `sparse_layout = "per_cell_pair"`.
#'
#' Key trick: h5ad's CSR-by-cells maps directly to CSC-by-cells for the
#' transposed (genes × cells) view. The arrays `indptr`, `indices`, `data`
#' become `@p`, `@i`, `@x` of the result `dgCMatrix` with only offset
#' adjustment — no sparse transpose required.
#'
#' @param handle From `h5ad_open()`.
#' @param start 0-based first cell index (inclusive).
#' @param end 0-based last cell index (exclusive).
#' @return A `dgCMatrix` of shape `(n_genes, end - start)`.
#' @export
h5ad_slice_cells <- function(handle, start, end) {
  if (start < 0 || end > handle$n_cells || start >= end) {
    stop(sprintf("invalid slice: start=%d, end=%d, n_cells=%d",
                 start, end, handle$n_cells))
  }
  start_off <- handle$indptr[start + 1]
  end_off   <- handle$indptr[end + 1]
  nnz       <- end_off - start_off
  n_batch   <- end - start

  x_group <- handle$file[["X"]]
  if (nnz > 0) {
    data_vals <- x_group[["data"]][(start_off + 1):end_off]
    gene_idx  <- x_group[["indices"]][(start_off + 1):end_off]
  } else {
    data_vals <- numeric(0)
    gene_idx  <- integer(0)
  }
  p_slot <- as.integer(handle$indptr[(start + 1):(end + 1)] - start_off)

  methods::new("dgCMatrix",
    p = p_slot,
    i = as.integer(gene_idx),
    x = as.numeric(data_vals),
    Dim = c(handle$n_genes, as.integer(n_batch))
  )
}
