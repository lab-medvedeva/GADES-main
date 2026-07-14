# Shared setup loaded automatically by testthat::test_dir before each test file.
#
# END-TO-END: the suite exercises the INSTALLED package via library(GADES) — the
# real deploy path. `.onLoad` loads the native libs (mtrx_cpu always, mtrx/GPU
# optionally) via library.dynam + R_registerRoutines. This is NOT a raw
# dyn.load of lib/*.so + source(R/*.R); run `R CMD INSTALL .` before the suite.

suppressMessages({
  library(Matrix)
  library(hdf5r)
  library(glue)
  library(methods)
})

if (!requireNamespace("GADES", quietly = TRUE)) {
  stop("GADES is not installed — run `R CMD INSTALL .` first. The test suite ",
       "imports the package end-to-end (library(GADES)), not lib/*.so directly.")
}
suppressMessages(library(GADES))

# The suite calls unexported internals (process_pair_per_cell_pair,
# metric_default_value, sparse_writer_*, h5ad_*) by bare name; attach the package
# namespace so they resolve alongside the exported API from library(GADES).
if (!"GADES-internals" %in% search()) {
  attach(loadNamespace("GADES"), name = "GADES-internals", warn.conflicts = FALSE)
}

# --- Utility helpers ---

# Silence verbose `print()` calls inside legacy R functions.
silent <- function(expr) {
  tc <- textConnection("sinkbuf", "w", local = TRUE)
  sink(tc)
  on.exit({ sink(); close(tc) }, add = TRUE)
  eval.parent(substitute(expr))
}

# Write a synthetic h5ad file for tests. Returns path; caller is responsible
# for unlink(). Layout in args is genes × cells (Hobotnica convention);
# stored on disk as cells × genes (anndata convention).
write_synthetic_h5ad <- function(M_gxc, cell_names = NULL) {
  n_genes <- nrow(M_gxc)
  n_cells <- ncol(M_gxc)
  if (is.null(cell_names)) {
    cell_names <- paste0("cell_", seq_len(n_cells))
  }
  M_cxg_csr <- as(t(M_gxc), "RsparseMatrix")
  path <- tempfile(fileext = ".h5ad")
  f <- H5File$new(path, mode = "w")
  X <- f$create_group("X")
  h5attr(X, "encoding-type") <- "csr_matrix"
  h5attr(X, "encoding-version") <- "0.1.0"
  h5attr(X, "shape") <- as.integer(c(n_cells, n_genes))
  X[["data"]]    <- as.numeric(M_cxg_csr@x)
  X[["indices"]] <- as.integer(M_cxg_csr@j)
  X[["indptr"]]  <- as.integer(M_cxg_csr@p)
  obs <- f$create_group("obs")
  h5attr(obs, "encoding-type") <- "dataframe"
  h5attr(obs, "_index") <- "cell_id"
  obs[["cell_id"]] <- cell_names
  f$close_all()
  path
}

# Reconstruct a dense symmetric distance matrix from sparse HDF5 COO output.
read_back_dense <- function(path, default_value) {
  f <- H5File$new(path, mode = "r")
  shp <- as.integer(h5attr(f, "shape"))
  data <- f[["data"]][]
  row  <- f[["row"]][]
  col  <- f[["col"]][]
  f$close_all()
  m <- shp[1]
  fill <- if (is.na(default_value)) NaN else default_value
  M <- matrix(fill, nrow = m, ncol = m)
  diag(M) <- 0
  if (length(data) > 0) {
    for (k in seq_along(data)) {
      M[row[k] + 1L, col[k] + 1L] <- data[k]
      M[col[k] + 1L, row[k] + 1L] <- data[k]
    }
  }
  M
}

# Compute the in-memory per_cell_pair distance matrix (genes × cells), batched.
inmem_reference <- function(M_gxc, metric, type = "cpu", batch_size = 20L) {
  n_genes <- nrow(M_gxc)
  m <- ncol(M_gxc)
  M_csc <- as(M_gxc, "CsparseMatrix")
  out <- matrix(0, nrow = m, ncol = m)
  for (first_idx in seq(0L, m - 1L, by = batch_size)) {
    for (second_idx in seq(0L, m - 1L, by = batch_size)) {
      if (first_idx > second_idx) next
      first_end  <- min(first_idx + batch_size, m)
      second_end <- min(second_idx + batch_size, m)
      sa <- M_csc[, (first_idx + 1L):first_end, drop = FALSE]
      sb <- if (first_idx == second_idx) sa
            else M_csc[, (second_idx + 1L):second_end, drop = FALSE]
      block <- process_pair_per_cell_pair(sa, sb, n_genes, metric, type)
      out[(first_idx + 1L):first_end, (second_idx + 1L):second_end] <- block
      if (first_idx != second_idx) {
        out[(second_idx + 1L):second_end, (first_idx + 1L):first_end] <- t(block)
      }
    }
  }
  diag(out) <- 0
  out
}

# Skip a test if a specific real dataset isn't present.
skip_unless_dataset <- function(path, name) {
  if (!file.exists(path)) {
    skip(sprintf("dataset %s not present at %s", name, path))
  }
}
