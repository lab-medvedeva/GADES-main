# Sparse HDF5 COO writer for Hobotnica distance results.
# See docs/adr/0001-sparse-output-hdf5-coo.md and CONTEXT.md.

#' Metric-specific default value (the constant kernel returns for empty-empty
#' pairs). Per CONTEXT.md: saved triplets must satisfy
#' `!is.nan(r) && r != default_value`. For Cosine/Pearson/Spearman the kernel
#' returns NaN for empty pairs, so there's no numerical default — `NA_real_`
#' marks this case; predicate degenerates to `!is.nan(r)`.
#'
#' @param metric Metric name (one of `euclidean`, `manhattan`, `kendall`,
#'   `cosine`, `pearson`, `spearman`).
#' @return Numeric scalar (or `NA_real_` for NaN-default metrics).
#' @export
metric_default_value <- function(metric) {
  switch(metric,
    "euclidean" = 0.0,
    "manhattan" = 0.0,
    "kendall"   = 0.0,
    "cosine"    = NA_real_,
    "pearson"   = NA_real_,
    "spearman"  = NA_real_,
    stop(sprintf("unknown metric: %s", metric))
  )
}

#' Open a sparse HDF5 output file for distance results.
#'
#' Creates (or resumes) an extendable HDF5 file with COO triplets:
#'   `/data` (float32), `/row` (int32), `/col` (int32), `/obs_names` (string),
#'   root attrs `shape`, `metric`, `encoding-type`, `last_first_idx`,
#'   `last_second_idx`.
#'
#' If the file already exists, this resumes: shape and metric must match,
#' otherwise an error is raised. Use `sparse_writer_get_checkpoint()` to
#' query where to continue from.
#'
#' @param path Output file path (.h5).
#' @param m Total number of cells (matrix is m × m).
#' @param obs_names Character vector of length `m`.
#' @param metric Metric name (for `metric_default_value`).
#' @param chunk_size Chunk size for COO arrays. Default 100k entries.
#' @return A handle list — pass to other `sparse_writer_*` functions.
#' @export
sparse_writer_open <- function(path, m, obs_names, metric, chunk_size = 100000L) {
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    stop("hdf5r is required to write sparse HDF5 output")
  }
  stopifnot(length(obs_names) == m)

  resume <- file.exists(path)
  if (resume) {
    f <- hdf5r::H5File$new(path, mode = "r+")
    attrs <- hdf5r::h5attributes(f)
    if (!identical(as.integer(attrs[["shape"]]), as.integer(c(m, m)))) {
      f$close_all()
      stop(sprintf("resume mismatch: existing shape = %s, expected = (%d, %d)",
                   paste(attrs[["shape"]], collapse = ","), m, m))
    }
    if (!identical(attrs[["metric"]], metric)) {
      f$close_all()
      stop(sprintf("resume mismatch: existing metric = '%s', expected = '%s'",
                   attrs[["metric"]], metric))
    }
    ds_data <- f[["data"]]
    ds_row  <- f[["row"]]
    ds_col  <- f[["col"]]
  } else {
    f <- hdf5r::H5File$new(path, mode = "w")
    hdf5r::h5attr(f, "shape") <- as.integer(c(m, m))
    hdf5r::h5attr(f, "metric") <- metric
    hdf5r::h5attr(f, "encoding-type") <- "coo_matrix"
    hdf5r::h5attr(f, "encoding-version") <- "0.1.0"
    hdf5r::h5attr(f, "last_first_idx")  <- -1L
    hdf5r::h5attr(f, "last_second_idx") <- -1L

    # obs_names as fixed dataset
    f[["obs_names"]] <- obs_names

    # Extensible COO datasets
    space_extendable <- function() {
      hdf5r::H5S$new("simple", dims = 0L, maxdims = Inf)
    }
    ds_data <- f$create_dataset(
      "data",
      dtype = hdf5r::h5types$H5T_NATIVE_FLOAT,
      space = space_extendable(),
      chunk_dims = as.integer(chunk_size)
    )
    ds_row <- f$create_dataset(
      "row",
      dtype = hdf5r::h5types$H5T_NATIVE_INT32,
      space = space_extendable(),
      chunk_dims = as.integer(chunk_size)
    )
    ds_col <- f$create_dataset(
      "col",
      dtype = hdf5r::h5types$H5T_NATIVE_INT32,
      space = space_extendable(),
      chunk_dims = as.integer(chunk_size)
    )
  }

  list(
    file = f,
    ds_data = ds_data,
    ds_row = ds_row,
    ds_col = ds_col,
    m = as.integer(m),
    metric = metric,
    default_value = metric_default_value(metric),
    path = path
  )
}

#' Save predicate per CONTEXT.md.
#'
#' @param r Numeric vector of metric values.
#' @param default_value `NA_real_` for NaN-default metrics, or a scalar.
#' @return Logical vector.
#' @keywords internal
.save_mask <- function(r, default_value) {
  if (is.na(default_value)) {
    !is.nan(r)
  } else {
    !is.nan(r) & r != default_value
  }
}

#' Filter a dense distance block to COO triplets and append.
#'
#' Caller passes the dense `m_a × m_b` block produced by `process_batch()`
#' along with its global position (`first_idx`, `second_idx` — 0-based).
#' For diagonal batches (first_idx == second_idx) only the strict upper
#' triangle (i < j) is kept. For off-diagonal upper batches all entries are
#' kept (they map to global i < j automatically since first_idx < second_idx).
#'
#' @param handle From `sparse_writer_open()`.
#' @param block Dense matrix (m_a × m_b) of metric values.
#' @param first_idx 0-based global column index of `block[,1]`.
#' @param second_idx 0-based global column index of `block[1,]`.
#' @export
sparse_writer_append_block <- function(handle, block, first_idx, second_idx) {
  m_a <- nrow(block)
  m_b <- ncol(block)
  if (first_idx > second_idx) {
    stop("sparse_writer_append_block requires first_idx <= second_idx (upper triangle)")
  }

  # Build (row, col) indices for every cell of the block, in 0-based globals.
  # block is column-major in R: block[i, j] sits at vector index (j-1)*m_a + i
  row_local <- rep(seq_len(m_a) - 1L, times = m_b)        # 0-based local row
  col_local <- rep(seq_len(m_b) - 1L, each  = m_a)        # 0-based local col
  global_row <- as.integer(first_idx) + row_local
  global_col <- as.integer(second_idx) + col_local
  vals <- as.numeric(block)

  if (first_idx == second_idx) {
    # Strict upper triangle within this diagonal block
    keep <- global_row < global_col
  } else {
    keep <- rep(TRUE, length(vals))
  }
  keep <- keep & .save_mask(vals, handle$default_value)

  if (!any(keep)) return(invisible(0L))

  r <- global_row[keep]
  c <- global_col[keep]
  v <- vals[keep]
  n_new <- length(v)

  cur <- handle$ds_data$dims
  new <- cur + n_new
  handle$ds_data$set_extent(new)
  handle$ds_row$set_extent(new)
  handle$ds_col$set_extent(new)

  handle$ds_data[(cur + 1L):new] <- as.numeric(v)
  handle$ds_row [(cur + 1L):new] <- as.integer(r)
  handle$ds_col [(cur + 1L):new] <- as.integer(c)

  invisible(n_new)
}

#' Persist checkpoint indices (last batch fully processed).
#'
#' Updates root attrs `last_first_idx` / `last_second_idx`. Cheap (single attr
#' write); call after each batch in the loop.
#' @export
sparse_writer_set_checkpoint <- function(handle, last_first_idx, last_second_idx) {
  hdf5r::h5attr(handle$file, "last_first_idx") <- as.integer(last_first_idx)
  hdf5r::h5attr(handle$file, "last_second_idx") <- as.integer(last_second_idx)
  invisible(NULL)
}

#' Read checkpoint from output file (used on resume).
#' @return Integer vector `c(last_first_idx, last_second_idx)`; both `-1` if
#'   the file was just created.
#' @export
sparse_writer_get_checkpoint <- function(handle) {
  attrs <- hdf5r::h5attributes(handle$file)
  c(as.integer(attrs[["last_first_idx"]]),
    as.integer(attrs[["last_second_idx"]]))
}

#' Close writer handle (flushes HDF5).
#' @export
sparse_writer_close <- function(handle) {
  handle$file$close_all()
  invisible(NULL)
}
