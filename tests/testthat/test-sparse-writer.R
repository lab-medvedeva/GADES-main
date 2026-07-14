# Sparse HDF5 writer correctness.
# Tests filter predicate, COO append, checkpoint, resume mismatch detection.

test_that("metric_default_value returns expected per-metric constants", {
  expect_equal(metric_default_value("euclidean"), 0.0)
  expect_equal(metric_default_value("manhattan"), 0.0)
  expect_equal(metric_default_value("kendall"),   0.0)
  expect_true(is.na(metric_default_value("cosine")))
  expect_true(is.na(metric_default_value("pearson")))
  expect_true(is.na(metric_default_value("spearman")))
  expect_error(metric_default_value("bogus"), "unknown metric")
})

test_that("single-batch euclidean output roundtrips", {
  set.seed(7)
  n <- 40; m <- 8
  M <- matrix(rpois(n * m, lambda = 0.6), nrow = n, ncol = m)
  ref <- as.matrix(dist(t(M), method = "euclidean"))

  path <- tempfile(fileext = ".h5")
  on.exit(unlink(path))
  obs <- paste0("c_", seq_len(m))

  M_csc <- as(M, "CsparseMatrix")
  h <- sparse_writer_open(path, m, obs, "euclidean")
  r <- silent({
    process_batch_cpu(M_csc, 0, 0, m, "euclidean",
                      sparse = TRUE, sparse_layout = "per_cell_pair")
  })
  n_written <- sparse_writer_append_block(h, r$correlation_matrix, 0, 0)
  sparse_writer_close(h)

  expect_equal(n_written, m * (m - 1) / 2)
  got <- read_back_dense(path, 0.0)
  expect_lt(max(abs(got - ref)), 1e-4)
})

test_that("multi-batch (diag + off-diag) matches single-batch", {
  set.seed(11)
  n <- 60; m <- 12
  M <- matrix(rpois(n * m, lambda = 0.5), nrow = n, ncol = m)
  M_csc <- as(M, "CsparseMatrix")
  ref <- inmem_reference(M, "euclidean", "cpu", batch_size = m)  # single batch

  path <- tempfile(fileext = ".h5")
  on.exit(unlink(path))
  obs <- paste0("c_", seq_len(m))
  bs <- 4L

  h <- sparse_writer_open(path, m, obs, "euclidean")
  for (first_idx in seq(0L, m - 1L, by = bs)) {
    for (second_idx in seq(0L, m - 1L, by = bs)) {
      if (first_idx > second_idx) next
      first_end <- min(first_idx + bs, m)
      second_end <- min(second_idx + bs, m)
      sa <- M_csc[, (first_idx + 1L):first_end, drop = FALSE]
      sb <- if (first_idx == second_idx) sa
            else M_csc[, (second_idx + 1L):second_end, drop = FALSE]
      block <- process_pair_per_cell_pair(sa, sb, n, "euclidean", "cpu")
      sparse_writer_append_block(h, block, first_idx, second_idx)
    }
  }
  sparse_writer_close(h)

  got <- read_back_dense(path, 0.0)
  expect_lt(max(abs(got - ref)), 1e-5)
})

test_that("NaN filtering: empty cell produces no triplets", {
  set.seed(13)
  n <- 30; m <- 6
  M <- matrix(rpois(n * m, lambda = 0.7), nrow = n, ncol = m)
  M[, 4] <- 0  # cell 4 (1-based) empty ⇒ all-NaN row for Spearman

  M_csc <- as(M, "CsparseMatrix")
  path <- tempfile(fileext = ".h5")
  on.exit(unlink(path))
  obs <- paste0("c_", seq_len(m))

  h <- sparse_writer_open(path, m, obs, "spearman")
  r <- silent({
    process_batch_cpu(M_csc, 0, 0, m, "spearman",
                      sparse = TRUE, sparse_layout = "per_cell_pair")
  })
  sparse_writer_append_block(h, r$correlation_matrix, 0, 0)
  sparse_writer_close(h)

  # 0-based idx 3 = cell 4 in 1-based; should not appear in row/col
  f <- H5File$new(path, mode = "r")
  rows <- f[["row"]][]
  cols <- f[["col"]][]
  f$close_all()
  expect_false(any(rows == 3L))
  expect_false(any(cols == 3L))
})

test_that("checkpoint attrs survive close/reopen", {
  m <- 5L
  path <- tempfile(fileext = ".h5")
  on.exit(unlink(path))
  obs <- paste0("c_", seq_len(m))

  h <- sparse_writer_open(path, m, obs, "euclidean")
  sparse_writer_set_checkpoint(h, 2L, 4L)
  sparse_writer_close(h)

  h <- sparse_writer_open(path, m, obs, "euclidean")
  chk <- sparse_writer_get_checkpoint(h)
  sparse_writer_close(h)

  expect_equal(chk, c(2L, 4L))
})

test_that("resume with mismatched metric / shape errors out", {
  m <- 5L
  path <- tempfile(fileext = ".h5")
  on.exit(unlink(path))
  obs <- paste0("c_", seq_len(m))

  h <- sparse_writer_open(path, m, obs, "euclidean")
  sparse_writer_close(h)

  expect_error(sparse_writer_open(path, m, obs, "manhattan"), "metric")
  expect_error(sparse_writer_open(path, m + 1L, c(obs, "extra"), "euclidean"),
               "shape")
})

test_that("diagonal-only block keeps strict upper triangle", {
  # block at (first=second=0), 4x4 dense with all entries = 1.0
  m <- 4L
  path <- tempfile(fileext = ".h5")
  on.exit(unlink(path))
  h <- sparse_writer_open(path, m, paste0("c_", seq_len(m)), "euclidean")
  block <- matrix(1.0, nrow = m, ncol = m)
  diag(block) <- 0  # self-distances are 0 → filtered by != default
  n_kept <- sparse_writer_append_block(h, block, 0L, 0L)
  sparse_writer_close(h)
  # Strict upper triangle of 4x4 = 6 entries
  expect_equal(n_kept, 6L)

  f <- H5File$new(path, mode = "r")
  rows <- f[["row"]][]
  cols <- f[["col"]][]
  f$close_all()
  expect_true(all(rows < cols))  # strict upper
})
