# End-to-end test: mtrx_distance(h5ad path) produces same result as
# in-memory mtrx_distance with sparse_layout="per_cell_pair".

run_mtrx <- function(h5ad_path, out_path, metric, type, batch_size = 20L) {
  silent({
    mtrx_distance(h5ad_path, filename = out_path, batch_size = batch_size,
                  metric = metric, type = type)
  })
}

test_that("h5ad streaming + sparse output matches in-memory reference (euclidean)", {
  set.seed(42)
  n_genes <- 200L; n_cells <- 60L
  nnz_target <- as.integer(0.05 * n_genes * n_cells)
  M <- matrix(0, nrow = n_genes, ncol = n_cells)
  M[sample(length(M), nnz_target)] <- rpois(nnz_target, lambda = 3) + 1L

  h5_path <- write_synthetic_h5ad(M)
  out_path <- tempfile(fileext = ".h5")
  on.exit({ unlink(h5_path); unlink(out_path) })

  run_mtrx(h5_path, out_path, "euclidean", "cpu", batch_size = 20L)

  got <- read_back_dense(out_path, 0.0)
  ref <- inmem_reference(M, "euclidean", "cpu", batch_size = 20L)
  expect_lt(max(abs(got - ref)), 1e-4)
})

test_that("h5ad streaming spearman: NaN structure preserved", {
  set.seed(99)
  n_genes <- 150L; n_cells <- 40L
  M <- matrix(0, nrow = n_genes, ncol = n_cells)
  M[sample(length(M), 400)] <- rpois(400, lambda = 2) + 1L
  M[, c(10, 25)] <- 0  # two empty cells

  h5_path <- write_synthetic_h5ad(M)
  out_path <- tempfile(fileext = ".h5")
  on.exit({ unlink(h5_path); unlink(out_path) })

  run_mtrx(h5_path, out_path, "spearman", "cpu", batch_size = 15L)

  got <- read_back_dense(out_path, NA_real_)
  ref <- inmem_reference(M, "spearman", "cpu", batch_size = 15L)

  # Finite entries match
  finite <- !is.nan(ref) & !is.nan(got)
  expect_lt(max(abs(got[finite] - ref[finite])), 1e-4)
  # NaN structure matches
  expect_equal(is.nan(got), is.nan(ref))
})

test_that("h5ad streaming pearson (GPU)", {
  set.seed(77)
  n_genes <- 180L; n_cells <- 50L
  M <- matrix(0, nrow = n_genes, ncol = n_cells)
  M[sample(length(M), 600)] <- rpois(600, lambda = 4) + 1L

  h5_path <- write_synthetic_h5ad(M)
  out_path <- tempfile(fileext = ".h5")
  on.exit({ unlink(h5_path); unlink(out_path) })

  run_mtrx(h5_path, out_path, "pearson", "gpu", batch_size = 25L)

  got <- read_back_dense(out_path, NA_real_)
  ref <- inmem_reference(M, "pearson", "gpu", batch_size = 25L)
  finite <- !is.nan(ref) & !is.nan(got)
  expect_lt(max(abs(got[finite] - ref[finite])), 1e-4)
})

test_that("h5ad streaming resume completes after partial run", {
  set.seed(55)
  n_genes <- 100L; n_cells <- 50L
  M <- matrix(0, nrow = n_genes, ncol = n_cells)
  M[sample(length(M), 500)] <- rpois(500, lambda = 2) + 1L

  h5_path <- write_synthetic_h5ad(M)
  out_path <- tempfile(fileext = ".h5")
  on.exit({ unlink(h5_path); unlink(out_path) })

  bs <- 15L

  # Phase 1: write 2 batches manually then close
  h <- h5ad_open(h5_path)
  obs <- h5ad_obs_names(h)
  w <- sparse_writer_open(out_path, n_cells, obs, "euclidean")
  done <- 0L
  for (first_idx in seq(0L, n_cells - 1L, by = bs)) {
    if (done >= 2L) break
    for (second_idx in seq(0L, n_cells - 1L, by = bs)) {
      if (first_idx > second_idx) next
      if (done >= 2L) break
      first_end  <- min(first_idx + bs, n_cells)
      second_end <- min(second_idx + bs, n_cells)
      sa <- h5ad_slice_cells(h, first_idx, first_end)
      sb <- if (first_idx == second_idx) sa
            else h5ad_slice_cells(h, second_idx, second_end)
      block <- process_pair_per_cell_pair(sa, sb, n_genes, "euclidean", "cpu")
      sparse_writer_append_block(w, block, first_idx, second_idx)
      sparse_writer_set_checkpoint(w, first_idx, second_idx)
      done <- done + 1L
    }
  }
  h5ad_close(h)
  sparse_writer_close(w)

  # Phase 2: call full pipeline — should resume and complete
  run_mtrx(h5_path, out_path, "euclidean", "cpu", batch_size = bs)

  got <- read_back_dense(out_path, 0.0)
  ref <- inmem_reference(M, "euclidean", "cpu", batch_size = bs)
  expect_lt(max(abs(got - ref)), 1e-4)
})

test_that("output HDF5 has correct schema (shape, metric, obs_names)", {
  set.seed(33)
  n_genes <- 80L; n_cells <- 25L
  M <- matrix(rpois(n_genes * n_cells, lambda = 0.5),
              nrow = n_genes, ncol = n_cells)
  custom_names <- paste0("test_", letters[1:n_cells])
  h5_path <- write_synthetic_h5ad(M, cell_names = custom_names)
  out_path <- tempfile(fileext = ".h5")
  on.exit({ unlink(h5_path); unlink(out_path) })

  run_mtrx(h5_path, out_path, "manhattan", "cpu", batch_size = 10L)

  f <- H5File$new(out_path, mode = "r")
  shape <- h5attr(f, "shape")
  metric <- h5attr(f, "metric")
  enc <- h5attr(f, "encoding-type")
  obs_out <- f[["obs_names"]][]
  f$close_all()

  expect_equal(as.integer(shape), c(n_cells, n_cells))
  expect_equal(metric, "manhattan")
  expect_equal(enc, "coo_matrix")
  expect_equal(obs_out, custom_names)
})

test_that("missing output path errors with helpful message", {
  set.seed(1)
  M <- matrix(rpois(60, lambda = 1), nrow = 20, ncol = 3)
  h5_path <- write_synthetic_h5ad(M)
  on.exit(unlink(h5_path))

  expect_error(
    silent({ mtrx_distance(h5_path, batch_size = 5L, metric = "euclidean") }),
    "filename"
  )
})
