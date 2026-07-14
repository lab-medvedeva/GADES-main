# h5ad streaming reader correctness.
# Mostly uses a synthetic h5ad (cheap to regenerate). One optional test against
# the real TS_Lymph_Node.h5ad file is gated on its presence.

TS_PATH <- "/8tbsata/Science/BioInfo/Hobotnica/Hobotnica-GPU/Datasets/TS_Lymph_Node.h5ad"

test_that("h5ad_open exposes dims and cached indptr", {
  set.seed(1)
  M <- matrix(rpois(200 * 30, lambda = 0.4), nrow = 200, ncol = 30)
  path <- write_synthetic_h5ad(M)
  on.exit(unlink(path))

  h <- h5ad_open(path)
  on.exit(h5ad_close(h), add = TRUE)

  expect_equal(h$n_cells, 30L)
  expect_equal(h$n_genes, 200L)
  expect_equal(length(h$indptr), 31L)
  expect_equal(h$indptr[1], 0)
})

test_that("h5ad_slice_cells returns dgCMatrix with correct dims and data", {
  set.seed(2)
  n_genes <- 100L; n_cells <- 25L
  M <- matrix(0, nrow = n_genes, ncol = n_cells)
  M[sample(length(M), 200)] <- rpois(200, lambda = 4) + 1L
  path <- write_synthetic_h5ad(M)
  on.exit(unlink(path))

  h <- h5ad_open(path)
  on.exit(h5ad_close(h), add = TRUE)

  slc <- h5ad_slice_cells(h, 5L, 15L)  # cells 5..14
  expect_s4_class(slc, "dgCMatrix")
  expect_equal(nrow(slc), n_genes)
  expect_equal(ncol(slc), 10L)

  # Reconstruct dense and compare to source slice
  got_dense <- as.matrix(slc)
  expect_equal(got_dense, M[, 6:15],
               ignore_attr = TRUE,
               tolerance = 0)
})

test_that("h5ad_obs_names reads cell index", {
  set.seed(3)
  M <- matrix(rpois(50 * 7, lambda = 0.5), nrow = 50, ncol = 7)
  names_in <- paste0("custom_", letters[1:7])
  path <- write_synthetic_h5ad(M, cell_names = names_in)
  on.exit(unlink(path))

  h <- h5ad_open(path)
  on.exit(h5ad_close(h), add = TRUE)

  expect_equal(h5ad_obs_names(h), names_in)
})

test_that("invalid slice range raises", {
  set.seed(4)
  M <- matrix(rpois(50 * 5, lambda = 0.5), nrow = 50, ncol = 5)
  path <- write_synthetic_h5ad(M)
  on.exit(unlink(path))

  h <- h5ad_open(path)
  on.exit(h5ad_close(h), add = TRUE)

  expect_error(h5ad_slice_cells(h, 0L, 0L), "invalid slice")
  expect_error(h5ad_slice_cells(h, -1L, 3L), "invalid slice")
  expect_error(h5ad_slice_cells(h, 0L, 6L), "invalid slice")
})

test_that("real h5ad file (TS_Lymph_Node) opens and slices correctly", {
  skip_unless_dataset(TS_PATH, "TS_Lymph_Node.h5ad")
  h <- h5ad_open(TS_PATH)
  on.exit(h5ad_close(h), add = TRUE)

  expect_equal(h$n_cells, 53275L)
  expect_equal(h$n_genes, 58870L)

  slc <- h5ad_slice_cells(h, 0L, 50L)
  expect_s4_class(slc, "dgCMatrix")
  expect_equal(ncol(slc), 50L)
  expect_equal(nrow(slc), 58870L)
  expect_gt(length(slc@x), 0)  # non-trivial nnz
})
