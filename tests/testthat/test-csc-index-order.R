# The sparse kernels walk each column's nonzeros in index order: the rank
# metrics merge two columns' lists, the distance metrics account for the zero
# gaps between consecutive nonzeros. Neither is meaningful if the row indices
# inside a column are unsorted, and the failure is silent -- Kendall returned
# values above 1 for a metric bounded by 1, and nothing crashed.
#
# R's Matrix canonicalises on construction, so the R path does not reach the C
# ABI with unsorted indices by accident (first test). SciPy does not, which is
# how the bug surfaced. The C boundary therefore repairs whatever it is handed,
# and the remaining tests check that by calling .Call directly with deliberately
# scrambled slots -- going through mtrx_distance would not do, because its
# as(x, "CsparseMatrix") step could sanitise the input before C ever sees it.

kendall_ref <- function(M) {                    # discordant fraction, pure R
  n <- nrow(M); m <- ncol(M); D <- matrix(0, m, m)
  for (a in seq_len(m - 1)) for (b in (a + 1):m) {
    disc <- sum((outer(M[, a], M[, a], "-") * outer(M[, b], M[, b], "-")) < 0) / 2
    d <- disc * 2 / (n * (n - 1)); D[a, b] <- d; D[b, a] <- d
  }
  D
}

column_indices_sorted <- function(p, i, n_cols) {
  all(vapply(seq_len(n_cols), function(col) {
    idx <- i[(p[col] + 1L):p[col + 1L]]
    length(idx) < 2L || !is.unsorted(idx, strictly = TRUE)
  }, logical(1)))
}

# Shuffle row indices within every column, carrying the values along: the same
# data, the same nnz, a violated layout contract.
scramble <- function(p, i, x, n_cols, seed = 11) {
  set.seed(seed)
  for (col in seq_len(n_cols)) {
    lo <- p[col] + 1L; hi <- p[col + 1L]
    if (hi - lo < 1L) next
    perm <- sample(lo:hi)
    i[lo:hi] <- i[perm]; x[lo:hi] <- x[perm]
  }
  list(i = i, x = x)
}

make_sparse <- function(n_genes = 40, n_cells = 12, density = 0.4, seed = 1) {
  set.seed(seed)
  dense <- matrix(rbinom(n_genes * n_cells, 1, density) *
                    ceiling(runif(n_genes * n_cells, 1, 9)), n_genes, n_cells)
  methods::as(methods::as(dense, "Matrix"), "CsparseMatrix")
}

call_sparse_block <- function(i, p, x, n_genes, n_cells, metric = 5L) {
  res <- .Call("C_sparse_block", as.integer(i), as.integer(p), as.double(x),
               as.integer(n_genes), as.integer(n_cells), length(x),
               as.integer(metric), 1L, PACKAGE = "mtrx")
  if (is.null(res)) skip("C_sparse_block declined (GPU memory guard)")
  matrix(res, n_cells, n_cells)
}

test_that("Matrix keeps row indices sorted, so the R path never trips the contract", {
  M <- make_sparse()
  expect_true(column_indices_sorted(M@p, M@i, ncol(M)))

  # Row reordering is the operation that breaks the invariant in SciPy.
  reordered <- methods::as(M[sample(nrow(M)), ], "CsparseMatrix")
  expect_true(column_indices_sorted(reordered@p, reordered@i, ncol(reordered)))
})

test_that("scramble() really does violate the contract", {
  # Guard the guard: were this to stop scrambling, the tests below would pass
  # vacuously against already-canonical input.
  M <- make_sparse()
  bad <- scramble(M@p, M@i, M@x, ncol(M))
  expect_false(column_indices_sorted(M@p, bad$i, ncol(M)))

  # same data, only reordered within columns
  expect_equal(sort(bad$i), sort(M@i))
  expect_equal(sort(bad$x), sort(M@x))
  for (col in seq_len(ncol(M))) {
    lo <- M@p[col] + 1L; hi <- M@p[col + 1L]
    if (hi < lo) next
    expect_setequal(paste(bad$i[lo:hi], bad$x[lo:hi]),
                    paste(M@i[lo:hi], M@x[lo:hi]))
  }
})

test_that("unsorted CSC is repaired at the C boundary", {
  M <- make_sparse(seed = 3)
  reference <- kendall_ref(as.matrix(M))
  bad <- scramble(M@p, M@i, M@x, ncol(M), seed = 17)
  expect_false(column_indices_sorted(M@p, bad$i, ncol(M)))

  got <- call_sparse_block(bad$i, M@p, bad$x, nrow(M), ncol(M))

  expect_equal(dim(got), dim(reference))
  expect_lte(max(abs(got - reference)), 1e-5)
  expect_lte(max(got), 1 + 1e-9)      # the symptom that exposed the bug
})

test_that("sorted and scrambled inputs give the same distances", {
  M <- make_sparse(seed = 5)
  bad <- scramble(M@p, M@i, M@x, ncol(M), seed = 19)

  clean <- call_sparse_block(M@i, M@p, M@x, nrow(M), ncol(M))
  repaired <- call_sparse_block(bad$i, M@p, bad$x, nrow(M), ncol(M))
  expect_lte(max(abs(clean - repaired)), 1e-9)
})

test_that("repair does not modify the caller's vectors", {
  M <- make_sparse(seed = 7)
  bad <- scramble(M@p, M@i, M@x, ncol(M), seed = 23)
  kept_i <- bad$i; kept_x <- bad$x

  invisible(call_sparse_block(bad$i, M@p, bad$x, nrow(M), ncol(M)))

  expect_identical(bad$i, kept_i)
  expect_identical(bad$x, kept_x)
})

test_that("every sparsity-aware metric survives unsorted input", {
  M <- make_sparse(seed = 9)
  bad <- scramble(M@p, M@i, M@x, ncol(M), seed = 29)
  for (metric in c(3L, 4L, 5L)) {         # manhattan, spearman, kendall
    clean <- call_sparse_block(M@i, M@p, M@x, nrow(M), ncol(M), metric)
    repaired <- call_sparse_block(bad$i, M@p, bad$x, nrow(M), ncol(M), metric)
    expect_lte(max(abs(clean - repaired)), 1e-9)
  }
})
