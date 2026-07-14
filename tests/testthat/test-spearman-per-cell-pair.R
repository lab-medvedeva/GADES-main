# Spearman per_cell_pair kernel correctness.
# Validates against R's cor(method="spearman") and the existing
# per_gene_pair Spearman implementation.

call_pcp_sparse <- function(fn, A_csc, B_csc, n_genes, m_a, m_b, pkg) {
  result <- .C(
    fn,
    a_index = A_csc@i,
    a_positions = A_csc@p,
    a_values = A_csc@x,
    b_index = B_csc@i,
    b_positions = B_csc@p,
    b_values = B_csc@x,
    dist_matrix = double(m_a * m_b),
    rows = as.integer(n_genes),
    cols_a = as.integer(m_a),
    cols_b = as.integer(m_b),
    num_elements_a = as.integer(length(A_csc@x)),
    num_elements_b = as.integer(length(B_csc@x)),
    PACKAGE = pkg
  )$dist_matrix
  dim(result) <- c(m_a, m_b)
  result
}

call_pgp_sparse <- function(fn, A_rsp, B_rsp, n_genes, m_a, m_b, pkg) {
  result <- .C(
    fn,
    a_index = A_rsp@j,
    a_positions = A_rsp@p,
    a_values = A_rsp@x,
    b_index = B_rsp@j,
    b_positions = B_rsp@p,
    b_values = B_rsp@x,
    dist_matrix = double(m_a * m_b),
    rows = as.integer(n_genes),
    cols_a = as.integer(m_a),
    cols_b = as.integer(m_b),
    num_elements_a = as.integer(length(A_rsp@x)),
    num_elements_b = as.integer(length(B_rsp@x)),
    PACKAGE = pkg
  )$dist_matrix
  dim(result) <- c(m_a, m_b)
  result
}

test_that("Spearman per_cell_pair same_block matches R cor()", {
  set.seed(123)
  n <- 50; m <- 12
  M <- matrix(rpois(n * m, lambda = 0.7), nrow = n, ncol = m)
  ref <- 1 - cor(M, method = "spearman")
  M_c <- as(M, "CsparseMatrix")

  for (pkg in c("mtrx_cpu", "mtrx")) {
    suffix <- if (pkg == "mtrx_cpu") "_cpu" else ""
    fn <- paste0("matrix_Spearman_sparse_per_cell_pair_distance_same_block", suffix)
    got <- call_pcp_sparse(fn, M_c, M_c, n, m, m, pkg)
    diag(got) <- 0  # cor diagonal is 0
    expect_lt(max(abs(got - ref)), 1e-5,
              label = paste(pkg, "vs R cor"))
  }
})

test_that("Spearman per_cell_pair different_blocks matches R cor()", {
  set.seed(456)
  n <- 80; m_a <- 8; m_b <- 10
  A <- matrix(rpois(n * m_a, lambda = 0.6), nrow = n, ncol = m_a)
  B <- matrix(rpois(n * m_b, lambda = 0.6), nrow = n, ncol = m_b)
  ref <- 1 - cor(A, B, method = "spearman")
  A_c <- as(A, "CsparseMatrix")
  B_c <- as(B, "CsparseMatrix")

  for (pkg in c("mtrx_cpu", "mtrx")) {
    suffix <- if (pkg == "mtrx_cpu") "_cpu" else ""
    fn <- paste0("matrix_Spearman_sparse_per_cell_pair_distance_different_blocks", suffix)
    got <- call_pcp_sparse(fn, A_c, B_c, n, m_a, m_b, pkg)
    expect_lt(max(abs(got - ref)), 1e-5,
              label = paste(pkg, "vs R cor"))
  }
})

test_that("per_cell_pair Spearman agrees with per_gene_pair (existing kernel)", {
  set.seed(789)
  n <- 40; m <- 7
  M <- matrix(rpois(n * m, lambda = 0.5), nrow = n, ncol = m)
  M_c <- as(M, "CsparseMatrix")
  M_r <- as(M, "RsparseMatrix")

  for (pkg in c("mtrx_cpu", "mtrx")) {
    suffix <- if (pkg == "mtrx_cpu") "_cpu" else ""
    fn_pcp <- paste0("matrix_Spearman_sparse_per_cell_pair_distance_same_block", suffix)
    fn_pgp <- paste0("matrix_Spearman_sparse_distance_same_block", suffix)
    got_pcp <- call_pcp_sparse(fn_pcp, M_c, M_c, n, m, m, pkg)
    got_pgp <- call_pgp_sparse(fn_pgp, M_r, M_r, n, m, m, pkg)
    diag(got_pcp) <- 0; diag(got_pgp) <- 0
    expect_lt(max(abs(got_pcp - got_pgp)), 1e-5,
              label = paste(pkg, "pcp vs pgp"))
  }
})

test_that("empty cells produce NaN (matches CONTEXT.md default-NaN semantics)", {
  n <- 30; m <- 5
  M <- matrix(0, nrow = n, ncol = m)
  M[, 1:3] <- rpois(n * 3, lambda = 0.8)
  M_c <- as(M, "CsparseMatrix")

  for (pkg in c("mtrx_cpu", "mtrx")) {
    suffix <- if (pkg == "mtrx_cpu") "_cpu" else ""
    fn <- paste0("matrix_Spearman_sparse_per_cell_pair_distance_same_block", suffix)
    got <- call_pcp_sparse(fn, M_c, M_c, n, m, m, pkg)
    # Cells 4 and 5 are empty
    expect_true(is.nan(got[4, 5]), label = paste(pkg, "d(empty,empty)"))
    expect_true(is.nan(got[1, 4]), label = paste(pkg, "d(active,empty)"))
    expect_false(is.nan(got[1, 2]), label = paste(pkg, "d(active,active)"))
  }
})

test_that("end-to-end Spearman via process_batch (R API integration)", {
  set.seed(42)
  n <- 60; m <- 14
  M <- matrix(rpois(n * m, lambda = 0.5), nrow = n, ncol = m)
  M_c <- as(M, "CsparseMatrix")
  ref <- 1 - cor(M, method = "spearman")

  for (type in c("cpu", "gpu")) {
    fn <- if (type == "cpu") process_batch_cpu else process_batch
    r <- silent({
      fn(M_c, 0, 0, m, "spearman", sparse = TRUE, sparse_layout = "per_cell_pair")
    })
    diag(r$correlation_matrix) <- 0
    expect_lt(max(abs(r$correlation_matrix - ref)), 1e-5,
              label = paste("process_batch", type))
  }
})
