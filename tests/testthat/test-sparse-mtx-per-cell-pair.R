# Per-cell-pair Kendall & Spearman validated on genuinely SPARSE inputs loaded
# from MatrixMarket (.mtx) files via readMM — the production input path used by
# the benchmark — rather than dense matrices wrapped as sparse. Covers both the
# single-batch (same_block) and multi-batch (different_blocks) code paths, on
# GPU and CPU, against pure-R references.
#
# Convention: generated/.mtx files store cells x genes; the pipeline transposes
# to genes x cells (rows = genes) before computing cell-by-cell distances.

# --- pure-R references (genes = rows, cells = columns) ---
kendall_ref_taunaive <- function(M) {           # 1 - tau_naive distance
  n <- nrow(M); m <- ncol(M); D <- matrix(0, m, m)
  for (a in 1:(m - 1)) for (b in (a + 1):m) {
    disc <- sum((outer(M[, a], M[, a], "-") * outer(M[, b], M[, b], "-")) < 0) / 2
    d <- disc * 2 / (n * (n - 1)); D[a, b] <- d; D[b, a] <- d
  }
  D
}
spearman_ref <- function(M) {
  D <- 1 - suppressWarnings(cor(M, method = "spearman")); D[!is.finite(D)] <- NA; as.matrix(D)
}

# Compare off-diagonal entries where the reference is defined (mask empty/
# constant cells whose Spearman correlation is undefined).
expect_close <- function(got, ref, tol, label) {
  g <- got; r <- ref; diag(g) <- 0; diag(r) <- 0
  ok <- is.finite(r) & is.finite(g)
  d <- if (any(ok)) max(abs(g[ok] - r[ok])) else 0
  expect_lt(d, tol, label = label)
}

run_pcp <- function(M_gxc, metric, type, batch_size) {
  silent(as.matrix(mtrx_distance(as(M_gxc, "CsparseMatrix"), batch_size = batch_size,
                                 metric = metric, type = type, sparse = TRUE,
                                 sparse_layout = "per_cell_pair", write = TRUE)))
}

# Write a genes x cells matrix to disk as a cells x genes .mtx, read it back as
# genes x cells — exercising the real MatrixMarket round-trip + transpose.
mtx_roundtrip <- function(M_gxc) {
  path <- tempfile(fileext = ".mtx")
  writeMM(as(t(M_gxc), "CsparseMatrix"), path)
  M <- t(readMM(path))
  unlink(path)
  M
}

# Sparse generator that guarantees every cell has >= 2 distinct nonzeros (so no
# undefined-variance columns) and includes negatives to exercise sign handling.
gen_sparse <- function(n_genes, n_cells, density, seed) {
  set.seed(seed)
  M <- matrix(0, n_genes, n_cells)
  mask <- runif(n_genes * n_cells) < density
  M[mask] <- sample(c(-9:-1, 1:9), sum(mask), replace = TRUE)
  for (c in seq_len(n_cells)) {                  # ensure >=2 distinct nonzeros / cell
    if (sum(M[, c] != 0) < 2) { M[1, c] <- 3; M[2, c] <- -5 }
  }
  M
}

test_that("Kendall per_cell_pair on .mtx round-trip matches pure-R (same_block & batched)", {
  M <- mtx_roundtrip(gen_sparse(120, 16, 0.20, 1))
  expect_s4_class(as(M, "CsparseMatrix"), "CsparseMatrix")
  ref <- kendall_ref_taunaive(as.matrix(M))
  for (type in c("cpu", "gpu")) {
    expect_close(run_pcp(M, "kendall", type, 16L), ref, 1e-6, paste("kendall", type, "same_block"))
    expect_close(run_pcp(M, "kendall", type, 6L),  ref, 1e-6, paste("kendall", type, "different_blocks"))
  }
})

test_that("Spearman per_cell_pair on .mtx round-trip matches pure-R (same_block & batched)", {
  M <- mtx_roundtrip(gen_sparse(150, 18, 0.25, 2))
  ref <- spearman_ref(as.matrix(M))
  for (type in c("cpu", "gpu")) {
    expect_close(run_pcp(M, "spearman", type, 18L), ref, 1e-4, paste("spearman", type, "same_block"))
    expect_close(run_pcp(M, "spearman", type, 7L),  ref, 1e-4, paste("spearman", type, "different_blocks"))
  }
})

test_that("Manhattan per_cell_pair on .mtx round-trip matches pure-R dist & default", {
  M  <- mtx_roundtrip(gen_sparse(120, 16, 0.20, 11))
  Mc <- as(M, "CsparseMatrix")
  refD <- as.matrix(dist(t(as.matrix(M)), method = "manhattan"))   # cell-by-cell L1
  # Manhattan distances are large sums; float32 accumulation -> use relative tol.
  rel_lt <- function(got, ref, lbl) {
    g <- got; diag(g) <- 0
    expect_lt(max(abs(g - ref) / pmax(1, abs(ref))), 1e-4, label = lbl)
  }
  for (type in c("cpu", "gpu")) {
    rel_lt(run_pcp(M, "manhattan", type, 16L), refD, paste("manhattan", type, "same_block vs dist"))
    rel_lt(run_pcp(M, "manhattan", type, 6L),  refD, paste("manhattan", type, "different_blocks vs dist"))
    def <- silent(as.matrix(mtrx_distance(Mc, batch_size = 16L, metric = "manhattan", type = type,
                                          sparse = TRUE, sparse_layout = "default", write = TRUE)))
    rel_lt(run_pcp(M, "manhattan", type, 16L), def, paste("manhattan", type, "pcp vs default"))
  }
})

test_that("per_cell_pair matches the default sparse path on .mtx data", {
  M <- mtx_roundtrip(gen_sparse(200, 14, 0.15, 3))
  Mc <- as(M, "CsparseMatrix")
  # The default GPU sparse Kendall path now delegates to per_cell_pair (the legacy
  # per-gene-pair kernel was numerically wrong). Default and per_cell_pair must
  # agree on CPU and GPU, in both single-batch (same_block) and batched
  # (different_blocks) modes.
  for (metric in c("kendall", "spearman")) {
    tol <- if (metric == "kendall") 1e-6 else 1e-4
    for (type in c("cpu", "gpu")) {
      for (bs in c(14L, 6L)) {       # 14 -> same_block, 6 -> different_blocks
        def <- silent(as.matrix(mtrx_distance(Mc, batch_size = bs, metric = metric, type = type,
                                              sparse = TRUE, sparse_layout = "default", write = TRUE)))
        pcp <- run_pcp(M, metric, type, bs)
        expect_close(pcp, def, tol, paste(metric, type, "pcp vs default bs", bs))
      }
    }
  }
})

test_that("per_cell_pair on a real GeneratedSparse .mtx (kendall + spearman, GPU & CPU)", {
  root <- Sys.getenv("HOBOTNICA_PROJECT_ROOT", unset = getwd())
  # small real dataset: cells x genes; pick the first that exists
  candidates <- file.path(root, "GeneratedSparse",
                          c("100_cells_100_features", "10_cells_1000_features"), "0.9.mtx")
  path <- candidates[file.exists(candidates)][1]
  skip_unless_dataset(if (is.na(path)) "" else path, "GeneratedSparse small .mtx")

  M <- t(readMM(path))                            # genes x cells
  m <- ncol(M)
  refK <- kendall_ref_taunaive(as.matrix(M))
  refS <- spearman_ref(as.matrix(M))
  bs_full <- m; bs_half <- max(2L, m %/% 2L)
  for (type in c("cpu", "gpu")) {
    expect_close(run_pcp(M, "kendall",  type, bs_full), refK, 1e-6, paste("real kendall",  type, "1 batch"))
    expect_close(run_pcp(M, "kendall",  type, bs_half), refK, 1e-6, paste("real kendall",  type, "batched"))
    expect_close(run_pcp(M, "spearman", type, bs_full), refS, 1e-4, paste("real spearman", type, "1 batch"))
    expect_close(run_pcp(M, "spearman", type, bs_half), refS, 1e-4, paste("real spearman", type, "batched"))
  }
})
