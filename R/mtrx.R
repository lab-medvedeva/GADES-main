.onLoad <- function(libname, pkgname) {
   # CPU backend (mtrx_cpu.so) is always built; the GPU backend (mtrx.so) only
   # when CUDA was available at build time. Load the GPU lib optionally so the
   # package still loads on CPU-only deploys.
   library.dynam('mtrx_cpu', package = pkgname, lib.loc = libname)
   gpu <- tryCatch({
       library.dynam('mtrx', package = pkgname, lib.loc = libname); TRUE
   }, error = function(e) FALSE)
   if (gpu) tryCatch(.C("check_gpu", PACKAGE = "mtrx"), error = function(e) NULL)
}

.onUnload <- function(libpath) {
   library.dynam.unload('mtrx_cpu', libpath)
   try(library.dynam.unload('mtrx', libpath), silent = TRUE)
}

# Debug-gated printing. Set env HOBO_DEBUG=TRUE to enable verbose per-block
# diagnostics. Off by default so they neither pollute logs nor inflate the
# timed region (printing the per-block result matrix used to dominate runtime).
.HOBO_DEBUG <- function() identical(toupper(Sys.getenv("HOBO_DEBUG")), "TRUE")
.dbg <- function(...) if (.HOBO_DEBUG()) print(...)

# Single source for the metric -> C-driver-name mapping shared by process_batch
# (GPU) and process_batch_cpu (CPU). Replaces the duplicated 6-way glue() chains.
# name = matrix_<CamelMetric><postfix>_distance_<block_type>[_cpu]
.MTRX_METRIC_CC <- c(kendall="Kendall", euclidean="Euclidean", cosine="Cosine",
                     pearson="Pearson", manhattan="Manhattan", spearman="Spearman")
.mtrx_fn_name <- function(metric, block_type, postfix, cpu = FALSE) {
    cc <- .MTRX_METRIC_CC[[metric]]
    if (is.null(cc)) cc <- "Kendall"          # unknown metric -> Kendall (legacy default)
    paste0("matrix_", cc, postfix, "_distance_", block_type, if (cpu) "_cpu" else "")
}




#' Function to process batch from shared objects for GPU.
#'
#' @param count_matrix Count Matrix.
#' @param first_index index.
#' @param second_index index.
#' @param batch_size  int.
#' @param metric string for metric.
#' @return A list of correlation matrix, batch_a size and batch_b size.
#' @export
process_batch <- function(count_matrix, first_index, second_index, batch_size, metric,  sparse = F, sparse_layout = "default") {
    .dbg(c(first_index, second_index, batch_size))
    start = as.numeric(Sys.time()) * 1000000
    if (sparse && startsWith(sparse_layout, "per_cell_pair")) {
        postfix <- paste0('_sparse_', sparse_layout)
    } else if (sparse) {
        postfix <- '_sparse'
    } else {
        postfix <- ''
    }
    block_type <- if (first_index == second_index) "same_block" else "different_blocks"
    fn_name <- .mtrx_fn_name(metric, block_type, postfix)
    first_right_border <- min(first_index + batch_size, ncol(count_matrix))
    second_right_border <- min(second_index + batch_size, ncol(count_matrix))

    first_start <- first_index + 1
    second_start <- second_index + 1
    count_submatrix_a <- count_matrix[, c(first_start:first_right_border)]
    if (second_index == first_index) {
        count_submatrix_b = count_submatrix_a
    } else {
        count_submatrix_b <- count_matrix[, c(second_start:second_right_border)]
    }
    .dbg(dim(count_submatrix_a))
    batch_a_size <- first_right_border - first_index
    batch_b_size <- second_right_border - second_index

    stop = as.numeric(Sys.time()) * 1000000

    .dbg(glue("{stop - start}: prepare"))
    #st_t <- as.numeric(Sys.time()) * 1000000
    #result <- .C(
    #    fn_name,
    #    matrix_a = as.double(count_submatrix_a),
    #    matrix_b = as.double(count_submatrix_b),
    #    dist_matrix = double(batch_a_size * batch_b_size),
    #    rows = as.integer(nrow(count_matrix)),
    #    cols_a = as.integer(batch_a_size),
    #    cols_b = as.integer(batch_b_size),
	#PACKAGE = "mtrx"
    #)$dist_matrix
    #end_t = as.numeric(Sys.time()) * 1000000
    #print('KERNEL CALL')
    #print(end_t - st_t)
    
    start_t <- as.numeric(Sys.time()) * 1000000
    if (sparse && startsWith(sparse_layout, "per_cell_pair")) {
        count_submatrix_a <- as(count_submatrix_a, "CsparseMatrix")
        count_submatrix_b <- as(count_submatrix_b, "CsparseMatrix")

        stop = as.numeric(Sys.time()) * 1000000
        .dbg(glue("{stop - start}: prepare + Csparse (per_cell_pair)"))

        a_positions <- count_submatrix_a@p
        a_index <- count_submatrix_a@i
        a_values <- count_submatrix_a@x

        b_positions <- count_submatrix_b@p
        b_index <- count_submatrix_b@i
        b_values <- count_submatrix_b@x

        result <- .C(
            fn_name,
            a_index = a_index,
            a_positions = a_positions,
            a_values = a_values,
            b_index = b_index,
            b_positions = b_positions,
            b_values = b_values,
            dist_matrix = double(batch_a_size * batch_b_size),
            rows = as.integer(nrow(count_matrix)),
            cols_a = as.integer(batch_a_size),
            cols_b = as.integer(batch_b_size),
            num_elements_a = as.integer(length(a_values)),
            num_elements_b = as.integer(length(b_values)),
            PACKAGE = "mtrx"
        )$dist_matrix
    } else if (sparse) {
        count_submatrix_a <- as(count_submatrix_a, "RsparseMatrix")
        count_submatrix_b <- as(count_submatrix_b, "RsparseMatrix")

        stop = as.numeric(Sys.time()) * 1000000
        .dbg(glue("{stop - start}: prepare + Rsparse"))

        a_positions <- count_submatrix_a@p
        a_index <- count_submatrix_a@j
        a_values <- count_submatrix_a@x

        b_positions <- count_submatrix_b@p
        b_index <- count_submatrix_b@j
        b_values <- count_submatrix_b@x

        result <- .C(
            fn_name,
            a_index = a_index,
            a_positions = a_positions,
            a_values = a_values,
            b_index = b_index,
            b_positions = b_positions,
            b_values = b_values,
            dist_matrix = double(batch_a_size * batch_b_size),
            rows = as.integer(nrow(count_matrix)),
            cols_a = as.integer(batch_a_size),
            cols_b = as.integer(batch_b_size),
            num_elements_a = as.integer(length(a_values)),
            num_elements_b = as.integer(length(b_values)),
            PACKAGE = "mtrx"
        )$dist_matrix
    } else {
    	result <- .C(
            fn_name,
            matrix_a = as.double(count_submatrix_a),
            matrix_b = as.double(count_submatrix_b),
            dist_matrix = double(batch_a_size * batch_b_size),
            rows = as.integer(nrow(count_matrix)),
            cols_a = as.integer(batch_a_size),
            cols_b = as.integer(batch_b_size),
    	    PACKAGE = "mtrx"
    	)$dist_matrix
    }
    end_t = as.numeric(Sys.time()) * 1000000
    .dbg(end_t - start_t)
    dim(result) <- c(batch_a_size, batch_b_size)

    return (
        list(
            correlation_matrix=result,
            batch_a_size=batch_a_size,
            batch_b_size=batch_b_size
        )
    )
}

#' Function to process batch from shared objects for CPU.
#'
#' @param count_matrix Count Matrix.
#' @param first_index index.
#' @param second_index index.
#' @param batch_size  int.
#' @param metric string for metric.
#' @return A list of correlation matrix, batch_a size and batch_b size.
#' @export
process_batch_cpu <- function(count_matrix, first_index, second_index, batch_size, metric, sparse = F, sparse_layout = "default") {
    library(glue)
    if (sparse && sparse_layout == "per_cell_pair") {
        postfix <- '_sparse_per_cell_pair'
    } else if (sparse) {
        postfix <- '_sparse'
    } else {
        postfix <- ''
    }

    block_type <- if (first_index == second_index) "same_block" else "different_blocks"
    fn_name <- .mtrx_fn_name(metric, block_type, postfix, cpu = TRUE)
    # print(fn_name)
    first_right_border <- min(first_index + batch_size, ncol(count_matrix))
    second_right_border <- min(second_index + batch_size, ncol(count_matrix))

    first_start <- first_index + 1
    second_start <- second_index + 1
    if (.HOBO_DEBUG()) str(count_matrix)
    count_submatrix_a <- count_matrix[, c(first_start:first_right_border)]
    # print(dim(count_matrix))

    count_submatrix_b <- count_matrix[, c(second_start:second_right_border)]

    batch_a_size <- first_right_border - first_index
    batch_b_size <- second_right_border - second_index
    if (sparse && sparse_layout == "per_cell_pair") {
        count_submatrix_a <- as(count_submatrix_a, "CsparseMatrix")
        count_submatrix_b <- as(count_submatrix_b, "CsparseMatrix")

        a_positions <- count_submatrix_a@p
        a_index <- count_submatrix_a@i
        a_values <- count_submatrix_a@x

        b_positions <- count_submatrix_b@p
        b_index <- count_submatrix_b@i
        b_values <- count_submatrix_b@x

        result <- .C(
            fn_name,
            a_index = a_index,
            a_positions = a_positions,
            a_values = a_values,
            b_index = b_index,
            b_positions = b_positions,
            b_values = b_values,
            dist_matrix = double(batch_a_size * batch_b_size),
            rows = as.integer(nrow(count_matrix)),
            cols_a = as.integer(batch_a_size),
            cols_b = as.integer(batch_b_size),
            num_elements_a = as.integer(length(a_values)),
            num_elements_b = as.integer(length(b_values)),
            PACKAGE = "mtrx_cpu"
        )$dist_matrix
    } else if (sparse) {
        count_submatrix_a <- as(count_submatrix_a, "RsparseMatrix")
        count_submatrix_b <- as(count_submatrix_b, "RsparseMatrix")

        a_positions <- count_submatrix_a@p
        a_index <- count_submatrix_a@j
        a_values <- count_submatrix_a@x

        b_positions <- count_submatrix_b@p
        b_index <- count_submatrix_b@j
        b_values <- count_submatrix_b@x

        result <- .C(
            fn_name,
            a_index = a_index,
            a_positions = a_positions,
            a_values = a_values,
            b_index = b_index,
            b_positions = b_positions,
            b_values = b_values,
            dist_matrix = double(batch_a_size * batch_b_size),
            rows = as.integer(nrow(count_matrix)),
            cols_a = as.integer(batch_a_size),
            cols_b = as.integer(batch_b_size),
            num_elements_a = as.integer(length(a_values)),
            num_elements_b = as.integer(length(b_values)),
            PACKAGE = "mtrx_cpu"
        )$dist_matrix
    } else {
        result <- .C(
            fn_name,
            matrix_a = as.double(count_submatrix_a),
            matrix_b = as.double(count_submatrix_b),
            dist_matrix = double(batch_a_size * batch_b_size),
            rows = as.integer(nrow(count_matrix)),
            cols_a = as.integer(batch_a_size),
            cols_b = as.integer(batch_b_size),
        PACKAGE = "mtrx_cpu"
        )$dist_matrix
    }

    dim(result) <- c(batch_a_size, batch_b_size)

    return (
        list(
            correlation_matrix=result,
            batch_a_size=batch_a_size,
            batch_b_size=batch_b_size
        )
    )
}

#' Run a single per_cell_pair kernel call on two pre-sliced CSC submatrices.
#'
#' Bypasses the column-slicing inside `process_batch()`. Used by the h5ad
#' streaming path where slices come from disk rather than an in-memory matrix.
#' Returns a dense `m_a × m_b` distance block.
#'
#' @keywords internal
process_pair_per_cell_pair <- function(submat_a, submat_b, n_genes, metric, type) {
  m_a <- ncol(submat_a)
  m_b <- ncol(submat_b)
  same_block <- identical(submat_a, submat_b)
  Metric <- switch(metric,
    "kendall"   = "Kendall",
    "euclidean" = "Euclidean",
    "cosine"    = "Cosine",
    "pearson"   = "Pearson",
    "manhattan" = "Manhattan",
    "spearman"  = "Spearman",
    stop(sprintf("unknown metric: %s", metric))
  )
  pkg <- if (type == "gpu") "mtrx" else "mtrx_cpu"
  cpu_suffix <- if (type == "cpu") "_cpu" else ""
  block_suffix <- if (same_block) "same_block" else "different_blocks"
  fn_name <- paste0("matrix_", Metric,
                    "_sparse_per_cell_pair_distance_", block_suffix, cpu_suffix)

  result <- .C(
    fn_name,
    a_index        = submat_a@i,
    a_positions    = submat_a@p,
    a_values       = submat_a@x,
    b_index        = submat_b@i,
    b_positions    = submat_b@p,
    b_values       = submat_b@x,
    dist_matrix    = double(m_a * m_b),
    rows           = as.integer(n_genes),
    cols_a         = as.integer(m_a),
    cols_b         = as.integer(m_b),
    num_elements_a = as.integer(length(submat_a@x)),
    num_elements_b = as.integer(length(submat_b@x)),
    PACKAGE = pkg
  )$dist_matrix
  dim(result) <- c(m_a, m_b)
  result
}

#' Streaming distance from h5ad with sparse HDF5 output.
#'
#' Internal helper called by `mtrx_distance()` when `a` is a path to .h5ad.
#' Pipeline: read batches of cells from disk → per_cell_pair kernel →
#' filter via metric-specific predicate → append COO triplets to output HDF5.
#' Neither full input nor full output matrix lives in RAM. Supports resume via
#' checkpoint attrs in the output file. See docs/adr/0001 and CONTEXT.md.
#'
#' @keywords internal
.mtrx_distance_h5ad <- function(h5ad_path, output_path, batch_size, metric, type) {
  if (!nzchar(output_path)) {
    stop("mtrx_distance(h5ad) requires `filename` for sparse HDF5 output")
  }
  h <- h5ad_open(h5ad_path)
  on.exit(h5ad_close(h), add = TRUE)
  m <- h$n_cells
  n_genes <- h$n_genes
  obs_names <- h5ad_obs_names(h)

  w <- sparse_writer_open(output_path, m, obs_names, metric)
  on.exit(sparse_writer_close(w), add = TRUE, after = FALSE)

  chk <- sparse_writer_get_checkpoint(w)
  chk_first  <- chk[1]
  chk_second <- chk[2]
  if (chk_first >= 0) {
    print(glue("resuming from checkpoint: last=({chk_first}, {chk_second})"))
  }

  for (first_idx in seq(0L, m - 1L, by = batch_size)) {
    for (second_idx in seq(0L, m - 1L, by = batch_size)) {
      if (first_idx > second_idx) next  # lower triangle — not stored

      # Skip batches at-or-before the checkpoint
      if (first_idx < chk_first ||
          (first_idx == chk_first && second_idx <= chk_second)) next

      first_end  <- min(first_idx + batch_size, m)
      second_end <- min(second_idx + batch_size, m)

      t0 <- as.numeric(Sys.time()) * 1e6
      submat_a <- h5ad_slice_cells(h, first_idx, first_end)
      submat_b <- if (first_idx == second_idx) submat_a
                  else h5ad_slice_cells(h, second_idx, second_end)
      t1 <- as.numeric(Sys.time()) * 1e6

      block <- process_pair_per_cell_pair(submat_a, submat_b, n_genes, metric, type)
      n_kept <- sparse_writer_append_block(w, block, first_idx, second_idx)
      sparse_writer_set_checkpoint(w, first_idx, second_idx)
      t2 <- as.numeric(Sys.time()) * 1e6

      print(glue("batch ({first_idx}, {second_idx}) ",
                 "read={round((t1-t0)/1000)}ms ",
                 "kernel+write={round((t2-t1)/1000)}ms ",
                 "kept={n_kept}"))
    }
  }
  invisible(TRUE)
}

#' Function for generating report for distance matric.
#'
#' @param a Count matrix (in-memory `Matrix`) or character path to `.h5ad`
#'   file. For h5ad mode, `filename` is required and becomes the output
#'   sparse HDF5 path; `sparse` and `sparse_layout` are forced (per_cell_pair).
#' @param filename CSV file (or HDF5 output path for h5ad mode).
#' @param batch_size int.
#' @param metric string for matric selection.
#' @param type "gpu" or "cpu".
#' @param memory_limit_gb Hard RSS cap (GB). If the upcoming allocation or the
#'   current process RSS would exceed this limit, the function prints
#'   "memory limit", writes a {filename}_memlimit marker, and returns early
#'   with a list(status="memory_limit", ...). Default 50 GB. Defense-in-depth;
#'   the primary mechanism is cgroup-wrapping at the benchmark-script level.
#' @return TRUE on success, or list(status="memory_limit", ...) if the limit
#'   was tripped.
#' @export
# Placement policy: try the .Call fast paths that avoid
# the .C block-loop's per-batch coercion + output duplication, in order:
#   1. dense GPU batched (euc/cos/pear/manh, any m)   -> C_dense_block_batched
#   2. dense single-.Call full matrix (ncol <= batch) -> C_dense_block[_cpu]
#   3. sparse GPU batched (euc/cos/pear, any m)       -> C_sparse_block_batched
#   4. sparse single-.Call full matrix (ncol <= batch)-> C_sparse_block[_cpu]
# Each C side returns NULL when the result won't fit the device -> we fall to the
# next attempt / the block loop. Returns list(done=TRUE, value=<result>) when a
# fast path handled the call, else list(done=FALSE).
.mtrx_try_fast <- function(a, metric, type, sparse, write, filename, batch_size, memory_limit_gb) {
  metric_code <- switch(metric,
    euclidean = 0L, cosine = 1L, pearson = 2L,
    manhattan = 3L, spearman = 4L, kendall = 5L, NULL)

  # 1. Honest C-side batched dense (GPU euclidean/cosine/pearson/manhattan, ANY m).
  # Input uploaded once; m x m output tiled batch x batch on device -> GPU mem
  # O(n*m + batch^2). Skipped only for per-block file output (write && filename).
  if (type == "gpu" && !sparse && !is.null(metric_code) && metric_code <= 3L &&
      !(write && filename != "")) {                    # 0-2 cuBLAS GEMM, 3 tiled L1 (manhattan)
    a_dense <- as.matrix(a)
    n0 <- nrow(a_dense); m0 <- ncol(a_dense)
    ram_ok <- (!write) ||
      (as.numeric(m0) * as.numeric(m0) * 8 < memory_limit_gb * 1024^3)
    if (ram_ok) {
      # pass the matrix UNCONVERTED (int or double) — C reads the SEXP directly.
      res <- .Call("C_dense_block_batched", a_dense,
                   as.integer(n0), as.integer(m0), metric_code,
                   as.integer(batch_size), if (write) 1L else 0L, PACKAGE = "mtrx")
      if (!is.null(res)) {
        if (write) {
          dim(res) <- c(m0, m0)
          colnames(res) <- colnames(a_dense); rownames(res) <- colnames(a_dense)
          return(list(done = TRUE, value = res))
        }
        return(list(done = TRUE, value = TRUE))
      }
    }
    rm(a_dense)
  }

  # 2. Dense single-.Call full matrix (GPU or CPU), ncol <= batch_size.
  if (type %in% c("gpu", "cpu") && !sparse && !is.null(metric_code) &&
      ncol(a) <= batch_size &&
      !(write && filename != "")) {
    a_dense <- as.matrix(a)
    n0 <- nrow(a_dense); m0 <- ncol(a_dense)
    # CPU RAM guard (GPU has its own cudaMemGetInfo guard in C).
    cpu_ok <- type == "gpu" ||
      (as.numeric(m0) * as.numeric(m0) * 8 < memory_limit_gb * 1024^3)
    if (cpu_ok) {
      fn  <- if (type == "gpu") "C_dense_block" else "C_dense_block_cpu"
      pkg <- if (type == "gpu") "mtrx" else "mtrx_cpu"
      res <- .Call(fn, as.double(a_dense),
                   as.integer(n0), as.integer(m0), metric_code, PACKAGE = pkg)
      if (!is.null(res)) {
        if (write) {                              # implies filename == "" here
          dim(res) <- c(m0, m0)
          colnames(res) <- colnames(a_dense)
          rownames(res) <- colnames(a_dense)
          return(list(done = TRUE, value = res))
        }
        return(list(done = TRUE, value = TRUE))   # benchmark / discard mode
      }
    }
    rm(a_dense)
  }

  # 3. Honest C-side batched sparse (GPU euclidean/cosine/pearson, ANY m).
  # Upload CSC once; per tile densify only the needed column-blocks on the GPU.
  if (type == "gpu" && sparse && !is.null(metric_code) && metric_code <= 2L &&
      !(write && filename != "")) {
    sm <- as(a, "CsparseMatrix")
    n0 <- nrow(sm); m0 <- ncol(sm)
    ram_ok <- (!write) ||
      (as.numeric(m0) * as.numeric(m0) * 8 < memory_limit_gb * 1024^3)
    if (ram_ok) {
      res <- .Call("C_sparse_block_batched",
                   as.integer(sm@i), as.integer(sm@p), as.double(sm@x),
                   as.integer(n0), as.integer(m0), as.integer(length(sm@x)),
                   metric_code, as.integer(batch_size), if (write) 1L else 0L,
                   PACKAGE = "mtrx")
      if (!is.null(res)) {
        if (write) {
          dim(res) <- c(m0, m0)
          colnames(res) <- colnames(sm); rownames(res) <- colnames(sm)
          return(list(done = TRUE, value = res))
        }
        return(list(done = TRUE, value = TRUE))
      }
    }
  }

  # 4. Sparse single-.Call full matrix (GPU or CPU), ncol <= batch_size.
  # euclidean/cosine/pearson densify (GPU/host) + dense path; manhattan/spearman/
  # kendall use the sparsity-aware per_cell_pair kernels.
  if (type %in% c("gpu", "cpu") && sparse && !is.null(metric_code) &&
      ncol(a) <= batch_size &&
      !(write && filename != "")) {
    sm <- as(a, "CsparseMatrix")
    n0 <- nrow(sm); m0 <- ncol(sm)
    cpu_bytes <- as.numeric(m0) * as.numeric(m0) * 8 +
                 (if (metric_code <= 2) as.numeric(n0) * as.numeric(m0) * 8 else 0)
    cpu_ok <- type == "gpu" || (cpu_bytes < memory_limit_gb * 1024^3)
    if (cpu_ok) {
      fn  <- if (type == "gpu") "C_sparse_block" else "C_sparse_block_cpu"
      pkg <- if (type == "gpu") "mtrx" else "mtrx_cpu"
      res <- .Call(fn, as.integer(sm@i), as.integer(sm@p), as.double(sm@x),
                   as.integer(n0), as.integer(m0), as.integer(length(sm@x)),
                   metric_code, 1L, PACKAGE = pkg)
      if (!is.null(res)) {
        if (write) {
          dim(res) <- c(m0, m0)
          colnames(res) <- colnames(sm); rownames(res) <- colnames(sm)
          return(list(done = TRUE, value = res))
        }
        return(list(done = TRUE, value = TRUE))
      }
    }
  }

  list(done = FALSE)   # no fast path applied -> caller runs the block loop
}

mtrx_distance <- function(a, filename = "", batch_size = 1000, metric = "kendall",type="gpu", sparse = F, write=F, sparse_layout = "default", memory_limit_gb = 50)
{
  # h5ad streaming path — neither full input nor full output in RAM
  if (is.character(a) && length(a) == 1 && endsWith(a, ".h5ad")) {
    return(.mtrx_distance_h5ad(a, filename, batch_size, metric, type))
  }

  # Placement policy: attempt the .Call fast paths; fall through to the block loop.
  .fp <- .mtrx_try_fast(a, metric, type, sparse, write, filename, batch_size, memory_limit_gb)
  if (.fp$done) return(.fp$value)

#  if (sparse) {
  a <- as(a, 'CsparseMatrix')
#  }
  n <- nrow(a)
  m <- ncol(a)
  .dbg(filename)
  .dbg(c(m, batch_size))
  .dbg('Distance started')

  # --- Memory-limit guard (defense-in-depth; primary limit is cgroup) -----
  bytes_limit <- memory_limit_gb * 1024^3

  get_rss_bytes <- function() {
    rss <- tryCatch(as.numeric(ps::ps_memory_info()["rss"]), error = function(e) NA)
    if (is.na(rss)) {
      rss <- tryCatch({
        status <- readLines("/proc/self/status")
        line <- grep("^VmRSS:", status, value = TRUE)
        as.numeric(sub("VmRSS:\\s+(\\d+).*", "\\1", line)) * 1024
      }, error = function(e) 0)
    }
    rss
  }

  mark_memory_limit <- function(current_bytes, where) {
    print("memory limit")
    msg <- sprintf("memory limit: %.2f GB used (>= %.0f GB cap) at %s",
                   current_bytes / 1024^3, memory_limit_gb, where)
    print(msg)
    if (nzchar(filename)) {
      writeLines(msg, paste0(filename, "_memlimit"))
    }
    return(list(status = "memory_limit",
                current_gb = current_bytes / 1024^3,
                limit_gb = memory_limit_gb,
                where = where, n = n, m = m))
  }

  if (filename == "") {
    result_bytes <- as.numeric(m) * as.numeric(m) * 8
    current <- get_rss_bytes()
    if (current + result_bytes > bytes_limit) {
      return(mark_memory_limit(current + result_bytes,
                               sprintf("pre-alloc result (%d x %d double)", m, m)))
    }
  }
  # ----------------------------------------------------------------------

  if (filename == ""){
    result_overall <- double(m * m)
    dim(result_overall) <- c(m, m)

    colnames(result_overall) = colnames(a)
    rownames(result_overall) = colnames(a)
  }
  .blk_log <- nzchar(Sys.getenv("HOBO_BLOCK_LOG"))
  .nb_same <- 0L; .nb_diff <- 0L; .nb_skip <- 0L
  .nb_total <- length(seq(0, m - 1, by=batch_size))^2
  for (first_index in seq(0, m - 1, by=batch_size)) {

    for (second_index in seq(0, m - 1, by=batch_size)) {
        current <- get_rss_bytes()
        if (current > bytes_limit) {
          return(mark_memory_limit(current,
                                   sprintf("batch (%d,%d)", first_index, second_index)))
        }
        start_t <- as.numeric(Sys.time()) * 1000000
        if (first_index > second_index) {
            a_left = first_index + 1
            b_left = second_index + 1
            a_right <- min(first_index + batch_size, ncol(a))
            b_right <- min(second_index + batch_size, ncol(a))
            if (write) {
                result_overall[c(a_left:a_right), c(b_left:b_right)] = t(result_overall[c(b_left:b_right), c(a_left:a_right)])
            }
        } else {
            if(type=="gpu"){
                result <- process_batch(
                    count_matrix = a,
                    first_index = first_index,
                    second_index = second_index,
                    batch_size = batch_size,
                    metric = metric,
                    sparse=sparse,
                    sparse_layout=sparse_layout
                )
            } else if (type=="cpu") { 
                result <- process_batch_cpu(
                    count_matrix = a,
                    first_index = first_index,
                    second_index = second_index,
                    batch_size = batch_size,
                    metric = metric,
                    sparse = sparse,
                    sparse_layout = sparse_layout
                )
            }
            a_left = first_index + 1
            a_right = first_index + result$batch_a_size
            b_left = second_index + 1
            b_right = second_index + result$batch_b_size
            
            correlation_matrix <- result$correlation_matrix
            .dbg(result)
            .dbg(dim(correlation_matrix))
            # print(correlation_matrix[1:10, 1:10])
            if (filename == "") {
                result_overall[c(a_left:a_right), c(b_left:b_right)] <- correlation_matrix
            } else if (write) {
                output_filename <- glue('{filename}_{a_left}_{a_right}_{b_left}_{b_right}.csv')
                write.csv(correlation_matrix, output_filename)
            }
        }
        end_t <- as.numeric(Sys.time()) * 1000000
        if (.blk_log) {
            if (first_index > second_index) {
                .nb_skip <- .nb_skip + 1L
                kind <- "skip(sym)"
            } else if (first_index == second_index) {
                .nb_same <- .nb_same + 1L
                kind <- "SAME"
            } else {
                .nb_diff <- .nb_diff + 1L
                kind <- "DIFF"
            }
            cat(sprintf("[blk %d/%d %-9s] (fi=%d,si=%d) %.2fs | same=%d diff=%d skip=%d\n",
                        .nb_same + .nb_diff + .nb_skip, .nb_total, kind,
                        first_index, second_index, (end_t - start_t)/1e6,
                        .nb_same, .nb_diff, .nb_skip))
            flush.console()
        }
        .dbg('GC called')
        #print(gc())
        .dbg('Full')
        .dbg(end_t - start_t)
    }
  }
  if (.blk_log) cat(sprintf("[blocks done] total=%d same=%d diff=%d skip=%d\n",
                            .nb_same + .nb_diff + .nb_skip, .nb_same, .nb_diff, .nb_skip))
  if (write) {
    return (result_overall)
  }

  return (TRUE)
}

#' get matrix from file.
#' 
#' @param filename File path for csv.
#' @return data countmatrix data.
#' @export
mtrx_read_kdm <- function(filename){
  MATRIXFILE <- file(filename, "rb")
  m <- readBin(MATRIXFILE, integer(), n = 1, size = 4)
  print(m)
  names <- readBin(MATRIXFILE, character(), m)
  print(names)
  data <- matrix(c(0), ncol = m, nrow = m)
  for (i in 1:(m-1)){
    datasample <- readBin(MATRIXFILE, numeric(), n = m-i, size = 8)
    print(datasample)
    data[i,(i+1):m] <- datasample
    data[(i+1):m,i] <- datasample
  }
  close(MATRIXFILE)
  colnames(data) <- names
  rownames(data) <- names
  return(data)
}
