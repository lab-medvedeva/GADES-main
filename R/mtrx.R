.onLoad <- function(libname, pkgname) {
   library.dynam('mtrx', package = 'GADES', lib.loc = NULL)
   library.dynam('mtrx_cpu', package = 'GADES', lib.loc = NULL)
   .C("check_gpu", PACKAGE="mtrx")
   library(glue)
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
process_batch <- function(count_matrix, first_index, second_index, batch_size, metric,  sparse = F) {
    start = as.numeric(Sys.time()) * 1000000
    if (sparse) {
        postfix <- '_sparse'
    } else {
        postfix <- ''
    }
    if (first_index == second_index) {
        if (metric == 'kendall') {
            fn_name <- glue("matrix_Kendall{postfix}_distance_same_block")
        } else if(metric =='euclidean') {
            fn_name <- glue("matrix_Euclidean{postfix}_distance_same_block")
        } else if(metric =='pearson') {
            fn_name <- glue("matrix_Pearson{postfix}_distance_same_block") #for not different block
        } else {
	    fn_name <- "matrix_Kendall_distance_same_block"
	}
    } else {
        if (metric == 'kendall') {
            fn_name <- glue("matrix_Kendall{postfix}_distance_different_blocks")
        } else if(metric =='euclidean') {
            fn_name <- glue("matrix_Euclidean{postfix}_distance_different_blocks")
        } else if(metric =='pearson') {
            fn_name <- glue("matrix_Pearson{postfix}_distance_different_blocks")
        } else {
	    fn_name <-"matrix_kendall_distance_different_blocks"
	}
    }
    first_right_border <- min(first_index + batch_size, ncol(count_matrix))
    second_right_border <- min(second_index + batch_size, ncol(count_matrix))

    first_start <- first_index + 1
    second_start <- second_index + 1
    count_submatrix_a <- count_matrix[, c(first_start:first_right_border)]
    count_submatrix_b <- count_matrix[, c(second_start:second_right_border)]
    batch_a_size <- first_right_border - first_index
    batch_b_size <- second_right_border - second_index

    stop = as.numeric(Sys.time()) * 1000000

    start_t <- as.numeric(Sys.time()) * 1000000
    if (sparse) {
        count_submatrix_a <- as(count_submatrix_a, "RsparseMatrix")
        count_submatrix_b <- as(count_submatrix_b, "RsparseMatrix")

        stop = as.numeric(Sys.time()) * 1000000
        print(glue("{stop - start}: prepare + Rsparse"))

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
process_batch_cpu <- function(count_matrix, first_index, second_index, batch_size, metric, sparse = F) {
    library(glue)
    if (sparse) {
        postfix <- '_sparse'
    } else {
        postfix <- ''
    }

    if (first_index == second_index) {
        if (metric == 'kendall') {
            fn_name <- glue("matrix_Kendall{postfix}_distance_same_block_cpu")
        } else if(metric =='euclidean') {
            fn_name <- glue("matrix_Euclidean{postfix}_distance_same_block_cpu")
        } else if(metric =='pearson') {
            fn_name <- glue("matrix_Pearson{postfix}_distance_same_block_cpu") #For not different block
        } else {
	    fn_name <- glue("matrix_Kendall{postfix}_distance_same_block_cpu")
	}
    } else {
        if (metric == 'kendall') {
            fn_name <- glue("matrix_Kendall{postfix}_distance_different_blocks_cpu")
        } else if(metric =='euclidean') {
            fn_name <- glue("matrix_Euclidean{postfix}_distance_different_blocks_cpu")
        } else if(metric =='pearson') {
            fn_name <- glue("matrix_Pearson{postfix}_distance_different_blocks_cpu")
        }
        else {
            fn_name <-glue("matrix_Kendall{postfix}_distance_different_blocks_cpu")
        }
    }
    first_right_border <- min(first_index + batch_size, ncol(count_matrix))
    second_right_border <- min(second_index + batch_size, ncol(count_matrix))

    first_start <- first_index + 1
    second_start <- second_index + 1
    count_submatrix_a <- count_matrix[, c(first_start:first_right_border)]

    count_submatrix_b <- count_matrix[, c(second_start:second_right_border)]

    batch_a_size <- first_right_border - first_index
    batch_b_size <- second_right_border - second_index
    if (sparse) {
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

#' Function for generating report for distance matric.
#'
#' @param a Something.
#' @param filename CSV file.
#' @param batch_size int.
#' @param metric string for matric selection.
#' @param type "gpu" or "cpu".
#' @return A list of correlation matrix, batch_a size and batch_b size.
#' @export
mtrx_distance <- function(a, filename = "", batch_size = 1000, metric = "kendall",type="gpu", sparse = F, write=F)
{
  if (sparse) {
    a <- as(a, 'CsparseMatrix')
  }
  n <- nrow(a)
  m <- ncol(a)
  print(filename)
  if (filename == ""){
    result_overall <- double(m * m)
    dim(result_overall) <- c(m, m)
  
    colnames(result_overall) = colnames(a)
    rownames(result_overall) = colnames(a)
  }
  for (first_index in seq(0, m - 1, by=batch_size)) {

    for (second_index in seq(0, m - 1, by=batch_size)) {
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
                    sparse=sparse
                )
            } else if (type=="cpu") { 
                result <- process_batch_cpu(
                    count_matrix = a,
                    first_index = first_index,
                    second_index = second_index,
                    batch_size = batch_size,
                    metric = metric,
                    sparse = sparse
                )
            }
            a_left = first_index + 1
            a_right = first_index + result$batch_a_size
            b_left = second_index + 1
            b_right = second_index + result$batch_b_size
            
            correlation_matrix <- result$correlation_matrix

            if (filename == "") {
                result_overall[c(a_left:a_right), c(b_left:b_right)] <- correlation_matrix
            } else if (write) {
                output_filename <- glue('{filename}_{a_left}_{a_right}_{b_left}_{b_right}.csv')
                write.csv(correlation_matrix, output_filename)
            }
        }
        end_t <- as.numeric(Sys.time()) * 1000000
    }
  }
  if (write) {
    return (result_overall)
  }

  return (TRUE)
}

