process_batch <- function(count_matrix, first_index, second_index, batch_size, metric) {
    if (first_index == second_index) {
        if (metric == 'kendall') {
            fn_name <- "matrix_Kendall_distance_same_block"
        } else if(metric =='euclidean') {
            fn_name <- "matrix_Euclidean_distance_same_block"
        } else if(metric =='pearson') {
            fn_name <- "matrix_Pearson_distance_same_block" #for not different block
        } else {
	    fn_name <- "matrix_Kendall_distance_same_block"
	}
    } else {
        if (metric == 'kendall') {
            fn_name <- "matrix_Kendall_distance_different_blocks"
        } else if(metric =='euclidean') {
            fn_name <- "matrix_Euclidean_distance_different_blocks"
        } else if(metric =='pearson') {
            fn_name <- "matrix_Pearson_distance_different_blocks"
        } else {
	    fn_name <-"matrix_kendall_distance_different_blocks"
	}
    }
    print(fn_name)
    first_right_border <- min(first_index + batch_size, ncol(count_matrix))
    second_right_border <- min(second_index + batch_size, ncol(count_matrix))

    first_start <- first_index + 1
    second_start <- second_index + 1
    count_submatrix_a <- count_matrix[, c(first_start:first_right_border)]
    count_submatrix_b <- count_matrix[, c(second_start:second_right_border)]

    batch_a_size <- first_right_border - first_index
    batch_b_size <- second_right_border - second_index

    result <- .C(
        fn_name,
        matrix_a = as.double(count_submatrix_a),
        matrix_b = as.double(count_submatrix_b),
        dist_matrix = double(batch_a_size * batch_b_size),
        rows = as.integer(nrow(count_matrix)),
        cols_a = as.integer(batch_a_size),
        cols_b = as.integer(batch_b_size)
    )$dist_matrix

    dim(result) <- c(batch_a_size, batch_b_size)

    return (
        list(
            correlation_matrix=result,
            batch_a_size=batch_a_size,
            batch_b_size=batch_b_size
        )
    )
}
process_batch_cpu <- function(count_matrix, first_index, second_index, batch_size, metric) {
    if (first_index == second_index) {
        if (metric == 'kendall') {
            fn_name <- "matrix_Kendall_distance_same_block_cpu"
        } else if(metric =='euclidean') {
            fn_name <- "matrix_Euclidean_distance_same_block_cpu"
        } else if(metric =='pearson') {
            fn_name <- "matrix_Pearson_distance_different_blocks_cpu" #For not different block
        } else {
	    fn_name <- "matrix_Kendall_distance_same_block_cpu"
	}
    } else {
        if (metric == 'kendall') {
            fn_name <- "matrix_Kendall_distance_different_blocks_cpu"
        } else if(metric =='euclidean') {
            fn_name <- "matrix_Euclidean_distance_different_blocks_cpu"
        } else if(metric =='pearson') {
            fn_name <- "matrix_Pearson_distance_different_blocks_cpu"
        }
        else {
	    fn_name <-"matrix_kendall_distance_different_blocks_cpu"
	}
       #fn_name <- "matrix_Kendall_distance_different_blocks_cpu"
    }
    print(fn_name)
    first_right_border <- min(first_index + batch_size, ncol(count_matrix))
    second_right_border <- min(second_index + batch_size, ncol(count_matrix))

    first_start <- first_index + 1
    second_start <- second_index + 1
    count_submatrix_a <- count_matrix[, c(first_start:first_right_border)]
    count_submatrix_b <- count_matrix[, c(second_start:second_right_border)]

    batch_a_size <- first_right_border - first_index
    batch_b_size <- second_right_border - second_index

    result <- .C(
        fn_name,
        matrix_a = as.double(count_submatrix_a),
        matrix_b = as.double(count_submatrix_b),
        dist_matrix = double(batch_a_size * batch_b_size),
        rows = as.integer(nrow(count_matrix)),
        cols_a = as.integer(batch_a_size),
        cols_b = as.integer(batch_b_size)
    )$dist_matrix

    dim(result) <- c(batch_a_size, batch_b_size)

    return (
        list(
            correlation_matrix=result,
            batch_a_size=batch_a_size,
            batch_b_size=batch_b_size
        )
    )
}


mtrx_distance <- function(a, filename = "", batch_size = 1000, metric = "kendall",type="gpu")
{
  if(!is.loaded("matrix_Kendall_distance_same_block")) {
    #dyn.load("mtrx.so")
  }
  n <- nrow(a)
  m <- ncol(a)


  result_overall <- double(m * m)
  dim(result_overall) <- c(m, m)
  
  colnames(result_overall) = colnames(a)
  rownames(result_overall) = colnames(a)

  if (filename == ""){
    for (first_index in seq(0, m - 1, by=batch_size)) {

        for (second_index in seq(0, m - 1, by=batch_size)) {
            if(type=="gpu"){
		result <- process_batch(
                	count_matrix = a,
                	first_index = first_index,
                	second_index = second_index,
                	batch_size = batch_size,
                	metric = metric)
	    } else if (type=="cpu"){ 
		result <- process_batch_cpu(
                	count_matrix = a,
                	first_index = first_index,
                	second_index = second_index,
                	batch_size = batch_size,
                	metric = metric)
            }
            a_left = first_index + 1
            a_right = first_index + result$batch_a_size
            b_left = second_index + 1
            b_right = second_index + result$batch_b_size
            
            correlation_matrix <- result$correlation_matrix
            result_overall[c(a_left:a_right), c(b_left:b_right)] <- correlation_matrix
            
        }
    } 
    

    return (result_overall)
  }
  else{
  RESULTFILE <- file(paste(filename,"kdm",sep="."), "wb")
  writeBin(as.integer(m), RESULTFILE, size = 4)
  if (length(colnames(a)) != 0){
    writeBin(colnames(a), RESULTFILE)
  }
  else{
    writeBin(as.character(c(1:m)), RESULTFILE)
  }
  close(RESULTFILE)
  result <- .C("file_Kendall_distance",
          data = as.double(a),
          rows = as.integer(n),
          cols = as.integer(m),
          fout = as.character(paste(filename,"kdm",sep=".")))
  }
}


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
