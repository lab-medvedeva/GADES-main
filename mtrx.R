mtrx_Kendall_distance <- function(a, filename = "")
{
  if(!is.loaded("matrix_Kendall_distance")) {
    dyn.load("mtrx.so")
  }
  n <- nrow(a)
  m <- ncol(a)
  if (filename == ""){
    result <- .C("matrix_Kendall_distance",
            data = as.double(a),
            dist_matrix = double(m*m),
            rows = as.integer(n),
            cols = as.integer(m))$dist_matrix
    dim(result) <- c(m, m)
    colnames(result) <- colnames(a)
    rownames(result) <- colnames(a)
    return(result)
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
