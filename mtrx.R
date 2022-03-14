mtrx_mul <- function(a, b)
{
  if(!is.loaded("matrix_multiplication")) {
    dyn.load("mtrx.so")
  }
  tmp_a = t(a)
  tmp_b = t(b)
  n1 <- nrow(a)
  m1 <- ncol(a)
  n2 <- nrow(b)
  m2 <- ncol(b)
  c <- vector(length = n1 * m2)
  rst <- .C("matrix_multiplication",
            as.double(tmp_a),
            as.double(tmp_b),
            as.double(c),
            as.integer(n1),
            as.integer(m1),
            as.integer(n2),
            as.integer(m2))
  tmp <- as.vector(rst[[3]])
  dim(tmp) <- c(n1, m2)
  return(tmp)
}

mtrx_L1 <- function(a)
{
  if(!is.loaded("matrix_L1_distance")) {
    dyn.load("mtrx.so")
  }
  tmp_a = t(a)
  n <- nrow(a)
  m <- ncol(a)
  c <- vector(length = m * m)
  rst <- .C("matrix_L1_distance",
            as.double(tmp_a),
            as.double(c),
            as.integer(n),
            as.integer(m))
  tmp <- as.vector(rst[[2]])
  dim(tmp) <- c(m, m)
  return(tmp)
}

mtrx_L2 <- function(a)
{
  if(!is.loaded("matrix_L2_distance")) {
    dyn.load("mtrx.so")
  }
  tmp_a = t(a)
  n <- nrow(a)
  m <- ncol(a)
  c <- vector(length = m * m)
  rst <- .C("matrix_L2_distance",
            as.double(tmp_a),
            as.double(c),
            as.integer(n),
            as.integer(m))
  tmp <- as.vector(rst[[2]])
  dim(tmp) <- c(m, m)
  return(tmp)
}

mtrx_Linf <- function(a)
{
  if(!is.loaded("matrix_Linf_distance")) {
    dyn.load("mtrx.so")
  }
  tmp_a = t(a)
  n <- nrow(a)
  m <- ncol(a)
  c <- vector(length = m * m)
  rst <- .C("matrix_Linf_distance",
            as.double(tmp_a),
            as.double(c),
            as.integer(n),
            as.integer(m))
  tmp <- as.vector(rst[[2]])
  dim(tmp) <- c(m, m)
  return(tmp)
}

mtrx_Kendall <- function(a)
{
  if(!is.loaded("matrix_Kendall_distance")) {
    dyn.load("mtrx.so")
  }
  tmp_a = t(a)
  n <- nrow(a)
  m <- ncol(a)
  c <- vector(length = m * m)
  rst <- .C("matrix_Kendall_distance",
            as.double(tmp_a),
            as.double(c),
            as.integer(n),
            as.integer(m))
  tmp <- as.vector(rst[[2]])
  dim(tmp) <- c(m, m)
  return(tmp)
}

mtrx_Kendall_naive <- function(a)
{
  if(!is.loaded("matrix_Kendall_distance_naive")) {
    dyn.load("mtrx.so")
  }
  tmp_a = t(a)
  n <- nrow(a)
  m <- ncol(a)
  c <- vector(length = m * m)
  rst <- .C("matrix_Kendall_distance_naive",
            as.double(tmp_a),
            as.double(c),
            as.integer(n),
            as.integer(m))
  tmp <- as.vector(rst[[2]])
  dim(tmp) <- c(m, m)
  return(tmp)
}
