dyn.load("mtrx.so")
source("mtrx.R")

args = commandArgs(trailingOnly=TRUE)
datain = args[1]
dataout = "Kendall_2.csv"
print('Reading table')
data <- t(as.matrix(read.table(datain, header=T, row.names = 1, sep=",")))
print('Completed reading')
st_t = as.numeric(Sys.time()) * 1000000
distMatrix_mtrx <- mtrx_Kendall_distance(data, batch_size = 50)
print(distMatrix_mtrx[1, c(101:104)])
distMatrix_mtrx_big_batch_size <- mtrx_Kendall_distance(data, batch_size = 200)
print(distMatrix_mtrx_big_batch_size[1, c(101:104)])
cat(as.numeric(Sys.time()) * 1000000 - st_t, file = dataout, append = TRUE)
cat('\n', file = dataout, append = TRUE)
