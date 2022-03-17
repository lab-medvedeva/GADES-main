dyn.load("mtrx.so")
source("mtrx.R")

args = commandArgs(trailingOnly=TRUE)
datain = args[1]
dataout = "Kendall_2.csv"
print('Reading table')
data <- t(as.matrix(read.table(datain, header=T, row.names = 1, sep=",")))
print('Completed reading')
st_t = as.numeric(Sys.time()) * 1000000
distMatrix_mtrx <- mtrx_Kendall_distance(data)
cat(as.numeric(Sys.time()) * 1000000 - st_t, file = dataout, append = TRUE)
cat('\n', file = dataout, append = TRUE)
