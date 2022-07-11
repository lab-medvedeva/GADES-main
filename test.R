dyn.load("mtrx.so")
source("mtrx.R")
library(amap)
library(edgeR)
library(biomaRt)
library(Hobotnica)
library(MASS)

args = commandArgs(trailingOnly=TRUE)
datain = args[1]
method = args[2]
times = strtoi(args[3])

dataout = "Kendall_2.csv"
print('Reading table')
data <- t(as.matrix(read.table(datain, header=T, row.names = 1, sep=",")))
print('Completed reading')

measurements <- numeric(times)

for (i in 1:times) {
    st_t <- as.numeric(Sys.time()) * 1000000
#data_matrix <- as.matrix(data)

    if (method == 'GPU') {
        distMatrix_mtrx <- mtrx_Kendall_distance(data, batch_size = 10000)
    } else {
        #print('Calc dist')
        distMatrix_mtrx <- Dist(t(data), method="kendall", nbproc=24)
    }
    end_time <- as.numeric(Sys.time()) * 1000000

    measurements[i] <- end_time - st_t
    #print(as.numeric(Sys.time()) * 1000000 - st_t)
}

print(mean(as.matrix(measurements)))
print(sd(as.matrix(measurements)))
print(as.matrix(measurements))
print('Matrix')
#write.table(distMatrix_mtrx, 'matrix.csv', sep=',')
#print(distMatrix_mtrx[1:10, 1:10])
cat('\n', file = dataout, append = TRUE)
