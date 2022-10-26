#dyn.load("mtrx.so")
dyn.load("mtrx_cpu.so")
source("mtrx.R")
library(amap)
#library(edgeR)
#library(biomaRt)
library(Hobotnica)
library(MASS)
#library("factoextra")

args = commandArgs(trailingOnly=TRUE)
datain = args[1]
method = args[2]
times = strtoi(args[3])
metric = args[4]

dataout = "Kendall_2.csv"
print('Reading table')
data <- t(as.matrix(read.table(datain, header=T, row.names = 1, sep=",")))
print('Completed reading')

measurements <- numeric(times)

for (i in 1:times) {
    st_t <- as.numeric(Sys.time()) * 1000000
#data_matrix <- as.matrix(data)

    if (method == 'GPU') {
        print(metric)
        distMatrix_mtrx <- mtrx_Kendall_distance(data, batch_size = 10000, metric = metric)
        print(dim(distMatrix_mtrx))
    } else if (method == 'amap') {
        print('Calc dist')
        distMatrix_mtrx <- as.matrix(Dist(t(data), method=metric, nbproc=24))
        #print(distMatrix_mtrx)
    } else if (method == 'factoextra') {
        distMatrix_mtrx <- as.matrix(get_dist(data, method = metric))
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
print(distMatrix_mtrx[1:10, 1:10])
cat('\n', file = dataout, append = TRUE)
