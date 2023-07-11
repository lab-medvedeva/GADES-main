if(!is.loaded("matrix_Kendall_distance_same_block")) {
        library.dynam('mtrx', package = 'HobotnicaGPU', lib.loc = NULL)
}
if(!is.loaded("matrix_Kendall_distance_same_block_cpu")) {
       library.dynam('mtrx_cpu', package = 'HobotnicaGPU', lib.loc = NULL)
    }

.C("check_gpu", PACKAGE = "mtrx")
#dyn.load("../lib/mtrx.so")
#dyn.load("../src/mtrx_cpu.so")
#source("mtrx.R")
library(amap)
library(HobotnicaGPU)
library(glue)
#library(MASS)
#library(philentropy)
library("factoextra")

args = commandArgs(trailingOnly=TRUE)
datain = args[1]
method = args[2]
times = strtoi(args[3])
metric = args[4]

dataout = "Kendall_2.csv"
print('Reading table')
print('Completed reading')

measurements <- numeric(times)

for (i in 1:times) {
    fname <- glue('{datain}/{i}.csv')
    print(fname)
    #fname <- datain
    data <- t(as.matrix(read.table(fname, header=T, row.names = 1, sep=",")))
    st_t <- as.numeric(Sys.time()) * 1000000
#data_matrix <- as.matrix(data)

    if (method == 'GPU') {
        #print(metric)
        distMatrix_mtrx <- mtrx_distance(data, batch_size = 5000, metric = metric,type="gpu")
        #print(dim(distMatrix_mtrx))
    } else if (method == 'CPU') {
        #print(metric)
        #library.dynam()
        distMatrix_mtrx <- mtrx_distance(data, batch_size = 5000, metric = metric,type="cpu")
        #print(dim(distMatrix_mtrx))
    } else if (method == 'amap') {
        print('Calc dist')
        distMatrix_mtrx <- as.matrix(Dist(t(data), method=metric, nbproc=24))
        #print(distMatrix_mtrx)
    } else if (method == 'factoextra') {
        distMatrix_mtrx <- as.matrix(get_dist(data, method = metric))
    } else if (method == 'philentropy') {
    	distMatrix_mtrx <- as.matrix(philentropy::distance(t(data), method=metric))
    }
    end_time <- as.numeric(Sys.time()) * 1000000

    measurements[i] <- end_time - st_t
    gc()
    #print(as.numeric(Sys.time()) * 1000000 - st_t)
}

print(mean(as.matrix(measurements)))
print(sd(as.matrix(measurements)))
print(as.matrix(measurements))
print('Matrix')
result = glue('{datain}.out.{metric}.csv')
write.table(measurements[2:times], result, sep=',')
#write.table(distMatrix_mtrx, 'matrix.csv', sep=',')

#' function for nothing...
