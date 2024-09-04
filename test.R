if(!is.loaded("matrix_Kendall_distance_same_block")) {
        library.dynam('mtrx', package = 'HobotnicaGPU', lib.loc = NULL)
}
if(!is.loaded("matrix_Kendall_distance_same_block_cpu")) {
       library.dynam('mtrx_cpu', package = 'HobotnicaGPU', lib.loc = NULL)
    }

#.C("check_gpu", PACKAGE = "mtrx")
#dyn.load("./lib/mtrx.so")
#dyn.load("./lib/mtrx_cpu.so")
#source("./R/mtrx.R")

library(amap)
library(HobotnicaGPU)
library(Matrix)
library("factoextra")
library(glue)

args = commandArgs(trailingOnly=TRUE)
datain = args[1]
method = args[2]
times = strtoi(args[3])
metric = args[4]
batch_size = strtoi(args[5])
output = args[6]
print(args)
sparse = as.logical(args[7])
filename = args[8]

profile = as.logical(args[9])

if (profile) {
    library(profmem)
}
print('Reading table')
print(glue("{output}_{method}_{metric}.csv"))
print(sparse)

data <- readMM(datain)
#data <- t(as.matrix(read.table(datain, header=T, row.names = 1, sep=",")))
print('Completed reading')

measurements <- numeric(times)

for (i in 1:times) {
    st_t <- as.numeric(Sys.time()) * 1000000

    if (method == 'GPU') {
        #print(metric)
        distMatrix_mtrx <- mtrx_distance(data, batch_size = batch_size , metric = metric,type="gpu",sparse=sparse, filename=filename)
        print(dim(distMatrix_mtrx))
    } else if (method == 'CPU') {
        print(metric)
        #library.dynam()
        distMatrix_mtrx <- mtrx_distance(data, batch_size = batch_size, metric = metric, type="cpu", sparse=sparse, filename=filename)
        print(dim(distMatrix_mtrx))
    } else if (method == 'amap') {
        print('Calc dist')
        distMatrix_mtrx <- as.matrix(Dist(t(data), method=metric, nbproc=24))
        #print(distMatrix_mtrx)
    } else if (method == 'factoextra') {
        if (!sparse) {
            data <- as.matrix(data)
        }

        distMatrix_mtrx <- as.matrix(get_dist(t(data), method = metric))
        print('Factoextra')
    } else if (method == 'philentropy') {
    	distMatrix_mtrx <- as.matrix(philentropy::distance(t(data), method=metric))
    }
    end_time <- as.numeric(Sys.time()) * 1000000

    measurements[i] <- end_time - st_t
    write.table(measurements, glue("{output}_{method}_{metric}.csv"), sep=',')

    gc()
}

print('Sparse Matrix')

print(mean(as.matrix(measurements[2:times])))
print(sd(as.matrix(measurements[2:times])))
print(as.matrix(measurements))
print('Matrix')

write.table(measurements, glue("{output}_{method}_{metric}.csv"), sep=',')
write.table(measurements, 'matrix.csv', sep=',')
#print(dim(distMatrix_mtrx))
#print(distMatrix_mtrx[1:10, 1:10])
