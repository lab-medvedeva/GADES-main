library(R.utils)

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
library(profmem)
args = commandArgs(trailingOnly=TRUE)
datain = args[1]
method = args[2]
times = strtoi(args[3])
metric = args[4]
#num_batches = strtoi(args[5])
batch_size = strtoi(args[5])
output = args[6]
print(args)
sparse = as.logical(args[7])
filename = "out"

profile = as.logical(args[8])

if (profile) {
    library(profmem)
}
#filename = args[8]

print('Reading table')
print(glue("{output}_{method}_{metric}.csv"))
print(sparse)

if (sparse) {
    data <- readMM(datain)
} else {
    data <- t(as.matrix(read.table(datain, header=T, row.names = 1, sep=",")))
}

#batch_size <- ceiling(dim(data)[2] / num_batches)
#batch_size <- 5000
print(batch_size)
print('Completed reading')
print(times)
print(dim(data))

if (method != 'GPU' && method != 'CPU') {
    t_mtx <- t(data)
}
#if (dim(data)[2] * 2 < batch_size) {
#    quit(1)
#}

measurements <- numeric(times)

memories <- matrix(0, times, 4)

for (i in 1:times) {
    st_t <- as.numeric(Sys.time()) * 1000000
    if (profile) {
        profmem_begin()
        start <- as.numeric(ps::ps_memory_info()['rss'][1])
        print('Start memory')
        print(start)
    }
    if (method == 'GPU') {
        #print(metric)
        print(i)
        print('Where')
        print(batch_size)
        print(dim(data))
        distMatrix_mtrx <- mtrx_distance(data, batch_size = batch_size , metric = metric,type="gpu",sparse=sparse, filename=filename)
        print(dim(distMatrix_mtrx))
    } else if (method == 'CPU') {
        print(metric)
        #library.dynam()
        distMatrix_mtrx <- mtrx_distance(data, batch_size = batch_size, metric = metric, type="cpu", sparse=sparse, filename=filename)
        print(dim(distMatrix_mtrx))
    } else if (method == 'amap') {
        
        print('Calc dist')

        distMatrix_mtrx <- as.matrix(Dist(t_mtx, method=metric, nbproc=24))
        print(dim(distMatrix_mtrx))
        print('amap')
        #print(distMatrix_mtrx)
    } else if (method == 'factoextra') {
        #if (!sparse) {
        #    data <- as.matrix(data)
        #}
        distMatrix_mtrx <- as.matrix(get_dist(t_mtx, method=metric))
        print(dim(distMatrix_mtrx))
        print('Factoextra')
    } else if (method == 'philentropy') {
    	distMatrix_mtrx <- as.matrix(philentropy::distance(t(data), method=metric))
    }

    if (profile) {
        p <- profmem_end()
        if (method == 'amap' || method == 'factoextra') {
            delta <- object.size(distMatrix_mtrx)
        } else {
            delta <- 0
        }
        print(p)
        sum_bytes <- sum(p$bytes, na.rm=T)
        end <- as.numeric(ps::ps_memory_info()['rss'][1])
        print('Memory usage')
        print(c(sum_bytes, delta))
        
        delta_manual <- 0
        print(method)
        if (method == 'CPU' || method == 'GPU') {
            batch_size_effective <- min(dim(data)[2], batch_size)
            features <- dim(data)[1]
            print(glue('Number of cells: {dim(data)[2]}'))
            print(glue('Batch size effective: {batch_size_effective}'))
            if (!sparse) {
                delta_manual <- (batch_size_effective * batch_size_effective + 2 * batch_size_effective * features) * 4
            } else {
                delta_manual <- (batch_size_effective * batch_size_effective + 2 * length(data@x) ) * 4
            }
            print(glue('Manual delta: {delta_manual}'))
        }
        memories[i, 1] <- sum_bytes + delta_manual
        memories[i, 2] <- delta
        memories[i, 3] <- delta_manual
        memories[i, 4] <- end - start
        write.table(memories, glue("{output}_{method}_{metric}_memory.csv", sep=','))
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
