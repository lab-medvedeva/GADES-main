# --- Server load: no installed HobotnicaGPU package; dyn.load the prebuilt .so
# --- + source the R interface (same as validate.R / run_real.R). Competitor and
# --- profiling libraries are loaded only if present, so GADES-only runs work
# --- without amap/factoextra/R.utils/profmem installed.
.GADES <- "/8tbsata/Science/BioInfo/Hobotnica/Hobotnica-GPU"
if (!is.loaded("matrix_Kendall_distance_same_block"))     dyn.load(file.path(.GADES, "build/mtrx.so"))
if (!is.loaded("matrix_Kendall_distance_same_block_cpu")) dyn.load(file.path(.GADES, "build/mtrx_cpu.so"))
suppressMessages({library(Matrix); library(glue)})
source(file.path(.GADES, "R/mtrx.R"))
for (.p in c("R.utils","amap","factoextra","profmem"))
  suppressWarnings(suppressMessages(if (requireNamespace(.p, quietly=TRUE)) library(.p, character.only=TRUE)))
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

transpose_data = TRUE
if (length(args) >= 9) {
    transpose_data = as.logical(args[9])
}

sparse_layout = "default"
if (length(args) >= 10) {
    sparse_layout = args[10]
}

if (profile) {
    library(profmem)
}
#filename = args[8]

print('Reading table')
print(glue("{output}_{method}_{metric}.csv"))
print(sparse)
print(glue("transpose={transpose_data}"))

if (sparse) {
    data <- readMM(datain)
    if (transpose_data) data <- t(data)
} else {
    if (grepl("\\.mtx$", datain, ignore.case = TRUE)) {
        data <- readMM(datain)
        if (transpose_data) data <- t(data)
        data <- as.matrix(data)
    } else {
        data <- as.matrix(read.table(datain, header=T, row.names = 1, sep=","))
        if (transpose_data) data <- t(data)
    }
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
        distMatrix_mtrx <- mtrx_distance(data, batch_size = batch_size , metric = metric,type="gpu",sparse=sparse, filename=filename, sparse_layout=sparse_layout, memory_limit_gb = as.numeric(Sys.getenv("MEM_LIMIT_GB", "12")))
        print(dim(distMatrix_mtrx))
    } else if (method == 'CPU') {
        print(metric)
        #library.dynam()
        distMatrix_mtrx <- mtrx_distance(data, batch_size = batch_size, metric = metric, type="cpu", sparse=sparse, filename=filename, sparse_layout=sparse_layout, memory_limit_gb = as.numeric(Sys.getenv("MEM_LIMIT_GB", "12")))
        print(dim(distMatrix_mtrx))
    } else if (method == 'amap') {
        
        print('Calc dist')
        if (metric == 'cosine') {
            metric = 'correlation'
        }
        distMatrix_mtrx <- as.matrix(Dist(t_mtx, method=metric, nbproc=24))
        print(dim(distMatrix_mtrx))
        print('amap')
        if (metric == 'correlation') {
            metric = 'cosine'
        }
        #print(distMatrix_mtrx)
    } else if (method == 'factoextra') {
        #if (!sparse) {
        #    data <- as.matrix(data)
        #}
        if (metric == 'cosine') {
            metric = 'pearson'
        }
        distMatrix_mtrx <- as.matrix(get_dist(t_mtx, method=metric))
        if (metric == 'pearson') {
            metric = 'cosine'
        }
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
