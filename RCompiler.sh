
#! /bin/bash
/usr/local/cuda-10.0/bin/nvcc -arch=sm_35 -g -G -O2 -I/usr/local/lib/R/include -Xcompiler "-Wall -fpic" -dc main.cu
/usr/local/cuda-10.0/bin/nvcc -shared -arch=sm_35 main.o -L/usr/local/cuda-10.0/lib64 -lcudadevrt -o mtrx.so -L/usr/lib/R/lib -lR
