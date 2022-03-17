#! /bin/bash
nvcc -arch=sm_35 -g -G -O2 -I/usr/share/R/include -Xcompiler "-Wall -fpic" -dc main.cu
nvcc -shared -arch=sm_35 main.o -L/usr/local/cuda/lib64 -lcudadevrt -o mtrx.so -L/usr/share/R/lib -lR
