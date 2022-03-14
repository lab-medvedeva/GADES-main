#! /bin/bash

echo "starting"
/usr/local/cuda-10.0/bin/nvcc -arch=sm_35 -g -G -O2 -I/usr/local/lib/R/include -Xcompiler "-Wall -fpic" -dc main.cu main.o
/usr/local/cuda-10.0/bin/nvcc -arch=sm_35 -Xcompiler "-Wall -fpic" -dc matrixgpufunctions.cu -o matrixgpufunctions.o
/usr/local/cuda-10.0/bin/nvcc -arch=sm_35 -Xcompiler "-Wall -fpic" -dc dmath.cu -o dmath.o
/usr/local/cuda-10.0/bin/nvcc -shared -arch=sm_35 main.o dmath.o matrixgpufunctions.o -L/usr/local/cuda-10.0/lib64 -lcudadevrt -o mtrx.so -L/usr/lib/R/lib -lR
echo "done"
