#include <time.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cstring>
#include <R.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void FinalizeCosine(int columns, float* results, float* x_squares, float* y_squares) {
    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
    int column_index = blockDim.y * blockIdx.y + threadIdx.y;

    int index = row_index * columns + column_index;

    if ((row_index < columns) && (column_index < columns)) {
        if (row_index < column_index) {
            results[index] = 1.0 - results[index] / sqrtf(x_squares[index]) / sqrtf(y_squares[index]);
        } else if (row_index > column_index) {
            results[index] = results[columns * column_index + row_index];
        }
    }
}

__global__ void FinalizeCosineSparse(int columns, float* results, float* squares) {
    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
    int column_index = blockDim.y * blockIdx.y + threadIdx.y;

    int index = row_index * columns + column_index;

    if ((row_index < columns) && (column_index < columns)) {
        if (row_index < column_index) {
            results[index] = 1.0 - results[index] / sqrtf(squares[row_index]) / sqrtf(squares[column_index]);
        } else if (row_index > column_index) {
            results[index] = results[columns * column_index + row_index];
        }
    }
}

__global__ void FinalizePearsonSparse(int columns, float* results, float* squares,
                                      float* col_sums, int rows) {
    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
    int column_index = blockDim.y * blockIdx.y + threadIdx.y;

    int index = row_index * columns + column_index;

    if ((row_index < columns) && (column_index < columns)) {
        if (row_index < column_index) {
            float dot_centered = results[index]
                - col_sums[row_index] * col_sums[column_index] / rows;
            float norm1 = sqrtf(squares[row_index]
                - col_sums[row_index] * col_sums[row_index] / rows);
            float norm2 = sqrtf(squares[column_index]
                - col_sums[column_index] * col_sums[column_index] / rows);
            results[index] = 1.0f - dot_centered / (norm1 * norm2);
        } else if (row_index > column_index) {
            results[index] = results[columns * column_index + row_index];
        }
    }
}


__global__ void FinalizeEuclidean(int columns, float* results) {
    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
    int column_index = blockDim.y * blockIdx.y + threadIdx.y;

    int index = row_index * columns + column_index;

    if ((row_index < columns) && (column_index < columns)) {
        if (row_index < column_index) {
            results[index] = sqrtf(results[index]);
        } else if (row_index > column_index) {
            results[index] = results[columns * column_index + row_index];
        }
    }
}


__global__ void Rkendall_gpu_atomic_float(float* array, const int n, const int m, unsigned int* result) {
  
  int row1 = blockIdx.y * blockDim.y + threadIdx.y;
  int row2 = blockIdx.x * blockDim.x + threadIdx.x;

  if (row1 >= row2 || row1 >= n || row2 >= n) {
    return;
  }

  //if (row2 % 5000 == 0 && row1 % 500 == 0) {
  //  printf("%d %d %d\n", row1, row2, m);
  //}
  for (int col1_num = 0; col1_num < m; ++col1_num) {
      for (int col2_num = col1_num + 1; col2_num < m; ++col2_num) {
          float* col1 = array + n * col1_num;
          float* col2 = array + n * col2_num;

          if ((col1[row1] - col1[row2]) * (col2[row1] - col2[row2]) < 0){
            atomicAdd(result + col1_num * m + col2_num, 1);
//              atomicAdd(result + col2_num * m + col1_num, 1);
          }
      }
  }
}


__global__ void FinalizeManhattan(int columns, float* results) {
    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
    int column_index = blockDim.y * blockIdx.y + threadIdx.y;

    int index = row_index * columns + column_index;

    if ((row_index < columns) && (column_index < columns)) {
        if (row_index > column_index) {
            results[index] = results[columns * column_index + row_index];
        }
    }
}

__global__ void Rmanhattan_gpu_row_atomic_float(float* array, const int n, const int m, float* result) {
    int col1_num = blockIdx.x * blockDim.x + threadIdx.x;
    int col2_num = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    if (col1_num < col2_num && col2_num < m) {
        float* col1 = array + n * col1_num;
        float* col2 = array + n * col2_num;
        for (int i = 0; i < n; ++i) {
            sum += fabsf(col1[i] - col2[i]);
        }
        result[col1_num * m + col2_num] = sum;
    }
}

__global__ void Rmanhattan_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, float* result) {
    int col1_num = blockIdx.x * blockDim.x + threadIdx.x;
    int col2_num = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    if (col1_num < m && col2_num < m_b) {
        float* col1 = array + n * col1_num;
        float* col2 = array2 + n * col2_num;
        for (int i = 0; i < n; ++i) {
            sum += fabsf(col1[i] - col2[i]);
        }
        result[col2_num * m + col1_num] = sum;
    }
}

__global__ void Reuclidean_gpu_row_atomic_float(float* array, const int n, const int m, float* result) {
    int col1_num = blockIdx.x * blockDim.x + threadIdx.x;
    int col2_num = blockIdx.y * blockDim.y + threadIdx.y;
    
    int sum = 0.0f;
    
    int local_tid = threadIdx.x * blockDim.x + threadIdx.y;

    if (col1_num < col2_num && col2_num < m) {
        float* col1 = array + n * col1_num;
        float* col2 = array + n * col2_num;
        float diff = 0;
        for (int i = 0; i < n; ++i) {
            diff = col1[i] - col2[i];
            diff = diff * diff;
            sum += diff;
        }
        result[col1_num * m + col2_num] = sum;
    }
}

__global__ void Reuclidean_gpu_atomic_float(float* array, const int n, const int m, float* result) {
  
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  for (int col1_num = 0; col1_num < m; ++col1_num) {
      for (int col2_num = col1_num + 1; col2_num < m; ++col2_num) {
          float* col1 = array + n * col1_num;
          float* col2 = array + n * col2_num;

          if (row < n) {
            float diff = col1[row] - col2[row];
            diff = diff * diff;
            atomicAdd(result + col1_num * m + col2_num, diff);

          }
      }
  }
}

__global__ void Rkendall_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, unsigned int* result) {
  
  int row1 = blockIdx.y * blockDim.y + threadIdx.y;
  int row2 = blockIdx.x * blockDim.x + threadIdx.x;

  if (row1 >= row2 || row2 >= n || row1 >= n) {
    return;
  }
  for (int col1_num = 0; col1_num < m; ++col1_num) {
      for (int col2_num = 0; col2_num < m_b; ++col2_num) {
          float* col1 = array + n * col1_num;
          float* col2 = array2 + n * col2_num;

          if ((col1[row1] - col1[row2]) * (col2[row1] - col2[row2]) < 0){
              atomicAdd(result + col2_num * m + col1_num, 1);
          }
      }
  }
}
__global__ void Reuclidean_gpu_atomic_row_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, float* result) {
    int col1_num = blockIdx.x * blockDim.x + threadIdx.x;
    int col2_num = blockIdx.y * blockDim.y + threadIdx.y;
    
    int sum = 0.0f;
    
    int local_tid = threadIdx.x * blockDim.x + threadIdx.y;

    if (col1_num < m && col2_num < m_b) {
        float* col1 = array + n * col1_num;
        float* col2 = array2 + n * col2_num;
        float diff = 0;
        for (int i = 0; i < n; ++i) {
            diff = col1[i] - col2[i];
            diff = diff * diff;
            sum += diff;
        }
        result[col2_num * m + col1_num] = sqrtf(sum);
    }

}

__global__ void Reuclidean_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, float* result) {
  
  int row = blockIdx.x * blockDim.x + threadIdx.x;


  for (int col1_num = 0; col1_num < m; ++col1_num) {
      for (int col2_num = 0; col2_num < m_b; ++col2_num) {
          float* col1 = array + n * col1_num;
          float* col2 = array2 + n * col2_num;

          if (row < n) {
            float diff = col1[row] - col2[row];
            diff = diff * diff;
            //atomicAdd(result + col1_num * m + col2_num, diff);
            atomicAdd(result + col2_num * m + col1_num, diff);

          }
      }
  }
}

__global__ void RcosineCorr_gpu_atomic_float_same_block(
  float* array,
  const int n, const int m,
  float* scalar_product,
  float* x_norm,
  float* y_norm
) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  for (int col1_num = 0; col1_num < m; ++col1_num) {
      for (int col2_num = col1_num+1; col2_num < m; ++col2_num) {
          float* col1 = array + n * col1_num;
          float* col2 = array + n * col2_num;

          if (row < n) {
            float diff = col1[row] * col2[row];
            atomicAdd(scalar_product + col1_num * m + col2_num, diff);

            float x_element_norm = col1[row] * col1[row];
            float y_element_norm = col2[row] * col2[row];

            atomicAdd(x_norm + col1_num * m + col2_num, x_element_norm);
            atomicAdd(y_norm + col1_num * m + col2_num, y_element_norm);
	
          }
      }
  }
}

__global__ void RcosineCorr_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, float* scalar_prod,float *x_norm, float* y_norm ){
    int row = blockIdx.x*blockDim.x +threadIdx.x;
    for (int col1_num=0;col1_num<m;++col1_num){
	for(int col2_num=0;col2_num<m_b;++col2_num){
		float* col1 = array + n * col1_num;
		float* col2 = array2 + n * col2_num;
		if(row<n) {
				float num = (col1[row] * col2[row]);
				float sum1 = (col1[row] * col1[row]);
				float sum2 = (col2[row] * col2[row]);
				//if(dist==1){}
				atomicAdd(scalar_prod+col1_num*m_b+col2_num,num);
				atomicAdd(x_norm+col1_num*m_b+col2_num,sum1);
				atomicAdd(y_norm+col1_num*m_b+col2_num,sum2);
				//atomicAdd(scalar_prod+col2_num*m+col1_num,num);
				//atomicAdd(x_norm+col2_num*m+col1_num,sum1);
				//atomicAdd(y_norm+col2_num*m+col1_num,sum2);
				//!debug if(threadIdx.x==0){printf("val1=%4.2f, val2=%4.2f, num=%4.2f, sum1=%4.2f, sum2=%4.2f, \n", col1[row],col2[row],num,sum1,sum2);}
			}
		}
	}
}


extern "C" bool check_gpu() {
    cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    return true;
}

//' Driver Function for calculation of Kendall matrix for same block.
//'
//' Allocates Memory required for the operation. Then,
//' efficiently calculate the distance matrix using the kernel, 
//' which is translated to appropriate R tables.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//' 
extern "C" void matrix_Kendall_distance_same_block(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){

  clock_t start_1, end;
  cudaEvent_t start, stop1, stop2, stop3;
  double cpu_time_used;
  start_1 = clock();

  float milliseconds = 0.0f;
  int array_size = *n * *m;

  float* array_new = new float[*n * *m];

  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }
  end = clock();
  cpu_time_used = ((double) (end - start_1)) / CLOCKS_PER_SEC * 1000.0;
  Rprintf("%lf create array time \n", cpu_time_used);

  //cudaEventCreate(&start);
  //cudaEventCreate(&stop1);
  //cudaEventCreate(&stop2);
  //cudaEventCreate(&stop3);
  //cudaEventRecord(start);
  float* d_array;  
  
  cudaMalloc(&d_array, array_size * sizeof(float));

  end = clock();

  cpu_time_used = ((double) (end - start_1)) / CLOCKS_PER_SEC * 1000.0;
  Rprintf("%lf create event and record \n", cpu_time_used);

  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
  //cudaEventRecord(stop1);
  //cudaEventSynchronize(stop1);
  end = clock();
  cpu_time_used = ((double) (end - start_1)) / CLOCKS_PER_SEC * 1000;
  Rprintf("%lf stop1 time \n", cpu_time_used);
  //cudaEventElapsedTime(&milliseconds, start, stop1);

  //Rprintf("%f memcpy\n", milliseconds);
  int threads = 32;
  int blocks_in_row = (*n + threads - 1) / threads;
  int blocks_in_col = (*n + threads - 1) / threads;

  dim3 THREADS(threads, threads);
  dim3 BLOCKS(blocks_in_row, blocks_in_col);

  unsigned int* d_result;
  unsigned int* h_result = new unsigned int[(*m) * (*m)];
  cudaMalloc(&d_result, (*m) * (*m) * sizeof(unsigned int));

  cudaMemset(d_result, 0, (*m) * (*m) * sizeof(unsigned int));

  //cudaEventRecord(stop2);
  //cudaEventSynchronize(stop2);

  //cudaEventElapsedTime(&milliseconds, start, stop2);

  // Rprintf("%f memset\n", milliseconds);

  Rkendall_gpu_atomic_float<<<BLOCKS, THREADS>>>(d_array, *n, *m, d_result);

  cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);

  
  for (int i = 0; i < (*m) * (*m); ++i) {
      int row_index = i / (*m);
      int column_index = i % (*m);
      if (row_index < column_index) {
        c[i] = static_cast<double>(h_result[i]) * 2.0 / (*n) / (*n - 1); //rows / (rows - 1);
      } else if (row_index > column_index) {
        c[i] = c[column_index * (*m) + row_index];
      } else {
        c[i] = 1.0;
      }
  }
  free(array_new);
  free(h_result);
  cudaFree(d_result);
  cudaFree(d_array);

  //cudaEventRecord(stop3);
  //cudaEventSynchronize(stop3);

  //cudaEventElapsedTime(&milliseconds, start, stop3);
  //Rprintf("%f kernel call\n", milliseconds);
  
  end = clock();
  cpu_time_used = ((double) (end - start_1)) / CLOCKS_PER_SEC;
  Rprintf("%lf all time \n", cpu_time_used);
}


//' Driver Function for calculation of Euclidean matrix for same block.
//'
//' Allocates Memory required for the operation. Then,
//' efficiently calculate the distance matrix using the kernel, 
//' which is translated to appropriate R tables.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//' 
extern "C" void matrix_Euclidean_distance_same_block(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){


  int array_size = *n * *m;

  float* array_new = new float[*n * *m];

  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  float* d_array;

  cudaMalloc(&d_array, array_size * sizeof(float));

  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);

  //int threads = 128;
  //int blocks_in_row = (*n + threads - 1) / threads;
  //int blocks_in_col = *n;


  float* d_result;
  float* h_result = new float[(*m) * (*m)];
  cudaMalloc(&d_result, (*m) * (*m) * sizeof(float));
  //cudaMemset(d_result, 0, (*m) * (*m) * sizeof(float));
  int columns = *m;
  dim3 block_size(32, 32);
  dim3 num_blocks((columns + 31) / 32, (columns + 31) / 32);

  Reuclidean_gpu_row_atomic_float<<<num_blocks, block_size, 1024 * sizeof(float)>>>(d_array, *n, *m, d_result);

  gpuErrchk(cudaPeekAtLastError());

  FinalizeEuclidean<<<num_blocks, block_size>>>(columns, d_result);

  cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);

  //for (int i = 0; i < (*m) * (*m); ++i) {
  //  int row_index = i / (*m);
  //    int column_index = i % (*m);
  //    if (row_index < column_index) {

  //      c[i] = sqrtf(h_result[i]);
  //    } else if (row_index > column_index) {
  //      c[i] = c[column_index * (*m) + row_index];
  //    } else {
  //      c[i] = 0.0f;
  //    }


  //}

  free(h_result);
  free(array_new);
  cudaFree(d_result);
  cudaFree(d_array);
}


//' Driver Function for calculation of Kendall matrix for different block.
//'
//' Allocates Memory required for the operation. Then,
//' efficiently calculate the distance matrix using the kernel, 
//' which is translated to appropriate R tables.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//' 
extern "C" void matrix_Kendall_distance_different_blocks(double* a, double* b, double* c, int* n, int* m, int* m_b){

  int array_size = *n * *m;
  float* array_new = new float[*n * *m];

  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  int array2_size = *n * (*m_b);
  float* array2_new = new float[array2_size];

  for (int i = 0; i < array2_size; ++i) {
    array2_new[i] = b[i];
  }

  float* d_array;
  float* d_array2;

  cudaMalloc(&d_array, array_size * sizeof(float));
  cudaMalloc(&d_array2, array_size * sizeof(float));

  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_array2, array2_new, array2_size * sizeof(float), cudaMemcpyHostToDevice);

  int threads = 32;
  int blocks_in_row = (*n + threads - 1) / threads;
  int blocks_in_col = (*n + threads - 1) / threads;

  dim3 THREADS(threads, threads);
  dim3 BLOCKS(blocks_in_row, blocks_in_col);

  unsigned int* d_result;
  unsigned int* h_result = new unsigned int[(*m) * (*m_b)];
  cudaMalloc(&d_result, (*m) * (*m_b) * sizeof(unsigned int));
  cudaMemset(d_result, 0, (*m) * (*m_b) * sizeof(unsigned int));

  Rkendall_gpu_atomic_float_different_blocks<<<BLOCKS, THREADS>>>(d_array, d_array2, *n, *m, *m_b, d_result);

  cudaMemcpy(h_result, d_result, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);


  for (int i = 0; i < (*m) * (*m_b); ++i) {
    c[i] = h_result[i] * 2.0f / (*n) / (*n - 1);
  }

  free(h_result);
  cudaFree(d_result);
  cudaFree(d_array);
  cudaFree(d_array2);
}


//' Driver Function for calculation of Euclidean matrix for different block.
//'
//' Allocates Memory required for the operation. Then,
//' efficiently calculate the distance matrix using the kernel, 
//' which is translated to appropriate R tables.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//' 
extern "C" void matrix_Euclidean_distance_different_blocks(double* a, double* b, double* c, int* n, int* m, int* m_b){

  int array_size = *n * *m;
  float* array_new = new float[*n * *m];

  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  int array2_size = *n * (*m_b);
  float* array2_new = new float[array2_size];

  for (int i = 0; i < array2_size; ++i) {
    array2_new[i] = b[i];
  }

  float* d_array;
  float* d_array2;

  cudaMalloc(&d_array, array_size * sizeof(float));
  cudaMalloc(&d_array2, array_size * sizeof(float));

  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_array2, array2_new, array2_size * sizeof(float), cudaMemcpyHostToDevice);
  int columns = *m;
  int columns_b = *m_b;
  dim3 block_size(32, 32);
  dim3 num_blocks((columns + 31) / 32, (columns_b + 31) / 32);


  float* d_result;
  float* h_result = new float[(*m) * (*m_b)];
  cudaMalloc(&d_result, (*m) * (*m_b) * sizeof(float));
  cudaMemset(d_result, 0, (*m) * (*m_b) * sizeof(float));

  Reuclidean_gpu_atomic_row_float_different_blocks<<<num_blocks, block_size>>>(d_array, d_array2, *n, *m, *m_b, d_result);

  cudaMemcpy(h_result, d_result, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
  gpuErrchk(cudaPeekAtLastError());

  //for (int i = 0; i < (*m) * (*m_b); ++i) {
  //  c[i] = sqrtf(h_result[i]);
  //}

  free(h_result);
  free(array_new);
  free(array2_new);
  cudaFree(d_result);
  cudaFree(d_array);
  cudaFree(d_array2);
}



//' Driver Function for calculation of Manhattan matrix for same block.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//'
extern "C" void matrix_Manhattan_distance_same_block(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){

  int array_size = *n * *m;

  float* array_new = new float[*n * *m];

  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  float* d_array;

  cudaMalloc(&d_array, array_size * sizeof(float));

  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);

  float* d_result;
  float* h_result = new float[(*m) * (*m)];
  cudaMalloc(&d_result, (*m) * (*m) * sizeof(float));
  cudaMemset(d_result, 0, (*m) * (*m) * sizeof(float));
  int columns = *m;
  dim3 block_size(32, 32);
  dim3 num_blocks((columns + 31) / 32, (columns + 31) / 32);

  Rmanhattan_gpu_row_atomic_float<<<num_blocks, block_size>>>(d_array, *n, *m, d_result);

  gpuErrchk(cudaPeekAtLastError());

  FinalizeManhattan<<<num_blocks, block_size>>>(columns, d_result);

  cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < (*m) * (*m); ++i) {
    c[i] = h_result[i];
  }

  free(h_result);
  free(array_new);
  cudaFree(d_result);
  cudaFree(d_array);
}

//' Driver Function for calculation of Manhattan matrix for different block.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//'
extern "C" void matrix_Manhattan_distance_different_blocks(double* a, double* b, double* c, int* n, int* m, int* m_b){

  int array_size = *n * *m;
  float* array_new = new float[*n * *m];

  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  int array2_size = *n * (*m_b);
  float* array2_new = new float[array2_size];

  for (int i = 0; i < array2_size; ++i) {
    array2_new[i] = b[i];
  }

  float* d_array;
  float* d_array2;

  cudaMalloc(&d_array, array_size * sizeof(float));
  cudaMalloc(&d_array2, array2_size * sizeof(float));

  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_array2, array2_new, array2_size * sizeof(float), cudaMemcpyHostToDevice);
  int columns = *m;
  int columns_b = *m_b;
  dim3 block_size(32, 32);
  dim3 num_blocks((columns + 31) / 32, (columns_b + 31) / 32);

  float* d_result;
  float* h_result = new float[(*m) * (*m_b)];
  cudaMalloc(&d_result, (*m) * (*m_b) * sizeof(float));
  cudaMemset(d_result, 0, (*m) * (*m_b) * sizeof(float));

  Rmanhattan_gpu_atomic_float_different_blocks<<<num_blocks, block_size>>>(d_array, d_array2, *n, *m, *m_b, d_result);

  cudaMemcpy(h_result, d_result, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
  gpuErrchk(cudaPeekAtLastError());

  for (int i = 0; i < (*m) * (*m_b); ++i) {
    c[i] = h_result[i];
  }

  free(h_result);
  free(array_new);
  free(array2_new);
  cudaFree(d_result);
  cudaFree(d_array);
  cudaFree(d_array2);
}


//' Driver Function for calculation of Cosine matrix for same block.
//'
//' Allocates Memory required for the operation. Then,
//' efficiently calculate the distance matrix using the kernel, 
//' which is translated to appropriate R tables.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//' 
extern "C" void matrix_Cosine_distance_same_block(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){
  int array_size = *n * *m;
  float* array_new = new float[*n * *m];

  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  // int array2_size = *n * (*m_b);
  // float* array2_new = new float[array2_size];

  // for (int i = 0; i < array2_size; ++i) {
  //   array2_new[i] = b[i];
  // }

  float* d_array;
  // float* d_array2;

  cudaMalloc(&d_array, array_size * sizeof(float));
  // cudaMalloc(&d_array2, array_size * sizeof(float));

  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_array2, array2_new, array2_size * sizeof(float), cudaMemcpyHostToDevice);

  int threads = 128;
  int blocks = (*n + threads - 1) / threads;
  // int blocks_in_col = (*n + threads - 1) / threads;


  float* d_result;
  float* h_result = new float[(*m) * (*m)];

  cudaMalloc(&d_result, (*m) * (*m) * sizeof(float)); 
  cudaMemset(d_result, 0, (*m) * (*m) * sizeof(float));

  float* d_x_norm_result;
  float* h_x_norm_result = new float[(*m) * (*m)];

  cudaMalloc(&d_x_norm_result, (*m) * (*m) * sizeof(float)); 
  cudaMemset(d_x_norm_result, 0, (*m) * (*m) * sizeof(float));

  float* d_y_norm_result;
  float* h_y_norm_result = new float[(*m) * (*m)];

  cudaMalloc(&d_y_norm_result, (*m) * (*m) * sizeof(float)); 
  cudaMemset(d_y_norm_result, 0, (*m) * (*m) * sizeof(float));

  RcosineCorr_gpu_atomic_float_same_block<<<blocks, threads>>>(
    d_array,
    *n, *m,
    d_result,
    d_x_norm_result,
    d_y_norm_result
  );
  int columns = (*m);

  dim3 block_size(32, 32);
  dim3 num_blocks((columns + 31) / 32, (columns + 31) / 32);

  FinalizeCosine<<<num_blocks, block_size>>>(columns, d_result, d_x_norm_result, d_y_norm_result);
  cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_x_norm_result, d_x_norm_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_y_norm_result, d_y_norm_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);
  // int j=0;
  // for (int i = 0; i < (*m) * (*m); ++i) {
    // printf("%4.2f ",h_result[i]);
  //  int row_index = i / (*m);
  //  int column_index = i % (*m);
  //  if (row_index < column_index) {
  //      if (!isnan(h_result[i])) {
  //          c[i] = 1.0 - h_result[i] / sqrtf(h_x_norm_result[i]) / sqrtf(h_y_norm_result[i]);
  //      }
  //  } else if (row_index > column_index) {
  //      c[i] = c[column_index * (*m) + row_index];
  //  } else {
  //      c[i] = 0.0f;
  //  }


  //}
  free(array_new);
  free(h_result);
  cudaFree(d_result);
  cudaFree(d_x_norm_result);
  cudaFree(d_y_norm_result);
  cudaFree(d_array);
  // cudaFree(d_array2);
}

//' Driver Function for calculation of Cosine matrix for different block.
//'
//' Allocates Memory required for the operation. Then,
//' efficiently calculate the distance matrix using the kernel, 
//' which is translated to appropriate R tables.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//' 
extern "C" void matrix_Cosine_distance_different_blocks(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){
    int array_size = *n * *m;
  float* array_new = new float[*n * *m];

  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  int array2_size = *n * (*m_b);
  float* array2_new = new float[array2_size];

  for (int i = 0; i < array2_size; ++i) {
    array2_new[i] = b[i];
  }

  float* d_array;
  float* d_array2;

  cudaMalloc(&d_array, array_size * sizeof(float));
  cudaMalloc(&d_array2, array2_size * sizeof(float));

  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_array2, array2_new, array2_size * sizeof(float), cudaMemcpyHostToDevice);

  int threads = 128;
  int blocks_in_row = (*n + threads - 1) / threads;
  int blocks_in_col = 1;

  dim3 THREADS(threads, 1);
  dim3 BLOCKS(blocks_in_row, blocks_in_col);

//  float* d_result;
//  float* h_result = new float[(*m) * (*m_b)];
  float* scalar;
  float* h_scalar = new float[(*m) * (*m_b)];
 float* prod1;
  float* h_prod1 = new float[(*m) * (*m_b)];
 float* prod2;
  float* h_prod2 = new float[(*m) * (*m_b)];

//  cudaMalloc(&d_result, (*m) * (*m_b) * sizeof(float)); 
//  cudaMemset(d_result, 0, (*m) * (*m_b) * sizeof(float));
  cudaMalloc(&scalar, (*m) * (*m_b) * sizeof(float)); 
  cudaMemset(scalar, 0, (*m) * (*m_b) * sizeof(float));
  cudaMalloc(&prod1, (*m) * (*m_b) * sizeof(float)); 
  cudaMemset(prod1, 0, (*m) * (*m_b) * sizeof(float));
  cudaMalloc(&prod2, (*m) * (*m_b) * sizeof(float)); 
  cudaMemset(prod2, 0, (*m) * (*m_b) * sizeof(float));

  RcosineCorr_gpu_atomic_float_different_blocks<<<blocks_in_row, threads>>>(d_array,d_array2, *n, *m,*m_b, scalar,prod1,prod2);
  cudaMemcpy(h_scalar, scalar, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_prod1, prod1, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_prod2, prod2, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
  
  int j=0;
  for (int i = 0; i < (*m) * (*m_b); ++i) {
    // printf("%4.2f ",h_result[i]);
    
    if(!isnan(h_scalar[i])){
      //if (i == 1 || i == (*m)) {
      //  printf("%f %f %f\n", h_result[i], h_x_norm_result[i], h_y_norm_result[i]);
      //}
      if (i == j * (*m+1)){
       c[i] = 0.0; //1.0 - h_result[i] / sqrtf(h_x_norm_result[i]) / sqrtf(h_y_norm_result[i]);
       j++;  
      } else {
        c[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);
      }
    }
  }


  free(h_scalar);
  free(h_prod2);
  free(h_prod1);
  free(array_new);
  free(array2_new);
  cudaFree(scalar);
  cudaFree(prod2);
  cudaFree(prod1);
  cudaFree(d_array);
  cudaFree(d_array2);
}

//' Driver Function for calculation of Cosine matrix for different block.
//'
//' Allocates Memory required for the operation. Then,
//' efficiently calculate the distance matrix using the kernel,
//' which is translated to appropriate R tables.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//'`
__global__ void ReuclideanSparse_gpu_atomic_float_same_block(
    int* a_index, int* a_positions, float* a_double_values,
    int rows, int columns, float* result) {

    int row_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_index < rows) {
        int start_column = a_positions[row_index];
        int end_column = a_positions[row_index + 1];

        for (int col1_index = start_column; col1_index < end_column; ++col1_index) {

            for (int col2_index = col1_index; col2_index < end_column; ++col2_index) {
                int prev_col_index = col1_index - 1;
                int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;

                int next_col_index = col2_index + 1;
                int next_col = (next_col_index < end_column) ? a_index[next_col_index] : columns;

                int col1 = a_index[col1_index];
                int col2 = a_index[col2_index];

                float value1 = a_double_values[col1_index];
                float value2 = a_double_values[col2_index];

                for (int left = prev_col + 1; left < col1; ++left) {
                    atomicAdd(&result[left * columns + col2], value2 * value2);
                }

                for (int right = col2 + 1; right < next_col; ++right) {
                    atomicAdd(&result[col1 * columns + right], value1 * value1);
                }
		// using diff calculation directcly passing to n,m col directly
                float diff = (value1 - value2) * (value1 - value2);
                atomicAdd(&result[col1 * columns + col2], diff);
            }
        }
    }
}

extern "C" void matrix_Euclidean_sparse_distance_same_block(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b) {
    int rows = *num_rows;
    int columns = *num_columns;
    int num_elements_a_int = *num_elements_a;

    float* a_values = new float[num_elements_a_int];
    float* float_result = new float[columns * columns];

    for (int i = 0; i < num_elements_a_int; ++i) {
        a_values[i] = static_cast<float>(a_double_values[i]);
    }

    for (int i = 0; i < columns * columns; ++i) {
        float_result[i] = 0.0f;
    }

    int* d_a_index;
    int* d_a_positions;
    float* d_a_values;
    float* d_result;

    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
    cudaMalloc(&d_result, columns * columns * sizeof(float));

    cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, columns * columns * sizeof(float));

    int threadsPerBlock = 128;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    ReuclideanSparse_gpu_atomic_float_same_block<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_index, d_a_positions, d_a_values, rows, columns, d_result);

    gpuErrchk( cudaPeekAtLastError() );
    
    dim3 block_size(32, 32);
    dim3 num_blocks((columns + 31) / 32, (columns + 31) / 32);

    FinalizeEuclidean<<<num_blocks, block_size>>>(columns, d_result);
    gpuErrchk( cudaPeekAtLastError() );

    cudaMemcpy(float_result, d_result, columns * columns * sizeof(float), cudaMemcpyDeviceToHost);
    gpuErrchk( cudaPeekAtLastError() );

    //for (int i = 0; i < columns * columns; ++i) {
    //    int row_index = i / columns;
    //    int column_index = i % columns;

    //    if (row_index < column_index) {
    //        result[i] = std::sqrt(float_result[i]);
    //    } else if (row_index > column_index) {
    //        result[i] = result[column_index * columns + row_index];
    //    } else {
    //        result[i] = 1.0;
    //    }
    //}

    cudaFree(d_a_index);
    cudaFree(d_a_positions);
    cudaFree(d_a_values);
    cudaFree(d_result);
    gpuErrchk(cudaPeekAtLastError());
    delete[] a_values;
    delete[] float_result;
}

//====================================TESTING SPARSE METHODS==============================
__global__ void ReuclideanSparse_gpu_atomic_float_different_blocks(
    int* a_index, int* a_positions, float* a_double_values,
    int* b_index, int* b_positions, float* b_double_values,
    int rows, int columns, int columns_b, float* result) {

    int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index < rows) {
        int start_column = a_positions[row_index];
        int end_column = a_positions[row_index + 1];

        int start_column_b = b_positions[row_index];
        int end_column_b = b_positions[row_index + 1];

        for (int col1_index = start_column; col1_index <= end_column; ++col1_index) {
            int prev_col_index = col1_index - 1;
            int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
            float value1 = (col1_index < end_column) ? a_double_values[col1_index] : 0.0f;

            int col1 = (col1_index < end_column) ? a_index[col1_index] : columns;
            for (int col2_index = start_column_b; col2_index <= end_column_b; ++col2_index) {
                int prev_col_b_index = col2_index - 1;
                int prev_col2 = (prev_col_b_index >= start_column_b) ? b_index[prev_col_b_index] : -1;

                int col2 = (col2_index < end_column_b) ? b_index[col2_index] : columns_b;

                float value2 = (col2_index < end_column_b) ? b_double_values[col2_index] : 0.0f;

                if (col2 < columns_b) {
                    for (int left = prev_col + 1; left < col1; ++left) {
                        atomicAdd(&result[col2 * columns + left], value2 * value2);
                    }
                }

                if (col1 < columns) {
                    for (int left = prev_col2 + 1; left < col2; ++left) {
                        atomicAdd(&result[left * columns + col1], value1 * value1);
                    }
                }

                if (col1 < columns && col2 < columns_b) {
                    float diff = (value1 - value2) * (value1 - value2);
                    atomicAdd(&result[col2 * columns + col1], diff);
                }
            }
        }
    }
}

extern "C" void matrix_Euclidean_sparse_distance_different_blocks(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b) {

    int rows = *num_rows;
    int columns = *num_columns;
    int columns_b = *num_columns_b;
    int num_elements_a_int = *num_elements_a;
    int num_elements_b_int = *num_elements_b;

    float* a_values = new float[num_elements_a_int];
    float* b_values = new float[num_elements_b_int];
    float* float_result = new float[columns * columns_b];

    for (int i = 0; i < num_elements_a_int; ++i) {
        a_values[i] = static_cast<float>(a_double_values[i]);
    }

    for (int i = 0; i < num_elements_b_int; ++i) {
        b_values[i] = static_cast<float>(b_double_values[i]);
    }

    for (int i = 0; i < columns * columns_b; ++i) {
        float_result[i] = 0.0f;
    }

    int* d_a_index;
    int* d_a_positions;
    float* d_a_values;
    int* d_b_index;
    int* d_b_positions;
    float* d_b_values;
    float* d_result;

    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
    cudaMalloc(&d_b_index, num_elements_b_int * sizeof(int));  // Use the same size as 'a' since 'b' is not used in this version
    cudaMalloc(&d_b_positions, (rows + 1) * sizeof(int));       // Use the same size as 'a' since 'b' is not used in this version
    cudaMalloc(&d_b_values, num_elements_b_int * sizeof(float)); // Use the same size as 'a' since 'b' is not used in this version
    cudaMalloc(&d_result, columns * columns_b * sizeof(float));
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b_index, b_index, num_elements_b_int * sizeof(int), cudaMemcpyHostToDevice));  // Use the same data for 'b' as they are not used
    gpuErrchk(cudaMemcpy(d_b_positions, b_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));  // Use the same data for 'b' as they are not used
    gpuErrchk(cudaMemcpy(d_b_values, b_values, num_elements_b_int * sizeof(float), cudaMemcpyHostToDevice)); // Use the same data for 'b' as they are not used
    gpuErrchk(cudaMemset(d_result, 0, columns * columns_b * sizeof(float)));

    gpuErrchk(cudaPeekAtLastError());
    int threadsPerBlock = 128;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    ReuclideanSparse_gpu_atomic_float_different_blocks<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_index, d_a_positions, d_a_values,
        d_b_index, d_b_positions, d_b_values, rows, columns, columns_b, d_result);

    gpuErrchk( cudaPeekAtLastError() );
    cudaMemcpy(float_result, d_result, columns * columns_b * sizeof(float), cudaMemcpyDeviceToHost);
    gpuErrchk( cudaPeekAtLastError() );

    for (int i = 0; i < columns * columns_b; ++i) {
        result[i] = std::sqrt(float_result[i]);
    }

    cudaFree(d_a_index);
    cudaFree(d_a_positions);
    cudaFree(d_a_values);
    cudaFree(d_b_index);
    cudaFree(d_b_positions);
    cudaFree(d_b_values);
    cudaFree(d_result);

    delete[] a_values;
    delete[] b_values;
    delete[] float_result;
}


__global__ void RmanhattanSparse_gpu_atomic_float_same_block(
    int* a_index, int* a_positions, float* a_double_values,
    int rows, int columns, float* result) {

    int row_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_index < rows) {
        int start_column = a_positions[row_index];
        int end_column = a_positions[row_index + 1];

        for (int col1_index = start_column; col1_index < end_column; ++col1_index) {

            for (int col2_index = col1_index; col2_index < end_column; ++col2_index) {
                int prev_col_index = col1_index - 1;
                int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;

                int next_col_index = col2_index + 1;
                int next_col = (next_col_index < end_column) ? a_index[next_col_index] : columns;

                int col1 = a_index[col1_index];
                int col2 = a_index[col2_index];

                float value1 = a_double_values[col1_index];
                float value2 = a_double_values[col2_index];

                for (int left = prev_col + 1; left < col1; ++left) {
                    atomicAdd(&result[left * columns + col2], fabsf(value2));
                }

                for (int right = col2 + 1; right < next_col; ++right) {
                    atomicAdd(&result[col1 * columns + right], fabsf(value1));
                }

                float diff = fabsf(value1 - value2);
                atomicAdd(&result[col1 * columns + col2], diff);
            }
        }
    }
}

extern "C" void matrix_Manhattan_sparse_distance_same_block(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b) {
    int rows = *num_rows;
    int columns = *num_columns;
    int num_elements_a_int = *num_elements_a;

    float* a_values = new float[num_elements_a_int];
    float* float_result = new float[columns * columns];

    for (int i = 0; i < num_elements_a_int; ++i) {
        a_values[i] = static_cast<float>(a_double_values[i]);
    }

    for (int i = 0; i < columns * columns; ++i) {
        float_result[i] = 0.0f;
    }

    int* d_a_index;
    int* d_a_positions;
    float* d_a_values;
    float* d_result;

    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
    cudaMalloc(&d_result, columns * columns * sizeof(float));

    cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, columns * columns * sizeof(float));

    int threadsPerBlock = 128;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    RmanhattanSparse_gpu_atomic_float_same_block<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_index, d_a_positions, d_a_values, rows, columns, d_result);

    gpuErrchk( cudaPeekAtLastError() );

    dim3 block_size(32, 32);
    dim3 num_blocks((columns + 31) / 32, (columns + 31) / 32);

    FinalizeManhattan<<<num_blocks, block_size>>>(columns, d_result);
    gpuErrchk( cudaPeekAtLastError() );

    cudaMemcpy(float_result, d_result, columns * columns * sizeof(float), cudaMemcpyDeviceToHost);
    gpuErrchk( cudaPeekAtLastError() );

    for (int i = 0; i < columns * columns; ++i) {
        result[i] = float_result[i];
    }

    cudaFree(d_a_index);
    cudaFree(d_a_positions);
    cudaFree(d_a_values);
    cudaFree(d_result);
    gpuErrchk(cudaPeekAtLastError());
    delete[] a_values;
    delete[] float_result;
}

__global__ void RmanhattanSparse_gpu_atomic_float_different_blocks(
    int* a_index, int* a_positions, float* a_double_values,
    int* b_index, int* b_positions, float* b_double_values,
    int rows, int columns, int columns_b, float* result) {

    int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index < rows) {
        int start_column = a_positions[row_index];
        int end_column = a_positions[row_index + 1];

        int start_column_b = b_positions[row_index];
        int end_column_b = b_positions[row_index + 1];

        for (int col1_index = start_column; col1_index <= end_column; ++col1_index) {
            int prev_col_index = col1_index - 1;
            int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
            float value1 = (col1_index < end_column) ? a_double_values[col1_index] : 0.0f;

            int col1 = (col1_index < end_column) ? a_index[col1_index] : columns;
            for (int col2_index = start_column_b; col2_index <= end_column_b; ++col2_index) {
                int prev_col_b_index = col2_index - 1;
                int prev_col2 = (prev_col_b_index >= start_column_b) ? b_index[prev_col_b_index] : -1;

                int col2 = (col2_index < end_column_b) ? b_index[col2_index] : columns_b;

                float value2 = (col2_index < end_column_b) ? b_double_values[col2_index] : 0.0f;

                if (col2 < columns_b) {
                    for (int left = prev_col + 1; left < col1; ++left) {
                        atomicAdd(&result[col2 * columns + left], fabsf(value2));
                    }
                }

                if (col1 < columns) {
                    for (int left = prev_col2 + 1; left < col2; ++left) {
                        atomicAdd(&result[left * columns + col1], fabsf(value1));
                    }
                }

                if (col1 < columns && col2 < columns_b) {
                    float diff = fabsf(value1 - value2);
                    atomicAdd(&result[col2 * columns + col1], diff);
                }
            }
        }
    }
}

extern "C" void matrix_Manhattan_sparse_distance_different_blocks(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b) {

    int rows = *num_rows;
    int columns = *num_columns;
    int columns_b = *num_columns_b;
    int num_elements_a_int = *num_elements_a;
    int num_elements_b_int = *num_elements_b;

    float* a_values = new float[num_elements_a_int];
    float* b_values = new float[num_elements_b_int];
    float* float_result = new float[columns * columns_b];

    for (int i = 0; i < num_elements_a_int; ++i) {
        a_values[i] = static_cast<float>(a_double_values[i]);
    }

    for (int i = 0; i < num_elements_b_int; ++i) {
        b_values[i] = static_cast<float>(b_double_values[i]);
    }

    for (int i = 0; i < columns * columns_b; ++i) {
        float_result[i] = 0.0f;
    }

    int* d_a_index;
    int* d_a_positions;
    float* d_a_values;
    int* d_b_index;
    int* d_b_positions;
    float* d_b_values;
    float* d_result;

    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
    cudaMalloc(&d_b_index, num_elements_b_int * sizeof(int));
    cudaMalloc(&d_b_positions, (rows + 1) * sizeof(int));
    cudaMalloc(&d_b_values, num_elements_b_int * sizeof(float));
    cudaMalloc(&d_result, columns * columns_b * sizeof(float));
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b_index, b_index, num_elements_b_int * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b_positions, b_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b_values, b_values, num_elements_b_int * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_result, 0, columns * columns_b * sizeof(float)));

    gpuErrchk(cudaPeekAtLastError());
    int threadsPerBlock = 128;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    RmanhattanSparse_gpu_atomic_float_different_blocks<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_index, d_a_positions, d_a_values,
        d_b_index, d_b_positions, d_b_values, rows, columns, columns_b, d_result);

    gpuErrchk( cudaPeekAtLastError() );
    cudaMemcpy(float_result, d_result, columns * columns_b * sizeof(float), cudaMemcpyDeviceToHost);
    gpuErrchk( cudaPeekAtLastError() );

    for (int i = 0; i < columns * columns_b; ++i) {
        result[i] = float_result[i];
    }

    cudaFree(d_a_index);
    cudaFree(d_a_positions);
    cudaFree(d_a_values);
    cudaFree(d_b_index);
    cudaFree(d_b_positions);
    cudaFree(d_b_values);
    cudaFree(d_result);

    delete[] a_values;
    delete[] b_values;
    delete[] float_result;
}


__global__ void RcosineSparseCorr_gpu_atomic_float_different_blocks(
    int* a_index, int* a_positions, float* a_double_values,
    int* b_index, int* b_positions, float* b_double_values,
    float* result, int rows, int columns, int columns_b,
    float* squares_a, float* squares_b) {

    int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index < rows) {
        int start_column = a_positions[row_index];
        int end_column = a_positions[row_index + 1];

        int start_column_b = b_positions[row_index];
        int end_column_b = b_positions[row_index + 1];

        for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
            float value1 = a_double_values[col1_index];
            int col1 = a_index[col1_index];

            for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
                float value2 = b_double_values[col2_index];
                int col2 = b_index[col2_index];

                atomicAdd(&result[col2 * columns + col1], value1 * value2);
            }
        }

        for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
            float value1 = a_double_values[col1_index];
            int col1 = a_index[col1_index];

            atomicAdd(&squares_a[col1], value1 * value1);
        }

        for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
            
            float value2 = b_double_values[col2_index];
            int col2 = b_index[col2_index];

            // printf(
            //   "ROW: %d %d %d %d %d %f\n",
            //   row_index,
            //   start_column_b,
            //   col2_index,
            //   end_column_b,
            //   col2,
            //   value2
            // );

            atomicAdd(&squares_b[col2], value2 * value2);
        }
    }
}

extern "C" void matrix_Cosine_sparse_distance_different_blocks(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b) {
    
    int rows = *num_rows;
    int columns = *num_columns;
    int columns_b = *num_columns_b;
    int num_elements_a_int = *num_elements_a;
    int num_elements_b_int = *num_elements_b;

    float* a_values = new float[num_elements_a_int];
    float* b_values = new float[num_elements_b_int];
    float* float_result = new float[columns * columns_b];
    float* squares_a = new float[columns];
    float* squares_b = new float[columns_b];

    for (int i = 0; i < num_elements_a_int; ++i) {
        a_values[i] = static_cast<float>(a_double_values[i]);
    }

    for (int i = 0; i < num_elements_b_int; ++i) {
        b_values[i] = static_cast<float>(b_double_values[i]);
    }

    for (int i = 0; i < columns * columns_b; ++i) {
        float_result[i] = 0.0f;
    }

    for (int i = 0; i < columns; ++i) {
        squares_a[i] = 0.0f;
    }

    for (int i = 0; i < columns_b; ++i) {
        squares_b[i] = 0.0f;
    }

    float* d_a_values;
    float* d_b_values;
    float* d_float_result;
    float* d_squares_a;
    float* d_squares_b;
    int* d_a_index;
    int* d_b_index;
    int* d_a_positions;
    int* d_b_positions;

    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
    cudaMalloc(&d_b_values, num_elements_b_int * sizeof(float));
    cudaMalloc(&d_float_result, columns * columns_b * sizeof(float));
    cudaMalloc(&d_squares_a, columns * sizeof(float));
    cudaMalloc(&d_squares_b, columns_b * sizeof(float));
    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
    cudaMalloc(&d_b_positions, (rows + 1) * sizeof(int));
    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
    cudaMalloc(&d_b_index, num_elements_b_int * sizeof(int));

    cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_values, b_values, num_elements_b_int * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_float_result, float_result, columns * columns_b * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_squares_a, squares_a, columns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_squares_b, squares_b, columns_b * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_index, b_index, num_elements_b_int * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_positions, b_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    RcosineSparseCorr_gpu_atomic_float_different_blocks<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_index, d_a_positions, d_a_values,
        d_b_index, d_b_positions, d_b_values,
        d_float_result, rows, columns, columns_b, d_squares_a, d_squares_b);
    
    gpuErrchk( cudaPeekAtLastError() );

    cudaMemcpy(float_result, d_float_result, columns * columns_b * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(squares_a, d_squares_a, columns * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(squares_b, d_squares_b, columns_b * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < columns * columns_b; ++i) {
        int row_index = i / columns;
        int column_index = i % columns;
        result[i] = 1.0f - float_result[i] / sqrtf(squares_b[row_index]) / sqrtf(squares_a[column_index]);
    }

    cudaFree(d_a_values);
    cudaFree(d_b_values);
    cudaFree(d_float_result);
    cudaFree(d_squares_a);
    cudaFree(d_squares_b);
    cudaFree(d_a_index);
    cudaFree(d_b_index);
    cudaFree(d_a_positions);
    cudaFree(d_b_positions);

    delete[] a_values;
    delete[] b_values;
    delete[] float_result;
    delete[] squares_a;
    delete[] squares_b;
}


__global__ void RcosineSparseCorr_gpu_atomic_float_same_block(
    int* a_index, int* a_positions, float* a_double_values,
    float* result, int rows, int columns, float* squares) {

    int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("%d %d\n", row_index, rows);
    if (row_index < rows) {
        // printf("Inside kernel\n");
        int start_column = a_positions[row_index];
        int end_column = a_positions[row_index + 1];
        // printf("Test: %d %d\n", start_column, end_column);
        for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
            // printf("Here\n");
            // int prev_col_index = col1_index - 1;
            // int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
            int col1 = a_index[col1_index];
            float value1 = a_double_values[col1_index];
            atomicAdd(&squares[col1], value1 * value1);
            
            for (int col2_index = col1_index + 1; col2_index < end_column; ++col2_index) {
                //int next_col_index = col2_index + 1;
                //int next_col = (next_col_index < end_column) ? a_index[next_col_index] : columns;
                int col2 = a_index[col2_index];
                float value2 = a_double_values[col2_index];
                
                atomicAdd(&result[col1 * columns + col2], value1 * value2);
            }
        }
    }
}

extern "C" void matrix_Cosine_sparse_distance_same_block(
    int* a_index,
    int* a_positions,
    double* a_double_values,
    int* b_index,
    int* b_positions,
    double* b_double_values,
    double* result,
    int* num_rows,
    int* num_columns,
    int* num_columns_b,
    int* num_elements_a,
    int* num_elements_b
) {

    int rows = *num_rows;
    int columns = *num_columns;
    int num_elements_a_int = *num_elements_a;

    float* a_values = new float[num_elements_a_int];
    float* float_result = new float[columns * columns];
    float* squares = new float[columns];

    for (int i = 0; i < num_elements_a_int; ++i) {
        a_values[i] = static_cast<float>(a_double_values[i]);
    }

    for (int i = 0; i < columns * columns; ++i) {
        float_result[i] = 0.0f;
    }

    for (int i = 0; i < columns; ++i) {
        squares[i] = 0.0f;
    }

    float* d_a_values;
    float* d_float_result;
    float* d_squares;
    int* d_a_index;
    int* d_a_positions;

    // std::cout << rows << std::endl;
    // for (int i = 0; i < rows; ++i) {
    //   std::cout << a_positions[i] << " ";
    // }
    // std::cout << std::endl;

    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
    cudaMalloc(&d_float_result, columns * columns * sizeof(float));
    cudaMalloc(&d_squares, columns * sizeof(float));
    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));

    cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_float_result, float_result, columns * columns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_squares, squares, columns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    

    RcosineSparseCorr_gpu_atomic_float_same_block<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_index, d_a_positions, d_a_values,
        //b_index, b_positions, d_a_values,  // Use the same values for 'b' as they are not used in this version
        d_float_result, rows, columns, d_squares);

    gpuErrchk( cudaPeekAtLastError() );
    dim3 block_size(32, 32);
    dim3 num_blocks((columns + 31) / 32, (columns + 31) / 32);

    FinalizeCosineSparse<<<num_blocks, block_size>>>(columns, d_float_result, d_squares);

    gpuErrchk( cudaPeekAtLastError() );
    cudaMemcpy(float_result, d_float_result, columns * columns * sizeof(float), cudaMemcpyDeviceToHost);

    gpuErrchk( cudaPeekAtLastError() );



    //cudaMemcpy(squares, d_squares, columns * sizeof(float), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < columns * columns; ++i) {
    //    int row_index = i / columns;
    //    int column_index = i % columns;
    //    if (row_index < column_index) {
    //        result[i] = 1.0f - float_result[i] / sqrtf(squares[row_index]) / sqrtf(squares[column_index]);
    //    } else if (row_index > column_index) {
    //        result[i] = result[column_index * columns + row_index];
    //    } else {
    //        result[i] = 1.0;
    //    }
    //}

    cudaFree(d_a_values);
    cudaFree(d_float_result);
    cudaFree(d_squares);
    cudaFree(d_a_index);
    cudaFree(d_a_positions);

    delete[] a_values;
    delete[] float_result;
    delete[] squares;
}

__global__ void RkendallSparseCorr_gpu_atomic_float_same_block(
    int* a_index, int* a_positions, float* a_values,
    int* concordant, int rows, int columns) 
{
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int row_jndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_index >= row_jndex || row_jndex >= rows) {
    return;
  }
  //printf("%d\n", columns);
  //if (row_index % 200 == 0 && row_jndex % 5000 == 0) { 
  //    printf("%d %d\n", row_index, row_jndex);
  //}
  int start_column = a_positions[row_index];
  int end_column = a_positions[row_index + 1];

  int start_column_b = a_positions[row_jndex];
  int end_column_b = a_positions[row_jndex + 1];

  bool left_thresholds[5000];
  //printf("%d\n", end_column_b - start_column_b);
  for (int i = 0; i < end_column_b - start_column_b; ++i) {
    left_thresholds[i] = false;
  }
  bool left_threshold_selected = false;
  bool right_threshold_selected = false;
  int right_down1_threshold = start_column_b;
  int left_down1_threshold = start_column_b;
  int left_down2_threshold = start_column_b;
  int right_down2_threshold = start_column_b;
  bool left_activated = false;
  bool right_activated = false;
  for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
    int prev_col_index = col1_index - 1;
    int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
    int col1 = a_index[col1_index];
    float value1 = a_values[col1_index];
    

    while (right_down1_threshold < end_column_b && a_index[right_down1_threshold] < col1) {
      right_down1_threshold += 1;
    }

    if (right_down1_threshold < end_column_b && a_index[right_down1_threshold] == col1) {
      left_activated = true;
    }
    if (right_down1_threshold < end_column_b && a_index[right_down1_threshold] == col1) {
      left_down2_threshold = right_down1_threshold + 1;
      right_activated = true;
    } else {
      left_down2_threshold = right_down1_threshold;
    }

    right_down2_threshold = left_down2_threshold;
    for (int col2_index = col1_index; col2_index < end_column; ++col2_index) {
      int col2 = a_index[col2_index];
      float value2 = a_values[col2_index];
      int next_col_index = col2_index + 1;
      int next_col = (next_col_index < end_column) ? a_index[next_col_index] : columns;

      while (right_down2_threshold < end_column_b && a_index[right_down2_threshold] < next_col) {
          right_down2_threshold += 1;
      }
      if (left_down1_threshold < end_column_b && !left_thresholds[left_down1_threshold - start_column_b]) {
        left_thresholds[left_down1_threshold - start_column_b] = true;
        for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
            for (int right = left + 1; right < right_down1_threshold; ++right) {
              float product = a_values[left] * a_values[right];
              if (product < 0) {
                atomicAdd(concordant + a_index[left] * columns + a_index[right], 1);
                // atomicAdd(concordant + a_index[right] * columns + a_index[left], 1);
                // disconcordant[a_index[left] * columns + a_index[right]] += 1;
                // disconcordant[a_index[right] * columns + a_index[left]] += 1;
              } 
              //else {
                //atomicAdd(concordant + a_index[left] * columns + a_index[right], 1);
                //atomicAdd(concordant + a_index[right] * columns + a_index[left], 1);
                // concordant[a_index[left] * columns + a_index[right]] += 1;
                // concordant[a_index[right] * columns + a_index[left]] += 1;
              //}
            }
        }
      }
      
      if (left_down2_threshold < end_column_b && !left_thresholds[left_down2_threshold - start_column_b]) {
        left_thresholds[left_down2_threshold - start_column_b] = true;
        for (int left = left_down2_threshold; left < right_down2_threshold; left++) {
            for (int right = left + 1; right < right_down2_threshold; ++right) {
              float product = a_values[left] * a_values[right];
              if (product < 0) {
                atomicAdd(concordant + a_index[left] * columns + a_index[right], 1);
                // atomicAdd(concordant + a_index[right] * columns + a_index[left], 1);
                // disconcordant[a_index[left] * columns + a_index[right]] += 1;
                // disconcordant[a_index[right] * columns + a_index[left]] += 1;
              } 
              //else {
              //  atomicAdd(concordant + a_index[left] * columns + a_index[right], 1);
              //  atomicAdd(concordant + a_index[right] * columns + a_index[left], 1);
                // concordant[a_index[left] * columns + a_index[right]] += 1;
                // concordant[a_index[right] * columns + a_index[left]] += 1;
              //}
            }
        }
      }
      
      for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
            for (int right = left_down2_threshold; right < right_down2_threshold; ++right) {
              float product = a_values[left] * a_values[right];
              if (product < 0) {
                atomicAdd(concordant + a_index[left] * columns + a_index[right], 1);
                // atomicAdd(concordant + a_index[right] * columns + a_index[left], 1);
                // disconcordant[a_index[left] * columns + a_index[right]] += 1;
                // disconcordant[a_index[right] * columns + a_index[left]] += 1;
              } 
              //else {
              //  atomicAdd(concordant + a_index[left] * columns + a_index[right], 1);
              //  atomicAdd(concordant + a_index[right] * columns + a_index[left], 1);
                // concordant[a_index[left] * columns + a_index[right]] += 1;
                // concordant[a_index[right] * columns + a_index[left]] += 1;
              //}
            }
        }
      
      float left_value = (left_activated) ? a_values[right_down1_threshold] : 0;
      float right_value = (right_activated) ? a_values[left_down2_threshold - 1] : 0;
      
      float left_diff = left_value - a_values[col1_index];
      float right_diff = right_value - a_values[col2_index];
      float product = left_diff * right_diff;
      //if (product > 0) {
      //  atomicAdd(concordant + col1 * columns + col2, 1);
      //  atomicAdd(concordant + col2 * columns + col1, 1);
      //} else 
      if (product < 0) {
        atomicAdd(concordant + col1 * columns + col2, 1);
        // atomicAdd(concordant + col2 * columns + col1, 1);
      }
      
      for (int right = left_down2_threshold; right < right_down2_threshold; ++right) {
        product = left_diff * a_values[right];
        if (product < 0) {
          atomicAdd(concordant + col1 * columns + a_index[right], 1);
          //atomicAdd(concordant + a_index[right] * columns + col1, 1);
        } 
        //else if (product > 0) {
        //  atomicAdd(concordant + col1 * columns + a_index[right], 1);
        //  atomicAdd(concordant + a_index[right] * columns + col1, 1);
        //}
      }
          
      for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
        // std::cout << a_index[left] << " " << a_index[col2_index] << std::endl;
        product = right_diff * a_values[left];
        if (product < 0) {
          atomicAdd(concordant + a_index[left] * columns + col2, 1);
          //atomicAdd(concordant + col2 * columns + a_index[left], 1);
        }
        //else if (product > 0) {
        //  atomicAdd(concordant + a_index[left]* columns + col2, 1);
        //  atomicAdd(concordant + col2 * columns + a_index[left], 1);
        //}
      }

      right_activated = false;
      while (left_down2_threshold < end_column_b && a_index[left_down2_threshold] <= next_col) {
          if (a_index[left_down2_threshold] == next_col) {
            right_activated = true;
          }
          left_down2_threshold += 1;
      }

    }
    
    while (left_down1_threshold < end_column_b && a_index[left_down1_threshold] <= col1) {
      left_down1_threshold += 1;
    }
  }
}

extern "C" void matrix_Kendall_sparse_distance_same_block(
    int* a_index,
    int* a_positions,
    double* a_double_values,
    int* b_index,
    int* b_positions,
    double* b_double_values,
    double* result,
    int* num_rows,
    int* num_columns,
    int* num_columns_b,
    int* num_elements_a,
    int* num_elements_b
) {

    int rows = *num_rows;
    int columns = *num_columns;
    int num_elements_a_int = *num_elements_a;

    float* a_values = new float[num_elements_a_int];
    int* h_concordant = new int[columns * columns];

    for (int i = 0; i < num_elements_a_int; ++i) {
        a_values[i] = static_cast<float>(a_double_values[i]);
    }

    for (int i = 0; i < columns * columns; ++i) {
        h_concordant[i] = 0;
    }

    float* d_a_values;
    int* d_concordant;
    int* d_a_index;
    int* d_a_positions;

    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
    cudaMalloc(&d_concordant, columns * columns * sizeof(int));
    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));

    cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(
      d_concordant,
      h_concordant,
      columns * columns * sizeof(int),
      cudaMemcpyHostToDevice
    );
    cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 32;
    int blocks_in_row = (rows + threads - 1) / threads;
    int blocks_in_col = (rows + threads - 1) / threads;

    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks_in_row, blocks_in_col);

    RkendallSparseCorr_gpu_atomic_float_same_block<<<BLOCKS, THREADS>>>(
        d_a_index, d_a_positions, d_a_values,
        // b_index, b_positions, d_b_values,  // Use the same values for 'b' as they are not used in this version
        d_concordant, rows, columns
      );

    cudaMemcpy(h_concordant, d_concordant, columns * columns * sizeof(int), cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    for (int i = 0; i < columns * columns; ++i) {
      int row_index = i / columns;
      int column_index = i % columns;
      if (row_index < column_index) {
        result[i] = static_cast<double>(h_concordant[i]) * 2.0 / rows / (rows - 1);
      } else if (row_index > column_index) {
        result[i] = result[column_index * columns + row_index];
      } else {
        result[i] = 0.0;
      }
    }

    cudaFree(d_a_index);
    cudaFree(d_a_positions);
    cudaFree(d_a_values);
    cudaFree(d_concordant);
    delete[] a_values;
    delete[] h_concordant;
}

__global__ void RkendallSparseCorr_gpu_atomic_float_different_blocks(
    int* a_index, int* a_positions, float* a_values,
    int* b_index, int* b_positions, float* b_values,
    int* concordant, int rows, int columns, int columns_b) 
{
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int row_jndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_index >= row_jndex || row_jndex >= rows) {
    return;
  }
  //if (row_index % 200 == 0 && row_jndex % 2000 == 0) { 
  //    printf("%d %d\n", row_index, row_jndex);
  //}

  int start_column = a_positions[row_index];
  int end_column = a_positions[row_index + 1];
  int start_column_down = a_positions[row_jndex];
  int end_column_down = a_positions[row_jndex + 1];

  int start_column_b = b_positions[row_index];
  int end_column_b = b_positions[row_index + 1];
  int start_column_down_b = b_positions[row_jndex];
  int end_column_down_b = b_positions[row_jndex + 1];
  bool left_threshold_selected = false;
  bool right_threshold_selected = false;
  int right_down1_threshold = start_column_down;
  int left_down1_threshold = start_column_down;
  bool left_activated = false;
  bool right_activated = false;
  // std::cout << "BEFORE THRESHOLD " << left_down1_threshold << " " << right_down1_threshold << " " << left_down2_threshold << " " << right_down2_threshold << std::endl;

  for (int col1_index = start_column; col1_index <= end_column; ++col1_index) {
    int prev_col_index = col1_index - 1;
    int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
    int col1 = (col1_index < end_column) ? a_index[col1_index]: columns;
    float value1 = (col1_index < end_column) ? a_values[col1_index]: 0;
    
    while (right_down1_threshold < end_column_down && a_index[right_down1_threshold] < col1) {
      right_down1_threshold += 1;
    }

    if (right_down1_threshold < end_column_down && a_index[right_down1_threshold] == col1) {
      left_activated = true;
    }

    int left_down2_threshold = start_column_down_b;
    int right_down2_threshold = start_column_down_b;
    right_activated = false;
    for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
      int col2 = b_index[col2_index];
      float value2 = b_values[col2_index];
      int next_col_index = col2_index + 1;
      int next_col = (next_col_index < end_column_b) ? b_index[next_col_index] : columns_b;

      while (right_down2_threshold < end_column_down_b && b_index[right_down2_threshold] < next_col) {
          right_down2_threshold += 1;
      }
      
      for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
            for (int right = left_down2_threshold; right < right_down2_threshold; ++right) {
              float product = a_values[left] * b_values[right];
              if (product < 0) {
                atomicAdd(
                  concordant + b_index[right] * columns + a_index[left], 1
                );
              }
              //else {
              //  atomicAdd(
              //    concordant + b_index[right] * columns + a_index[left], 1
              //  );
              //}
            }
        }
      // std::cout << "COL" << col1 << " " << col2 << std::endl;
      // std::cout << "THRESHOLD " << left_down1_threshold << " " << right_down1_threshold << " " << left_down2_threshold << " " << right_down2_threshold << std::endl;
      
      float left_value = (left_activated) ? a_values[right_down1_threshold] : 0;
      float right_value = (right_activated) ? b_values[left_down2_threshold - 1] : 0;
      
      
      float right_diff = right_value - value2;
      
      float left_diff = left_value - value1;
      float product = left_diff * right_diff;
      if (product < 0) {
        atomicAdd(concordant + col2 * columns + col1, 1);
      }
      
      for (int right = left_down2_threshold; right < right_down2_threshold; ++right) {
        product = left_diff * b_values[right];
        if (product < 0) {
          atomicAdd(concordant + b_index[right] * columns + col1, 1);
        } 
        //else if (product > 0) {
        //  atomicAdd(concordant + b_index[right] * columns + col1, 1);
        //}
      }
          
      for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
        product = right_diff * a_values[left];
        if (product < 0) {
          atomicAdd(concordant + col2 * columns + a_index[left], 1);
        } 
        //else if (product > 0) {
        //  atomicAdd(concordant + col2 * columns + a_index[left], 1);
        //}
      }

      right_activated = false;
      while (left_down2_threshold < end_column_down_b && b_index[left_down2_threshold] <= next_col) {
          if (b_index[left_down2_threshold] == next_col) {
            right_activated = true;
          }
          left_down2_threshold += 1;
      }

    }
    
    while (left_down1_threshold < end_column_down && a_index[left_down1_threshold] <= col1) {
      left_down1_threshold += 1;
    }
  }
}

extern "C" void matrix_Kendall_sparse_distance_different_blocks(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b) {

    int rows = *num_rows;
    int columns = *num_columns;
    int columns_b = *num_columns_b;
    int num_elements_a_int = *num_elements_a;
    int num_elements_b_int = *num_elements_b;

    float* a_values = new float[num_elements_a_int];
    float* b_values = new float[num_elements_b_int];
    float* float_result = new float[columns * columns_b];

    for (int i = 0; i < num_elements_a_int; ++i) {
        a_values[i] = static_cast<float>(a_double_values[i]);
    }

    for (int i = 0; i < num_elements_b_int; ++i) {
        b_values[i] = static_cast<float>(b_double_values[i]);
    }

    int* h_concordant = new int[columns * columns_b];
    for (int i = 0; i < columns * columns_b; ++i) {
        h_concordant[i] = 0;
    }

    float* d_a_values;
    float* d_b_values;
    int* d_concordant;
    int* d_a_index;
    int* d_b_index;
    int* d_a_positions;
    int* d_b_positions;

    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
    cudaMalloc(&d_b_values, num_elements_b_int * sizeof(float));
    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
    cudaMalloc(&d_b_positions, (rows + 1) * sizeof(int));
    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
    cudaMalloc(&d_b_index, num_elements_b_int * sizeof(int));
    cudaMalloc(&d_concordant, columns * columns_b * sizeof(int));

    cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_values, b_values, num_elements_b_int * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(
      d_concordant,
      h_concordant,
      columns * columns_b * sizeof(int),
      cudaMemcpyHostToDevice
    );
    cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_index, b_index, num_elements_b_int * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_positions, b_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 32;
    int blocks_in_row = (rows + threads - 1) / threads;
    int blocks_in_col = (rows + threads - 1) / threads;

    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks_in_row, blocks_in_col);

    RkendallSparseCorr_gpu_atomic_float_different_blocks<<<BLOCKS, THREADS>>>(
        d_a_index, d_a_positions, d_a_values,
        d_b_index, d_b_positions, d_b_values,  // Use the same values for 'b' as they are not used in this version
        d_concordant, rows, columns, columns_b
      );

    cudaMemcpy(h_concordant, d_concordant, columns * columns_b * sizeof(int), cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    for (int i = 0; i < columns * columns_b; ++i) {
        result[i] = static_cast<double>(h_concordant[i]) * 2.0 / rows / (rows - 1);
    }

    cudaFree(d_a_values);
    cudaFree(d_concordant);
    cudaFree(d_a_index);
    cudaFree(d_a_positions);
    cudaFree(d_b_index);
    cudaFree(d_b_positions);
    cudaFree(d_b_values);

    delete[] a_values;
    delete[] b_values;
    delete[] h_concordant;
}

// ==================== Spearman ====================

// Rank columns and center (subtract mean rank = (n+1)/2) so that
// the existing Cosine kernels (which compute cosine-style correlation)
// produce the correct Pearson-on-ranks = Spearman result.
static void rank_columns(float* array, int n, int m) {
    std::vector<std::pair<float, int>> col_data(n);
    float mean_rank = (n + 1) / 2.0f;
    for (int j = 0; j < m; ++j) {
        float* col = array + j * n;
        for (int i = 0; i < n; ++i) col_data[i] = {col[i], i};
        std::sort(col_data.begin(), col_data.end());
        int i = 0;
        while (i < n) {
            int end = i + 1;
            while (end < n && col_data[end].first == col_data[i].first) ++end;
            float rank = (i + 1 + end) / 2.0f - mean_rank;
            for (int k = i; k < end; ++k) col[col_data[k].second] = rank;
            i = end;
        }
    }
}

// Sweep-line ranking directly from CSR: sorts only non-zeros per column,
// inserts the implicit-zero group at the right position, assigns centered
// average-tie ranks (rank - mean_rank) so Cosine kernels give Spearman.
// Returns a dense column-major ranked array (rows x columns).
static float* csr_to_ranked_dense(int* index, int* positions, double* values, int rows, int columns) {
    float* ranked = new float[rows * columns];
    float mean_rank = (rows + 1) / 2.0f;

    // 1. Gather per-column non-zero entries (value, row_index)
    std::vector<std::vector<std::pair<float, int>>> col_entries(columns);
    for (int row = 0; row < rows; ++row) {
        for (int idx = positions[row]; idx < positions[row + 1]; ++idx) {
            col_entries[index[idx]].push_back({static_cast<float>(values[idx]), row});
        }
    }

    for (int j = 0; j < columns; ++j) {
        float* col_out = ranked + j * rows;
        auto& entries = col_entries[j];
        int nnz = static_cast<int>(entries.size());

        // 2. Sort non-zeros by value
        std::sort(entries.begin(), entries.end());

        // 3. Count negatives and explicit zeros among stored entries
        int neg_count = 0;
        int explicit_zero_count = 0;
        for (int i = 0; i < nnz; ++i) {
            if (entries[i].first < 0.0f) neg_count++;
            else if (entries[i].first == 0.0f) explicit_zero_count++;
        }
        int total_zeros = (rows - nnz) + explicit_zero_count;

        // 4. Zero-group rank (covers implicit + explicit zeros), centered
        float zero_rank = 0.0f;
        if (total_zeros > 0) {
            zero_rank = (neg_count + 1 + neg_count + total_zeros) / 2.0f - mean_rank;
        }
        // Pre-fill all rows with zero_rank (implicit zeros get it directly)
        for (int r = 0; r < rows; ++r) col_out[r] = zero_rank;

        // 5. Rank negative entries (sorted positions 0..neg_count-1)
        int global_pos = 0;
        int i = 0;
        while (i < neg_count) {
            int end = i + 1;
            while (end < neg_count && entries[end].first == entries[i].first) ++end;
            float rank = (global_pos + 1 + global_pos + (end - i)) / 2.0f - mean_rank;
            for (int k = i; k < end; ++k) col_out[entries[k].second] = rank;
            global_pos += (end - i);
            i = end;
        }

        // Explicit zeros already got zero_rank from pre-fill

        // 6. Rank positive entries (sorted positions neg_count+explicit_zero_count .. nnz-1)
        global_pos = neg_count + total_zeros;
        i = neg_count + explicit_zero_count;
        while (i < nnz) {
            int end = i + 1;
            while (end < nnz && entries[end].first == entries[i].first) ++end;
            float rank = (global_pos + 1 + global_pos + (end - i)) / 2.0f - mean_rank;
            for (int k = i; k < end; ++k) col_out[entries[k].second] = rank;
            global_pos += (end - i);
            i = end;
        }
    }
    return ranked;
}

extern "C" void matrix_Spearman_distance_same_block(double* a, double* b, double* c, int* n, int* m, int* m_b) {
    int array_size = *n * *m;
    float* array_new = new float[array_size];
    for (int i = 0; i < array_size; ++i) {
        array_new[i] = static_cast<float>(a[i]);
    }
    rank_columns(array_new, *n, *m);

    float* d_array;
    cudaMalloc(&d_array, array_size * sizeof(float));
    cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (*n + threads - 1) / threads;

    float* d_result;
    float* h_result = new float[(*m) * (*m)];
    cudaMalloc(&d_result, (*m) * (*m) * sizeof(float));
    cudaMemset(d_result, 0, (*m) * (*m) * sizeof(float));

    float* d_x_norm_result;
    cudaMalloc(&d_x_norm_result, (*m) * (*m) * sizeof(float));
    cudaMemset(d_x_norm_result, 0, (*m) * (*m) * sizeof(float));

    float* d_y_norm_result;
    cudaMalloc(&d_y_norm_result, (*m) * (*m) * sizeof(float));
    cudaMemset(d_y_norm_result, 0, (*m) * (*m) * sizeof(float));

    RcosineCorr_gpu_atomic_float_same_block<<<blocks, threads>>>(
        d_array, *n, *m, d_result, d_x_norm_result, d_y_norm_result
    );

    int columns = *m;
    dim3 block_size(32, 32);
    dim3 num_blocks((columns + 31) / 32, (columns + 31) / 32);
    FinalizeCosine<<<num_blocks, block_size>>>(columns, d_result, d_x_norm_result, d_y_norm_result);

    cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);

    // FinalizeCosine computes upper triangle (row < col) in row-major.
    // Mirror on host to fill the lower triangle reliably.
    for (int row = 0; row < columns; ++row) {
        for (int col = 0; col < columns; ++col) {
            if (row < col) {
                c[row * columns + col] = h_result[row * columns + col];
            } else if (row > col) {
                c[row * columns + col] = h_result[col * columns + row];
            } else {
                c[row * columns + col] = 0.0;
            }
        }
    }

    free(array_new);
    free(h_result);
    cudaFree(d_result);
    cudaFree(d_x_norm_result);
    cudaFree(d_y_norm_result);
    cudaFree(d_array);
}

extern "C" void matrix_Spearman_distance_different_blocks(double* a, double* b, double* c, int* n, int* m, int* m_b) {
    int array_size = *n * *m;
    float* array_new = new float[array_size];
    for (int i = 0; i < array_size; ++i) {
        array_new[i] = static_cast<float>(a[i]);
    }
    rank_columns(array_new, *n, *m);

    int array2_size = *n * (*m_b);
    float* array2_new = new float[array2_size];
    for (int i = 0; i < array2_size; ++i) {
        array2_new[i] = static_cast<float>(b[i]);
    }
    rank_columns(array2_new, *n, *m_b);

    float* d_array;
    float* d_array2;
    cudaMalloc(&d_array, array_size * sizeof(float));
    cudaMalloc(&d_array2, array2_size * sizeof(float));
    cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, array2_new, array2_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks_in_row = (*n + threads - 1) / threads;

    float* scalar;
    float* h_scalar = new float[(*m) * (*m_b)];
    float* prod1;
    float* h_prod1 = new float[(*m) * (*m_b)];
    float* prod2;
    float* h_prod2 = new float[(*m) * (*m_b)];

    cudaMalloc(&scalar, (*m) * (*m_b) * sizeof(float));
    cudaMemset(scalar, 0, (*m) * (*m_b) * sizeof(float));
    cudaMalloc(&prod1, (*m) * (*m_b) * sizeof(float));
    cudaMemset(prod1, 0, (*m) * (*m_b) * sizeof(float));
    cudaMalloc(&prod2, (*m) * (*m_b) * sizeof(float));
    cudaMemset(prod2, 0, (*m) * (*m_b) * sizeof(float));

    RcosineCorr_gpu_atomic_float_different_blocks<<<blocks_in_row, threads>>>(
        d_array, d_array2, *n, *m, *m_b, scalar, prod1, prod2
    );

    cudaMemcpy(h_scalar, scalar, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prod1, prod1, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prod2, prod2, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);

    // GPU kernel writes row-major [col1*m_b + col2], R needs column-major [col2*m + col1]
    for (int col1 = 0; col1 < *m; ++col1) {
        for (int col2 = 0; col2 < *m_b; ++col2) {
            int gpu_idx = col1 * (*m_b) + col2;
            int r_idx = col2 * (*m) + col1;
            c[r_idx] = 1.0 - h_scalar[gpu_idx] / sqrtf(h_prod1[gpu_idx]) / sqrtf(h_prod2[gpu_idx]);
        }
    }

    delete[] h_scalar;
    delete[] h_prod2;
    delete[] h_prod1;
    delete[] array_new;
    delete[] array2_new;
    cudaFree(scalar);
    cudaFree(prod2);
    cudaFree(prod1);
    cudaFree(d_array);
    cudaFree(d_array2);
}

extern "C" void matrix_Spearman_sparse_distance_same_block(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
    int rows = *num_rows;
    int columns = *num_columns;

    float* dense = csr_to_ranked_dense(a_index, a_positions, a_double_values, rows, columns);

    float* d_array;
    cudaMalloc(&d_array, rows * columns * sizeof(float));
    cudaMemcpy(d_array, dense, rows * columns * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (rows + threads - 1) / threads;

    float* d_result;
    float* h_result = new float[columns * columns];
    cudaMalloc(&d_result, columns * columns * sizeof(float));
    cudaMemset(d_result, 0, columns * columns * sizeof(float));

    float* d_x_norm;
    cudaMalloc(&d_x_norm, columns * columns * sizeof(float));
    cudaMemset(d_x_norm, 0, columns * columns * sizeof(float));

    float* d_y_norm;
    cudaMalloc(&d_y_norm, columns * columns * sizeof(float));
    cudaMemset(d_y_norm, 0, columns * columns * sizeof(float));

    RcosineCorr_gpu_atomic_float_same_block<<<blocks, threads>>>(
        d_array, rows, columns, d_result, d_x_norm, d_y_norm
    );

    dim3 block_size(32, 32);
    dim3 num_blocks((columns + 31) / 32, (columns + 31) / 32);
    FinalizeCosine<<<num_blocks, block_size>>>(columns, d_result, d_x_norm, d_y_norm);

    cudaMemcpy(h_result, d_result, columns * columns * sizeof(float), cudaMemcpyDeviceToHost);

    // Mirror on host: FinalizeCosine computes upper triangle (row < col) row-major
    for (int row = 0; row < columns; ++row) {
        for (int col = 0; col < columns; ++col) {
            if (row < col) {
                result[row * columns + col] = h_result[row * columns + col];
            } else if (row > col) {
                result[row * columns + col] = h_result[col * columns + row];
            } else {
                result[row * columns + col] = 0.0;
            }
        }
    }

    delete[] dense;
    delete[] h_result;
    cudaFree(d_array);
    cudaFree(d_result);
    cudaFree(d_x_norm);
    cudaFree(d_y_norm);
}

extern "C" void matrix_Spearman_sparse_distance_different_blocks(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
    int rows = *num_rows;
    int columns = *num_columns;
    int columns_b = *num_columns_b;

    float* dense_a = csr_to_ranked_dense(a_index, a_positions, a_double_values, rows, columns);
    float* dense_b = csr_to_ranked_dense(b_index, b_positions, b_double_values, rows, columns_b);

    float* d_array;
    float* d_array2;
    cudaMalloc(&d_array, rows * columns * sizeof(float));
    cudaMalloc(&d_array2, rows * columns_b * sizeof(float));
    cudaMemcpy(d_array, dense_a, rows * columns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, dense_b, rows * columns_b * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks_in_row = (rows + threads - 1) / threads;

    float* scalar;
    float* h_scalar = new float[columns * columns_b];
    float* prod1;
    float* h_prod1 = new float[columns * columns_b];
    float* prod2;
    float* h_prod2 = new float[columns * columns_b];

    cudaMalloc(&scalar, columns * columns_b * sizeof(float));
    cudaMemset(scalar, 0, columns * columns_b * sizeof(float));
    cudaMalloc(&prod1, columns * columns_b * sizeof(float));
    cudaMemset(prod1, 0, columns * columns_b * sizeof(float));
    cudaMalloc(&prod2, columns * columns_b * sizeof(float));
    cudaMemset(prod2, 0, columns * columns_b * sizeof(float));

    RcosineCorr_gpu_atomic_float_different_blocks<<<blocks_in_row, threads>>>(
        d_array, d_array2, rows, columns, columns_b, scalar, prod1, prod2
    );

    cudaMemcpy(h_scalar, scalar, columns * columns_b * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prod1, prod1, columns * columns_b * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prod2, prod2, columns * columns_b * sizeof(float), cudaMemcpyDeviceToHost);

    // GPU kernel writes row-major [col1*columns_b + col2], R needs column-major [col2*columns + col1]
    for (int col1 = 0; col1 < columns; ++col1) {
        for (int col2 = 0; col2 < columns_b; ++col2) {
            int gpu_idx = col1 * columns_b + col2;
            int r_idx = col2 * columns + col1;
            result[r_idx] = 1.0 - h_scalar[gpu_idx] / sqrtf(h_prod1[gpu_idx]) / sqrtf(h_prod2[gpu_idx]);
        }
    }

    delete[] dense_a;
    delete[] dense_b;
    delete[] h_scalar;
    delete[] h_prod1;
    delete[] h_prod2;
    cudaFree(scalar);
    cudaFree(prod1);
    cudaFree(prod2);
    cudaFree(d_array);
    cudaFree(d_array2);
}

// ==================== True Pearson (centered cosine) ====================

static void center_columns_host(float* array, int n, int m) {
    for (int col = 0; col < m; ++col) {
        float* col_data = array + col * n;
        float sum = 0.0f;
        for (int row = 0; row < n; ++row) {
            sum += col_data[row];
        }
        float mean = sum / n;
        for (int row = 0; row < n; ++row) {
            col_data[row] -= mean;
        }
    }
}

static float* csr_to_dense_host(int* index, int* positions, double* values, int rows, int columns) {
    float* dense = new float[rows * columns];
    std::memset(dense, 0, rows * columns * sizeof(float));
    for (int row = 0; row < rows; ++row) {
        int start = positions[row];
        int end = positions[row + 1];
        for (int j = start; j < end; ++j) {
            int col = index[j];
            dense[col * rows + row] = static_cast<float>(values[j]);
        }
    }
    return dense;
}

extern "C" void matrix_Pearson_distance_same_block(double* a, double* b, double* c, int* n, int* m, int* m_b) {
    int array_size = *n * *m;
    float* array_new = new float[array_size];
    for (int i = 0; i < array_size; ++i) {
        array_new[i] = static_cast<float>(a[i]);
    }
    center_columns_host(array_new, *n, *m);

    float* d_array;
    cudaMalloc(&d_array, array_size * sizeof(float));
    cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (*n + threads - 1) / threads;

    float* d_result;
    float* h_result = new float[(*m) * (*m)];
    cudaMalloc(&d_result, (*m) * (*m) * sizeof(float));
    cudaMemset(d_result, 0, (*m) * (*m) * sizeof(float));

    float* d_x_norm_result;
    cudaMalloc(&d_x_norm_result, (*m) * (*m) * sizeof(float));
    cudaMemset(d_x_norm_result, 0, (*m) * (*m) * sizeof(float));

    float* d_y_norm_result;
    cudaMalloc(&d_y_norm_result, (*m) * (*m) * sizeof(float));
    cudaMemset(d_y_norm_result, 0, (*m) * (*m) * sizeof(float));

    RcosineCorr_gpu_atomic_float_same_block<<<blocks, threads>>>(
        d_array, *n, *m, d_result, d_x_norm_result, d_y_norm_result
    );

    int columns = *m;
    dim3 block_size(32, 32);
    dim3 num_blocks((columns + 31) / 32, (columns + 31) / 32);
    FinalizeCosine<<<num_blocks, block_size>>>(columns, d_result, d_x_norm_result, d_y_norm_result);

    cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);

    for (int row = 0; row < columns; ++row) {
        for (int col = 0; col < columns; ++col) {
            if (row < col) {
                c[row * columns + col] = h_result[row * columns + col];
            } else if (row > col) {
                c[row * columns + col] = h_result[col * columns + row];
            } else {
                c[row * columns + col] = 0.0;
            }
        }
    }

    delete[] array_new;
    delete[] h_result;
    cudaFree(d_result);
    cudaFree(d_x_norm_result);
    cudaFree(d_y_norm_result);
    cudaFree(d_array);
}

extern "C" void matrix_Pearson_distance_different_blocks(double* a, double* b, double* c, int* n, int* m, int* m_b) {
    int array_size = *n * *m;
    float* array_new = new float[array_size];
    for (int i = 0; i < array_size; ++i) {
        array_new[i] = static_cast<float>(a[i]);
    }
    center_columns_host(array_new, *n, *m);

    int array2_size = *n * (*m_b);
    float* array2_new = new float[array2_size];
    for (int i = 0; i < array2_size; ++i) {
        array2_new[i] = static_cast<float>(b[i]);
    }
    center_columns_host(array2_new, *n, *m_b);

    float* d_array;
    float* d_array2;
    cudaMalloc(&d_array, array_size * sizeof(float));
    cudaMalloc(&d_array2, array2_size * sizeof(float));
    cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, array2_new, array2_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks_in_row = (*n + threads - 1) / threads;

    float* scalar;
    float* h_scalar = new float[(*m) * (*m_b)];
    float* prod1;
    float* h_prod1 = new float[(*m) * (*m_b)];
    float* prod2;
    float* h_prod2 = new float[(*m) * (*m_b)];

    cudaMalloc(&scalar, (*m) * (*m_b) * sizeof(float));
    cudaMemset(scalar, 0, (*m) * (*m_b) * sizeof(float));
    cudaMalloc(&prod1, (*m) * (*m_b) * sizeof(float));
    cudaMemset(prod1, 0, (*m) * (*m_b) * sizeof(float));
    cudaMalloc(&prod2, (*m) * (*m_b) * sizeof(float));
    cudaMemset(prod2, 0, (*m) * (*m_b) * sizeof(float));

    RcosineCorr_gpu_atomic_float_different_blocks<<<blocks_in_row, threads>>>(
        d_array, d_array2, *n, *m, *m_b, scalar, prod1, prod2
    );

    cudaMemcpy(h_scalar, scalar, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prod1, prod1, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prod2, prod2, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);

    for (int col1 = 0; col1 < *m; ++col1) {
        for (int col2 = 0; col2 < *m_b; ++col2) {
            int gpu_idx = col1 * (*m_b) + col2;
            int r_idx = col2 * (*m) + col1;
            c[r_idx] = 1.0 - h_scalar[gpu_idx] / sqrtf(h_prod1[gpu_idx]) / sqrtf(h_prod2[gpu_idx]);
        }
    }

    delete[] h_scalar;
    delete[] h_prod2;
    delete[] h_prod1;
    delete[] array_new;
    delete[] array2_new;
    cudaFree(scalar);
    cudaFree(prod2);
    cudaFree(prod1);
    cudaFree(d_array);
    cudaFree(d_array2);
}

extern "C" void matrix_Pearson_sparse_distance_same_block(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
    int rows = *num_rows;
    int columns = *num_columns;
    int num_elements_a_int = *num_elements_a;

    float* a_values = new float[num_elements_a_int];
    float* float_result = new float[columns * columns];
    float* squares = new float[columns];
    float* col_sums = new float[columns];

    for (int i = 0; i < num_elements_a_int; ++i) {
        a_values[i] = static_cast<float>(a_double_values[i]);
    }
    std::memset(float_result, 0, columns * columns * sizeof(float));
    std::memset(squares, 0, columns * sizeof(float));
    std::memset(col_sums, 0, columns * sizeof(float));

    // Compute column sums on host from CSR (O(nnz))
    for (int row = 0; row < rows; ++row) {
        for (int j = a_positions[row]; j < a_positions[row + 1]; ++j) {
            col_sums[a_index[j]] += a_values[j];
        }
    }

    // Upload CSR data to GPU
    float* d_a_values;
    float* d_float_result;
    float* d_squares;
    int* d_a_index;
    int* d_a_positions;

    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
    cudaMalloc(&d_float_result, columns * columns * sizeof(float));
    cudaMalloc(&d_squares, columns * sizeof(float));
    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));

    cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_float_result, float_result, columns * columns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_squares, squares, columns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Reuse Cosine sparse kernel for dot products and squared norms
    int threadsPerBlock = 128;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    RcosineSparseCorr_gpu_atomic_float_same_block<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_index, d_a_positions, d_a_values,
        d_float_result, rows, columns, d_squares);

    gpuErrchk( cudaPeekAtLastError() );

    gpuErrchk( cudaPeekAtLastError() );

    cudaMemcpy(float_result, d_float_result, columns * columns * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(squares, d_squares, columns * sizeof(float), cudaMemcpyDeviceToHost);

    // Finalize on host: mean correction + symmetry (avoids GPU race condition)
    for (int r = 0; r < columns; ++r) {
        result[r * columns + r] = 0.0;
        for (int c = r + 1; c < columns; ++c) {
            float dot_centered = float_result[r * columns + c]
                - col_sums[r] * col_sums[c] / rows;
            float norm1 = sqrtf(squares[r] - col_sums[r] * col_sums[r] / rows);
            float norm2 = sqrtf(squares[c] - col_sums[c] * col_sums[c] / rows);
            double val = 1.0 - dot_centered / (norm1 * norm2);
            result[r * columns + c] = val;
            result[c * columns + r] = val;
        }
    }

    cudaFree(d_a_values);
    cudaFree(d_float_result);
    cudaFree(d_squares);
    cudaFree(d_a_index);
    cudaFree(d_a_positions);

    delete[] a_values;
    delete[] float_result;
    delete[] squares;
    delete[] col_sums;
}

extern "C" void matrix_Pearson_sparse_distance_different_blocks(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
    int rows = *num_rows;
    int columns = *num_columns;
    int columns_b = *num_columns_b;
    int num_elements_a_int = *num_elements_a;
    int num_elements_b_int = *num_elements_b;

    float* a_values = new float[num_elements_a_int];
    float* b_values = new float[num_elements_b_int];
    float* float_result = new float[columns * columns_b];
    float* squares_a = new float[columns];
    float* squares_b = new float[columns_b];
    float* col_sums_a = new float[columns];
    float* col_sums_b = new float[columns_b];

    for (int i = 0; i < num_elements_a_int; ++i)
        a_values[i] = static_cast<float>(a_double_values[i]);
    for (int i = 0; i < num_elements_b_int; ++i)
        b_values[i] = static_cast<float>(b_double_values[i]);

    std::memset(float_result, 0, columns * columns_b * sizeof(float));
    std::memset(squares_a, 0, columns * sizeof(float));
    std::memset(squares_b, 0, columns_b * sizeof(float));
    std::memset(col_sums_a, 0, columns * sizeof(float));
    std::memset(col_sums_b, 0, columns_b * sizeof(float));

    // Compute column sums on host from CSR (O(nnz))
    for (int row = 0; row < rows; ++row) {
        for (int j = a_positions[row]; j < a_positions[row + 1]; ++j)
            col_sums_a[a_index[j]] += a_values[j];
        for (int j = b_positions[row]; j < b_positions[row + 1]; ++j)
            col_sums_b[b_index[j]] += b_values[j];
    }

    // Upload CSR data to GPU
    float* d_a_values;
    float* d_b_values;
    float* d_float_result;
    float* d_squares_a;
    float* d_squares_b;
    int* d_a_index;
    int* d_b_index;
    int* d_a_positions;
    int* d_b_positions;

    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
    cudaMalloc(&d_b_values, num_elements_b_int * sizeof(float));
    cudaMalloc(&d_float_result, columns * columns_b * sizeof(float));
    cudaMalloc(&d_squares_a, columns * sizeof(float));
    cudaMalloc(&d_squares_b, columns_b * sizeof(float));
    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
    cudaMalloc(&d_b_positions, (rows + 1) * sizeof(int));
    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
    cudaMalloc(&d_b_index, num_elements_b_int * sizeof(int));

    cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_values, b_values, num_elements_b_int * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_float_result, float_result, columns * columns_b * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_squares_a, squares_a, columns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_squares_b, squares_b, columns_b * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_index, b_index, num_elements_b_int * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_positions, b_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Reuse Cosine sparse kernel for dot products and squared norms
    int threadsPerBlock = 128;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    RcosineSparseCorr_gpu_atomic_float_different_blocks<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_index, d_a_positions, d_a_values,
        d_b_index, d_b_positions, d_b_values,
        d_float_result, rows, columns, columns_b, d_squares_a, d_squares_b);

    gpuErrchk( cudaPeekAtLastError() );

    cudaMemcpy(float_result, d_float_result, columns * columns_b * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(squares_a, d_squares_a, columns * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(squares_b, d_squares_b, columns_b * sizeof(float), cudaMemcpyDeviceToHost);

    // Finalize on host with mean correction
    for (int i = 0; i < columns * columns_b; ++i) {
        int col_b = i / columns;
        int col_a = i % columns;
        float dot_centered = float_result[i]
            - col_sums_a[col_a] * col_sums_b[col_b] / rows;
        float norm_a = sqrtf(squares_a[col_a]
            - col_sums_a[col_a] * col_sums_a[col_a] / rows);
        float norm_b = sqrtf(squares_b[col_b]
            - col_sums_b[col_b] * col_sums_b[col_b] / rows);
        result[i] = 1.0f - dot_centered / (norm_a * norm_b);
    }

    cudaFree(d_a_values);
    cudaFree(d_b_values);
    cudaFree(d_float_result);
    cudaFree(d_squares_a);
    cudaFree(d_squares_b);
    cudaFree(d_a_index);
    cudaFree(d_b_index);
    cudaFree(d_a_positions);
    cudaFree(d_b_positions);

    delete[] a_values;
    delete[] b_values;
    delete[] float_result;
    delete[] squares_a;
    delete[] squares_b;
    delete[] col_sums_a;
    delete[] col_sums_b;
}

