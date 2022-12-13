#include <iostream>
#include <fstream>
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

__global__ void Rkendall_gpu_atomic(const double* col1, const double* col2, const int n, const int m, unsigned long long* R){
  int row1 = blockIdx.y * blockDim.y + threadIdx.y;
  int row2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (row1 < row2 && row2 < n){
    if ((col1[row1] - col1[row2]) * (col2[row1] - col2[row2]) < 0){
      atomicAdd(R, 1);
    }
  }
}

__global__ void Rkendall_gpu_atomic_float(float* array, const int n, const int m, unsigned int* result) {
  
  int row1 = blockIdx.y * blockDim.y + threadIdx.y;
  int row2 = blockIdx.x * blockDim.x + threadIdx.x;


  for (int col1_num = 0; col1_num < m; ++col1_num) {
      for (int col2_num = col1_num + 1; col2_num < m; ++col2_num) {
          float* col1 = array + n * col1_num;
          float* col2 = array + n * col2_num;

          if (row1 < row2 && row2 < n){
            if ((col1[row1] - col1[row2]) * (col2[row1] - col2[row2]) < 0){
              atomicAdd(result + col1_num * m + col2_num, 1);
              atomicAdd(result + col2_num * m + col1_num, 1);
            }
          }
      }
  }
}

__global__ void Reuclidean_gpu_atomic_float(float* array, const int n, const int m, unsigned int* result) {
  
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  for (int col1_num = 0; col1_num < m; ++col1_num) {
      for (int col2_num = col1_num + 1; col2_num < m; ++col2_num) {
          float* col1 = array + n * col1_num;
          float* col2 = array + n * col2_num;

          if (row < n) {
            float diff = col1[row] - col2[row];
            diff = diff * diff;
            atomicAdd(result + col1_num * m + col2_num, diff);
            atomicAdd(result + col2_num * m + col1_num, diff);

          }
      }
  }
}

__global__ void Rkendall_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, unsigned int* result) {
  
  int row1 = blockIdx.y * blockDim.y + threadIdx.y;
  int row2 = blockIdx.x * blockDim.x + threadIdx.x;


  for (int col1_num = 0; col1_num < m; ++col1_num) {
      for (int col2_num = 0; col2_num < m_b; ++col2_num) {
          float* col1 = array + n * col1_num;
          float* col2 = array2 + n * col2_num;

          if (row1 < row2 && row2 < n){
            if ((col1[row1] - col1[row2]) * (col2[row1] - col2[row2]) < 0){
              atomicAdd(result + col2_num * m + col1_num, 1);
            }
          }
      }
  }
}

__global__ void Reuclidean_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, unsigned int* result) {
  
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

__global__ void RpearsonCorr_gpu_atomic_float_same_block(
  float* array,
  const int n, const int m,
  float* scalar_product,
  float* x_norm,
  float* y_norm
) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  for (int col1_num = 0; col1_num < m; ++col1_num) {
      for (int col2_num = col1_num; col2_num < m; ++col2_num) {
          float* col1 = array + n * col1_num;
          float* col2 = array + n * col2_num;

          if (row < n) {
            float diff = col1[row] * col2[row];
            atomicAdd(scalar_product + col1_num * m + col2_num, diff);
            atomicAdd(scalar_product + col2_num * m + col1_num, diff);

            float x_element_norm = col1[row] * col1[row];
            float y_element_norm = col2[row] * col2[row];

            atomicAdd(x_norm + col1_num * m + col2_num, x_element_norm);
            atomicAdd(x_norm + col2_num * m + col1_num, x_element_norm);
            atomicAdd(y_norm + col1_num * m + col2_num, y_element_norm);
            atomicAdd(y_norm + col2_num * m + col1_num, y_element_norm);
          }
      }
  }
}
__global__ void RpearsonCorr_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, float* result ){
	int row = blockIdx.y*blockDim.y +threadIdx.y;
	int row2 = blockIdx.x*blockDim.x +threadIdx.x;
	float epsilon=0.01;
	for (int col1_num=0;col1_num<m;++col1_num){
	float num = 0;
	float sum1 = 0;
	float sum2 = 0;
	float dist = 0;
	for(int col2_num=0;col2_num<m_b;++col2_num){
			float* col1 = array + n * col1_num;
			float* col2 = array2 + n * col2_num;
			if(row<n && row<row2) {
				if(col2[row]==0.0 || col1[row]==0.0) {
					atomicAdd(result+col2_num*m+col1_num,1-(num/sqrt(sum1*sum2)));
				} else {
					num = (col1[row2] * col2[row]);

					sum1 = (col1[row2] *col1[row2]);
					sum2 = (col2[row] * col2[row]);
					dist = 1-(num/ sqrt(sum1+sum2));
					if(dist==1){}
					atomicAdd(result+col2_num*m+col1_num,dist+epsilon);
					if(threadIdx.x==0){printf("val1=%4.2f, val2=%4.2f, num=%4.2f, sum1=%4.2f, sum2=%4.2f, res=%4.2f  \n", col1[row],col2[row],num,sum1,sum2,dist);}
				}
			}
		}
	}
} 
__global__ void Rpearson_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, unsigned int* result){
  //int i,j,k;
  //int row1 = blockIdx.y * blockDim.y + threadIdx.y;
	//int row2 = blockIdx.x* blockDim.x + threadIdx.x;
  float epsilon = 0.01;
  int row = blockIdx.x * blockDim.x + threadIdx.x;


  for (int col1_num = 0; col1_num < m; ++col1_num) {
    for (int col2_num = 0; col2_num < m_b; ++col2_num) {
      float* col1 = array + n * col1_num;
      float* col2 = array2 + n * col2_num;
      if (row < m) {
        if(col2[row]==0.0){
          
          float diff = col1[row] - col2[row];
          diff = diff * diff;
          diff = diff/epsilon;
          //atomicAdd(result + col1_num * m + col2_num, diff);
          atomicAdd(result + col2_num * m + col1_num, diff);
        } else {
          float diff = col1[row] - col2[row];
          diff = diff * diff;
          diff = diff/col2[row];
          //atomicAdd(result + col1_num * m + col2_num, diff);
          atomicAdd(result + col2_num * m + col1_num, diff);
        }
      }
    }
  }
}

__global__ void  RpearsonCorr2_gpu_atomic_float_different_blocks(float *gA, float *gB, const int nrow,const int ncol, const int m_b, float*gC) {
  __shared__ float sA[16][16];
  __shared__ float sB[16][16];
  int i,j,k; float epsilon=0.001;
  int offset;  float a,b;
  float sum_a, sum_b, sum_a2, sum_b2, sum_ab, corrcoef;
  i = blockIdx.y*blockDim.y + threadIdx.y;
  j = blockIdx.x*blockDim.x + threadIdx.x;
  sum_a = sum_a2 = sum_b = sum_b2 = sum_ab = 0;
  for (offset=0; offset < ncol; offset += blockDim.x) {
     sA[threadIdx.y][threadIdx.x] = gA[(blockIdx.y*blockDim.y + threadIdx.y)*ncol+offset+threadIdx.x];
     sB[threadIdx.y][threadIdx.x] = gB[(blockIdx.x*blockDim.x + threadIdx.y)*ncol+offset+threadIdx.x];
     __syncthreads();
     for (k=0; k < blockDim.x; k++) {
 	a = sA[threadIdx.y][k];
        b = sB[threadIdx.x][k];
        printf("val1=%4.2f,k=%d , thread=%d,block=%d \n",sA[threadIdx.y][k],k,threadIdx.y,blockDim.x);
       	sum_a += a;
        sum_a2 += a*a;
        sum_b += b;
        sum_b2 += b*b;
        sum_ab += a*b;
     }
     __syncthreads();
  }
  corrcoef = (ncol*sum_ab - sum_a*sum_b)/sqrtf((ncol*sum_a2-sum_a*sum_a)*(ncol*sum_b2-sum_b*sum_b)+epsilon);
  printf("CorrCoeff=%4.2f",corrcoef);
  //if (corrcoef <0){ gC[i*nrow+j] = 1+corrcoef; 
 // } else if (corrcoef >=0) {gC[i*nrow+j]=1+corrcoef;
  //} else {
  //	gC[i*nrow+j] = null;
  //}
  if (corrcoef>0 ){
   atomicAdd(gC + i * nrow + j, 1- abs(corrcoef));
  }
}


extern "C" void matrix_Kendall_distance_same_block(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){

  int array_size = *n * *m;

  float* array_new = new float[*n * *m];

  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  float* d_array;

  cudaMalloc(&d_array, array_size * sizeof(float));

  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);

  int threads = 16;
  int blocks_in_row = (*n + threads - 1) / threads;
  int blocks_in_col = (*n + threads - 1) / threads;

  dim3 THREADS(threads, threads);
  dim3 BLOCKS(blocks_in_row, blocks_in_col);

  unsigned int* d_result;
  unsigned int* h_result = new unsigned int[(*m) * (*m)];
  cudaMalloc(&d_result, (*m) * (*m) * sizeof(unsigned int));
  cudaMemset(d_result, 0, (*m) * (*m) * sizeof(unsigned int));

  Rkendall_gpu_atomic_float<<<BLOCKS, THREADS>>>(d_array, *n, *m, d_result);

  cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);


  for (int i = 0; i < (*m) * (*m); ++i) {
    c[i] = h_result[i] * 2.0f / (*n) / (*n - 1);
  }

  free(h_result);
  cudaFree(d_result);
  cudaFree(d_array);
}


extern "C" void matrix_Euclidean_distance_same_block(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){


  int array_size = *n * *m;

  float* array_new = new float[*n * *m];

  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  float* d_array;

  cudaMalloc(&d_array, array_size * sizeof(float));

  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks_in_row = (*n + threads - 1) / threads;
  int blocks_in_col = *n;


  unsigned int* d_result;
  unsigned int* h_result = new unsigned int[(*m) * (*m)];
  cudaMalloc(&d_result, (*m) * (*m) * sizeof(unsigned int));
  cudaMemset(d_result, 0, (*m) * (*m) * sizeof(unsigned int));

  Reuclidean_gpu_atomic_float<<<blocks_in_row, threads>>>(d_array, *n, *m, d_result);

  cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);


  for (int i = 0; i < (*m) * (*m); ++i) {
    c[i] = sqrtf(h_result[i]);
  }

  free(h_result);
  cudaFree(d_result);
  cudaFree(d_array);
}


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

  int threads = 16;
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

  int threads = 256;
  int blocks_in_row = (*n + threads - 1) / threads;
  int blocks_in_col = *n ;

  unsigned int* d_result;
  unsigned int* h_result = new unsigned int[(*m) * (*m_b)];
  cudaMalloc(&d_result, (*m) * (*m_b) * sizeof(unsigned int));
  cudaMemset(d_result, 0, (*m) * (*m_b) * sizeof(unsigned int));

  Reuclidean_gpu_atomic_float_different_blocks<<<blocks_in_row, threads>>>(d_array, d_array2, *n, *m, *m_b, d_result);

  cudaMemcpy(h_result, d_result, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);


  for (int i = 0; i < (*m) * (*m_b); ++i) {
    c[i] = sqrtf(h_result[i]);
  }

  free(h_result);
  cudaFree(d_result);
  cudaFree(d_array);
  cudaFree(d_array2);
}

extern "C" void matrix_Pearson_distance_same_block(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){
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

  int threads = 256;
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

  RpearsonCorr_gpu_atomic_float_same_block<<<blocks, threads>>>(
    d_array,
    *n, *m,
    d_result,
    d_x_norm_result,
    d_y_norm_result
  );
  cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_x_norm_result, d_x_norm_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_y_norm_result, d_y_norm_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < (*m) * (*m); ++i) {
    // printf("%4.2f ",h_result[i]);
    if(!isnan(h_result[i])){
      if (i == 1 || i == (*m)) {
        printf("%f %f %f\n", h_result[i], h_x_norm_result[i], h_y_norm_result[i]);
      }
      c[i] = 1.0 - h_result[i] / sqrtf(h_x_norm_result[i]) / sqrtf(h_y_norm_result[i]);
    }
  }

  free(h_result);
  cudaFree(d_result);
  cudaFree(d_x_norm_result);
  cudaFree(d_y_norm_result);
  cudaFree(d_array);
  // cudaFree(d_array2);
}

extern "C" void matrix_Pearson_distance_different_blocks(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){
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

  int threads = 16;
  int blocks_in_row = (*n + threads - 1) / threads;
  int blocks_in_col = (*n + threads - 1) / threads;

  dim3 THREADS(threads, threads);
  dim3 BLOCKS(blocks_in_row, blocks_in_col);

  float* d_result;
  float* h_result = new float[(*m) * (*m_b)];

  cudaMalloc(&d_result, (*m) * (*m_b) * sizeof(float)); 
  cudaMemset(d_result, 0, (*m) * (*m_b) * sizeof(float));

  RpearsonCorr_gpu_atomic_float_different_blocks<<<blocks_in_row, threads>>>(d_array,d_array2, *n, *m,*m_b, d_result);
  cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);
 for (int i = 0; i < (*m) * (*m_b); ++i) {
	 printf("%4.2f ",h_result[i]);
	 if(!isnan(h_result[i])){
		 c[i] = (h_result[i]);} //* 2.0f / (*n) / (*n - 1);
  }

  free(h_result);
  cudaFree(d_result);
  cudaFree(d_array);
  cudaFree(d_array2);
}

extern "C" void matrix_Pearson2_distance_different_blocks(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){
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

  int threads = 16;
  int blocks_in_row = (*n + threads - 1) / threads;
  int blocks_in_col = (*n + threads - 1) / threads;

  dim3 THREADS(threads, threads);
  dim3 BLOCKS(blocks_in_row, blocks_in_col);

//  float* d_result;
//  float* h_result = new float[(*m) * (*m_b)];
  float* r1;
  float* h_r1 = new float[(*m) * (*m_b)];
 float* r2;
  float* h_r2 = new float[(*m) * (*m_b)];
 float* r3;
  float* h_r3 = new float[(*m) * (*m_b)];

//  cudaMalloc(&d_result, (*m) * (*m_b) * sizeof(float)); 
//  cudaMemset(d_result, 0, (*m) * (*m_b) * sizeof(float));
  cudaMalloc(&r1, (*m) * (*m_b) * sizeof(float)); 
  cudaMemset(r1, 0, (*m) * (*m_b) * sizeof(float));
  cudaMalloc(&r2, (*m) * (*m_b) * sizeof(float)); 
  cudaMemset(r2, 0, (*m) * (*m_b) * sizeof(float));
  cudaMalloc(&r3, (*m) * (*m_b) * sizeof(float)); 
  cudaMemset(r3, 0, (*m) * (*m_b) * sizeof(float));

  //RpearsonCorr_gpu_atomic_float_different_blocks<<<blocks_in_row, threads>>>(d_array,d_array2, *n, *m,*m_b, r1,r2,r3);
  cudaMemcpy(h_r1, r1, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_r2, r2, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_r3, r3, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);
  
 for (int i = 0; i < (*m) * (*m_b); ++i) {
	 printf("%4.2f ",h_r1[i]/sqrt(h_r2[i]*h_r3[i]));
	 if(!isnan(h_r1[i])){
		 c[i] = (h_r1[i]/sqrt(h_r2[i]*h_r3[i]));} //* 2.0f / (*n) / (*n - 1);
  }

  free(h_r1);
  free(h_r2);
  free(h_r3);
  
  cudaFree(r1);
  cudaFree(r2);
  cudaFree(r3);
  cudaFree(d_array);
  cudaFree(d_array2);
}

extern "C" void file_Kendall_distance(double* a, int* n, int* m, char** fout){
  std::ofstream RESULTFILE(*fout, std::ios::binary|std::ios::app);
  size_t dataset_column_size = *n * sizeof(double);
  size_t reverse_max_size = sizeof(unsigned long long);
  for (int col1 = 0; col1 < *m; col1++){
    double* first_column_device_ptr;
    cudaMalloc(&first_column_device_ptr, dataset_column_size);
    cudaMemcpy(first_column_device_ptr, a + col1 * *n, dataset_column_size, cudaMemcpyHostToDevice);
    for (int col2 = col1 + 1; col2 < *m; col2 ++){
      double* second_column_device_ptr;
      cudaMalloc(&second_column_device_ptr, dataset_column_size);
      cudaMemcpy(second_column_device_ptr, a + col2 * *n, dataset_column_size, cudaMemcpyHostToDevice);
      unsigned long long host_R = 0;
      unsigned long long* device_R;
      cudaMalloc(&device_R, reverse_max_size);
      cudaMemcpy(device_R, &host_R, reverse_max_size, cudaMemcpyHostToDevice);
      int threads = 16;
      int blocks_in_row = (*n + threads - 1) / threads;
      int blocks_in_col = (*n + threads - 1) / threads;

      dim3 THREADS(threads, threads);
      dim3 BLOCKS(blocks_in_row, blocks_in_col);

      Rkendall_gpu_atomic<<<BLOCKS, THREADS>>>(first_column_device_ptr, second_column_device_ptr, *n, *m, device_R);
      cudaDeviceSynchronize();

      cudaMemcpy(&host_R, device_R, reverse_max_size, cudaMemcpyDeviceToHost);

      double distance = host_R * 2.0 / *n / (*n - 1);
      RESULTFILE.write((char*)&distance, sizeof(distance));

      cudaFree(second_column_device_ptr);
      cudaFree(device_R);
    }
    cudaFree(first_column_device_ptr);
  }
  RESULTFILE.close();
}
