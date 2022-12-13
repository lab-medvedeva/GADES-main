//=============================
#include <iostream>
#include <fstream>
#include <R.h>
#include<stdio.h>
#include <cstring>
#include<math.h>
//=========================================

//Naive Implementation of Euclidean_distance_matrix (BruteForce)
extern "C" void matrix_Euclidean_distance_same_block_cpu(double * a, double * b, double * c, int * n, int * m, int * m_b) {
  int array_size = * n * * m;
  float * array_new = new float[ * n * * m];
  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  float * d_array = new float[array_size];

  std::memcpy(d_array, array_new, array_size * sizeof(float));

  //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  unsigned int * h_result = new unsigned int[( * m) * ( * m)];
  std::memset(h_result, 0, ( * m) * ( * m) * sizeof(unsigned int));

  //CPU Implementation
  for (int row = 0; row < * n; row++) {
    //Reuclidean_cpu_atomic_float(i,d_array, *n, *m, d_result);
    for (int col1_num = 0; col1_num < * m; ++col1_num) {
      for (int col2_num = col1_num + 1; col2_num < * m; ++col2_num) {
        float * col1 = d_array + * n * col1_num;
        float * col2 = d_array + * n * col2_num;
        if (row < * n) {
          float diff = col1[row] - col2[row];
          diff = diff * diff;
          h_result[col1_num * * m + col2_num] += diff;
          h_result[col2_num * * m + col1_num] += diff;
        }
      }
    }
  }

  //std::memcpy(h_result, d_result, ( * m) * ( * m) * sizeof(float));

  for (int i = 0; i < ( * m) * ( * m); ++i) {
    c[i] = std::sqrt(h_result[i]); //Using sqrt instead of sqrtf
  }
  //‘free’ is not a member of ‘std’ in gcc -dref at time of integration build.
  free(h_result);
  //std::free(d_result);
  free(d_array);
}
//=============================================
//Naive Implementation of Kendall_distance_matrix for different blocks
extern "C" void matrix_Kendall_distance_same_block_cpu(double * a, double * b, double * c, int * n, int * m, int * m_b) {
  int array_size = * n * * m;
  float * array_new = new float[ * n * * m];
  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  float * d_array = new float[array_size];

  std::memcpy(d_array, array_new, array_size * sizeof(float));

  //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  unsigned int * h_result = new unsigned int[( * m) * ( * m)];
  std::memset(h_result, 0, ( * m) * ( * m) * sizeof(unsigned int));

  //CPU Implementation
  for (int row1 = 0; row1 < * n; row1++) {
    for (int row2=0;row2<*n;row2++){
      for (int col1_num = 0; col1_num < * m; ++col1_num) {
        for (int col2_num = col1_num+1; col2_num < * m; ++col2_num) {
          float * col1 = d_array + * n * col1_num;
          float * col2 = d_array + * n * col2_num;
          if (row1 < row2 && row2 < *n){
            if ((col1[row1] - col1[row2]) * (col2[row1] - col2[row2]) < 0){
              h_result[col1_num * * m + col2_num] += 1;
              h_result[col2_num * * m + col1_num] += 1;
            }
          }      
        }
      }
    }
  }
  
  for (int i = 0; i < ( * m) * ( * m); ++i) {
    c[i] = h_result[i] * 2.0f / (*n) / (*n - 1); //Using sqrt instead of sqrtf
  }
  free(h_result);
  free(d_array);
}

extern "C" void  matrix_Pearson_distance_same_block_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b){
  int array_size = * n * * m;
  float * array_new = new float[ * n * * m];
  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  float * d_array = new float[array_size];

  std::memcpy(d_array, array_new, array_size * sizeof(float));

  //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  unsigned int * h_r1 = new unsigned int[( * m) * ( * m)];
  std::memset(h_r1, 0, ( * m) * ( * m) * sizeof(unsigned int));
 
  //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  unsigned int * h_r2 = new unsigned int[( * m) * ( * m)];
  std::memset(h_r2, 0, ( * m) * ( * m) * sizeof(unsigned int));
 
 //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  unsigned int * h_r3 = new unsigned int[( * m) * ( * m)];
  std::memset(h_r3, 0, ( * m) * ( * m) * sizeof(unsigned int));
 

  for (int row=0;row<*n;row++){
    for (int col1_num = 0; col1_num < * m; ++col1_num) {
      for (int col2_num = col1_num+1; col2_num < * m; ++col2_num) {
        float * col1 = d_array + * n * col1_num;
        float * col2 = d_array + * n * col2_num;
        if (row < *n ) {    
	    float num = (col1[row] * col2[row]);
            float sum1 = (col1[row] * col1[row]);
            float sum2 = (col2[row] * col2[row]);
	    h_r1[col1_num * * m + col2_num] += num;
            h_r2[col1_num * * m + col2_num] += sum1;
            h_r3[col1_num * * m + col2_num] += sum2;
            h_r1[col2_num * * m + col1_num] += num;
            h_r2[col2_num * * m + col1_num] += sum1;
            h_r3[col2_num * * m + col1_num] += sum2;
                                  
        }
      }
    }
  }

 int j=0;
  for (int i = 0; i < (*m) * (*m); ++i) {
    // printf("%4.2f ",h_result[i]);

    if(!isnan(h_r1[i])){
      //if (i == 1 || i == (*m)) {
      //  printf("%f %f %f\n", h_result[i], h_x_norm_result[i], h_y_norm_result[i]);
      //}
      if (i == j * (*m+1)){
       c[i] = 0.0; //1.0 - h_result[i] / sqrtf(h_x_norm_result[i]) / sqrtf(h_y_norm_result[i]);
       j++;
      } else {
        c[i] = 1.0 - h_r1[i] / sqrtf(h_r2[i]) / sqrtf(h_r2[i]);
      }
    }
  }

  free(h_r1);
  free(h_r2);
  free(h_r3);
  free(d_array);
}

//======================================================

extern "C" void matrix_Euclidean_distance_different_block_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b) {
  int array_size = * n * * m;
  float * array_new = new float[ * n * * m];
  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  int array_size2 = * n * (*m_b);
  float * array2_new = new float[array_size2];
  for (int i = 0; i < array_size2; ++i) {
    array2_new[i] = b[i];
  }

  float * d_array = new float[array_size];
  float *d_array2 = new float[array_size];

  std::memcpy(d_array, array_new, array_size * sizeof(float));
  std::memcpy(d_array2, array2_new, array_size * sizeof(float));

  //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  unsigned int * h_result = new unsigned int[( * m) * ( * m_b)];
  std::memset(h_result, 0, ( * m) * ( * m_b) * sizeof(unsigned int));

  //CPU Implementation
  for (int row = 0; row < * n; row++) {
    for (int col1_num = 0; col1_num < * m; ++col1_num) {
      for (int col2_num = 0; col2_num < * m_b; ++col2_num) {
        float * col1 = d_array + * n * col1_num;
        float * col2 = d_array + * n * col2_num;
        if (row < * n) {
          float diff = col1[row] - col2[row];
          diff = diff * diff;
          h_result[col2_num * * m + col1_num] += diff;
        }
      }
    }
  }


  for (int i = 0; i < ( * m) * ( * m); ++i) {
    c[i] = std::sqrt(h_result[i]); //Using sqrt instead of sqrtf
  }
  free(h_result);
  free(d_array);
  free(d_array2);
}

extern "C" void  matrix_Kendall_distance_different_block_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b){
  int array_size = * n * * m;
  float * array_new = new float[ * n * * m];
  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  int array_size2 = * n * (*m_b);
  float * array2_new = new float[array_size2];
  for (int i = 0; i < array_size2; ++i) {
    array2_new[i] = b[i];
  }

  float * d_array = new float[array_size];
  float *d_array2 = new float[array_size];

  std::memcpy(d_array, array_new, array_size * sizeof(float));
  std::memcpy(d_array2, array2_new, array_size * sizeof(float));

  unsigned int * h_result = new unsigned int[( * m) * ( * m_b)];
  std::memset(h_result, 0, ( * m) * ( * m_b) * sizeof(unsigned int));

  //CPU Implementation
  for (int row1 = 0; row1 < * n; row1++) {
    for (int row2=0;row2<*n;row2++){
    for (int col1_num = 0; col1_num < * m; ++col1_num) {
      for (int col2_num = 0; col2_num < * m_b; ++col2_num) {
        float * col1 = d_array + * n * col1_num;
        float * col2 = d_array2 + * n * col2_num;
        if (row1 < row2 && row2 < *n){
          if ((col1[row1] - col1[row2]) * (col2[row1] - col2[row2]) < 0){
            h_result[col2_num * * m + col1_num] += 1;
          }
        }
     }
    }
    }
  }


  for (int i = 0; i < ( * m) * ( * m); ++i) {
    c[i] = h_result[i] * 2.0f / (*n) / (*n - 1);
  }
  free(h_result);
  free(d_array);
  free(d_array2);
}

extern "C" void  matrix_Pearson_distance_different_blocks_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b){
  int array_size = * n * * m;
  float * array_new = new float[ * n * * m];
  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  int array_size2 = * n * (*m_b);
  float * array2_new = new float[array_size2];
  for (int i = 0; i < array_size2; ++i) {
    array2_new[i] = b[i];
  }
  float * d_array = new float[array_size];
  float *d_array2 = new float[array_size];

  std::memcpy(d_array, array_new, array_size * sizeof(float));
  std::memcpy(d_array2, array2_new, array_size * sizeof(float));

  //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  unsigned int * h_r1 = new unsigned int[( * m) * ( * m_b)];
  std::memset(h_r1, 0, ( * m) * ( * m_b) * sizeof(unsigned int));
 
  //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  unsigned int * h_r2 = new unsigned int[( * m) * ( * m_b)];
  std::memset(h_r2, 0, ( * m) * ( * m_b) * sizeof(unsigned int));
 
 //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  unsigned int * h_r3 = new unsigned int[( * m) * ( * m_b)];
  std::memset(h_r3, 0, ( * m) * ( * m_b) * sizeof(unsigned int));
 

  for (int row=0;row<*n;row++){
    for (int col1_num = 0; col1_num < * m; ++col1_num) {
      for (int col2_num = 0; col2_num < * m_b; ++col2_num) {
        float * col1 = d_array + * n * col1_num;
        float * col2 = d_array + * n * col2_num;
        if (row < *n ) {    
	    float num = (col1[row] * col2[row]);
            float sum1 = (col1[row] * col1[row]);
            float sum2 = (col2[row] * col2[row]);
	    h_r1[col1_num * * m_b + col2_num] += num;
            h_r2[col1_num * * m_b + col2_num] += sum1;
            h_r3[col1_num * * m_b + col2_num] += sum2;
            h_r1[col2_num * * m + col1_num] += num;
            h_r2[col2_num * * m + col1_num] += sum1;
            h_r3[col2_num * * m + col1_num] += sum2;
                                  
       }
      }
    }
  }

 int j=0;
  for (int i = 0; i < (*m) * (*m); ++i) {
    // printf("%4.2f ",h_result[i]);

    if(!isnan(h_r1[i])){
      //if (i == 1 || i == (*m)) {
      //  printf("%f %f %f\n", h_result[i], h_x_norm_result[i], h_y_norm_result[i]);
      //}
      if (i == j * (*m+1)){
       c[i] = 0.0; //1.0 - h_result[i] / sqrtf(h_x_norm_result[i]) / sqrtf(h_y_norm_result[i]);
       j++;
      } else {
        c[i] = 1.0 - h_r1[i] / sqrtf(h_r2[i]) / sqrtf(h_r2[i]);
      }
    }
  }

  free(h_r1);
  free(h_r2);
  free(h_r3);
  free(d_array);
  free(d_array2);
}
extern "C" void  matrix_PearsonChi_distance_different_blocks_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b){
  int array_size = * n * * m;
  float * array_new = new float[ * n * * m];
  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }

  int array_size2 = * n * (*m_b);
  float * array2_new = new float[array_size2];
  for (int i = 0; i < array_size2; ++i) {
    array2_new[i] = b[i];
  }
  float epsilon=0.01;
  float * d_array = new float[array_size];
  float *d_array2 = new float[array_size];

  std::memcpy(d_array, array_new, array_size * sizeof(float));
  std::memcpy(d_array2, array2_new, array_size * sizeof(float));

  //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  unsigned int * h_result = new unsigned int[( * m) * ( * m_b)];
  std::memset(h_result, 0, ( * m) * ( * m_b) * sizeof(unsigned int));
 
 
  float dist,num=0,sum1=0,sum2=0; 
  float sumx=0,sumy=0,sumxx=0,sumyy=0,sumxy=0,denum2=0,count=0;
  for (int row=0;row<*n;row++){
    for (int col1_num = 0; col1_num < * m; ++col1_num) {
      for (int col2_num = 0; col2_num < * m_b; ++col2_num) {
        float * col1 = d_array + * n * col1_num;
        float * col2 = d_array + * n * col2_num;
        if (row < *n ) {    
	    sumxy = (col1[row] * col2[row]);
      	    sumx = col1[row];
      	    sumy = col2[row];
            sumxx = col1[row] *col1[row];
            sumyy = col1[row] * col2[row];
	    num = sumxy - ( sumx*sumy /count );
	    denum2 =  (sumxx - (sumx*sumx /count ) )* (sumyy - (sumy*sumy /count ) ) ;
	    //if(col2_num<2){std::cout<<"("<<num<<denum2<<")";std::cout<<"\n";}
  //num += col1[row] * col2[row];
//	    sum1 += col1[row] * col1[row];
//	    sum2 += col2[row] * col2[row];
	   dist = (num/sqrt(sum1*sum2));
            //std::cout << "("<<dist<<")";
    if(col2_num<2){std::cout<<"("<<num<<","<<denum2<<","<<(1-num/sqrt(denum2))<<")";std::cout<<"\n";}
 	if(denum2 <=0) {
      		h_result[col2_num* *m + col1_num]+=0;
    	} else{
	    h_result[col2_num * * m + col1_num] += (1-num/sqrt(denum2));
	}
	    count++;
	}
      }
    }      
   }
 for (int i = 0; i < ( * m) * ( * m); ++i) {
    c[i] =  h_result[i];
  }
/*  
  //CPU Implementation
  for (int row = 0; row < * n; row++) {
    //Reuclidean_cpu_atomic_float(i,d_array, *n, *m, d_result);
    for (int col1_num = 0; col1_num < * m; ++col1_num) {
      for (int col2_num = 0; col2_num < * m_b; ++col2_num) {
        float * col1 = d_array + * n * col1_num;
        float * col2 = d_array + * n * col2_num;
        if (row < *m ) {
          if(col2[row]==0.0){
          
            float diff = col1[row] - col2[row];
            diff = diff * diff;
            diff = diff/epsilon;
            
            h_result[col2_num * * m + col1_num] += diff;
          } else {
            float diff = col1[row] - col2[row];
            diff = diff * diff;
            diff = diff/col2[row];
            //atomicAdd(result + col1_num * m + col2_num, diff);
            h_result[col2_num * * m + col1_num] += diff;
          }
        }
      }
    }
  }
  for (int i = 0; i < ( * m) * ( * m); ++i) {
    c[i] =  h_result[i] * 2.0f / (*n) / (*n - 1);
  }*/
  free(h_result);
  free(d_array);
  free(d_array2);
}
