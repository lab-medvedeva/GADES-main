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
  float * h_result = new float[( * m) * ( * m)];
  std::memset(h_result, 0, ( * m) * ( * m) * sizeof(float));

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

extern "C" void  matrix_Pearson_distance_same_block_cpu(
  double* a,
  double* b,
  double* c,
  int* n,
  int* m,
  int* m_b
) {
  int array_size = * n * * m;
  float * array_new = new float[ * n * * m];
  for (int i = 0; i < array_size; ++i) {
    array_new[i] = a[i];
  }


  //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  float * h_scalar = new float[( * m) * ( * m)];
  std::memset(h_scalar, 0, ( * m) * ( * m) * sizeof(float));
 
  //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  float * h_prod1 = new float[( * m) * ( * m)];
  std::memset(h_prod1, 0, ( * m) * ( * m) * sizeof(float));
 
 //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  float * h_prod2 = new float[( * m) * ( * m)];
  std::memset(h_prod2, 0, ( * m) * ( * m) * sizeof(float));
 

  for (int row=0;row<*n;row++){
    for (int col1_num = 0; col1_num < * m; ++col1_num) {
      for (int col2_num = col1_num; col2_num < * m; ++col2_num) {
        float * col1 = array_new + * n * col1_num;
        float * col2 = array_new + * n * col2_num;
        float num = (col1[row] * col2[row]);
        float sum1 = (col1[row] * col1[row]);
        float sum2 = (col2[row] * col2[row]);
        // if (col1_num == 0 && col2_num == 1) {
        //   std::cout << row << " " << num << std::endl;
        // }
        h_scalar[col1_num * * m + col2_num] += num;
        h_prod1[col1_num * * m + col2_num] += sum1;
        h_prod2[col1_num * * m + col2_num] += sum2;
        h_scalar[col2_num * * m + col1_num] += num;
        h_prod1[col2_num * * m + col1_num] += sum1;
        h_prod2[col2_num * * m + col1_num] += sum2;
      }
    }
  }

  int j=0;
  for (int i = 0; i < (*m) * (*m); ++i) {
    c[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);

  }

  free(h_prod1);
  free(h_prod2);
  free(h_scalar);
}

//======================================================

extern "C" void matrix_Euclidean_distance_different_blocks_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b) {
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

extern "C" void  matrix_Kendall_distance_different_blocks_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b){
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
  unsigned int * h_scalar = new unsigned int[( * m) * ( * m_b)];
  std::memset(h_scalar, 0, ( * m) * ( * m_b) * sizeof(unsigned int));
 
  //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  unsigned int * h_prod1 = new unsigned int[( * m) * ( * m_b)];
  std::memset(h_prod1, 0, ( * m) * ( * m_b) * sizeof(unsigned int));
 
 //unsigned int * d_result = new unsigned int[( * m) * ( * m)];
  unsigned int * h_prod2 = new unsigned int[( * m) * ( * m_b)];
  std::memset(h_prod2, 0, ( * m) * ( * m_b) * sizeof(unsigned int));
 

  for (int row=0;row<*n;row++){
    for (int col1_num = 0; col1_num < * m; ++col1_num) {
      for (int col2_num = col1_num+1; col2_num < * m_b; ++col2_num) {
        float * col1 = d_array + * n * col1_num;
        float * col2 = d_array + * n * col2_num;
        if (row < *n ) {    
	    float num = (col1[row] * col2[row]);
            float sum1 = (col1[row] * col1[row]);
            float sum2 = (col2[row] * col2[row]);
	    h_scalar[col1_num * * m_b + col2_num] += num;
            h_prod1[col1_num * * m_b + col2_num] += sum1;
            h_prod2[col1_num * * m_b + col2_num] += sum2;
            h_scalar[col2_num * * m + col1_num] += num;
            h_prod1[col2_num * * m + col1_num] += sum1;
            h_prod2[col2_num * * m + col1_num] += sum2;
                                  
       }
      }
    }
  }
    for (int i = 0; i < (*m) * (*m); ++i) {
    c[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);

  }
/*
 int j=0;
  for (int i = 0; i < (*m) * (*m); ++i) {
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
*/
  free(h_prod1);
  free(h_prod2);
  free(h_scalar);
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


extern "C" void  matrix_Euclidean_sparse_distance_same_block_cpu(
  int *a_index,
  int *a_positions,
  double *a_double_values,
  int *b_index,
  int *b_positions,
  double *b_double_values,
  double *result,
  int *num_rows,
  int *num_columns,
  int *num_columns_b,
  int *num_elements_a,
  int *num_elements_b
){
  int rows = *num_rows;
  int columns = *num_columns;

  float * a_values = new float[*num_elements_a];
  float * float_result = new float[columns * columns];
  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = a_double_values[i];
  }

  for (int i = 0; i < columns * columns; ++i) {
    float_result[i] = 0.0f;
  }

  for (int row_index = 0; row_index < rows; ++row_index) {
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

        float value1 = a_values[col1_index];
        float value2 = a_values[col2_index];

        for (int left = prev_col + 1; left < col1; ++left) {
          float_result[left * columns + col2] += value2 * value2;
          float_result[col2 * columns + left] += value2 * value2;
          // if (left + col2 == 1) {
          //   std::cout << "L" << " " << prev_col << " " << row_index << " left " << left << " col2 " << col2 << " col1 " << col1 << " "  << value2 << std::endl;
          // }
        }

        for (int right = col2 + 1; right < next_col; ++right) {
          float_result[right * columns + col1] += value1 * value1;
          float_result[col1 * columns + right] += value1 * value1;

          // if (right + col1 == 1) {
          //   std::cout << "R" << row_index << " " << right << " " << col1 << " " << col2 << " " << value1 << std::endl;
          // }
        }

        float_result[col1 * columns + col2] += (value1 - value2) * (value1 - value2);
        float_result[col2 * columns + col1] += (value1 - value2) * (value1 - value2);
        // if (col1 + col2 == 1) {
        //     std::cout << "D" << row_index << " " << col1 << " " << col2 << " " << value1 - value2 << std::endl;
        // }
      }
    }
  }

  for (int i = 0; i < columns * columns; ++i) {
    result[i] = std::sqrt(float_result[i]);
  }
  
  free(float_result);
  free(a_values);
}

extern "C" void  matrix_Euclidean_sparse_distance_different_blocks_cpu(
  int *a_index,
  int *a_positions,
  double *a_double_values,
  int *b_index,
  int *b_positions,
  double *b_double_values,
  double *result,
  int *num_rows,
  int *num_columns,
  int *num_columns_b,
  int *num_elements_a,
  int *num_elements_b
){
  int rows = *num_rows;
  int columns = *num_columns;
  int columns_b = *num_columns_b;

  float * a_values = new float[*num_elements_a];
  float * float_result = new float[columns * columns_b];
  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = a_double_values[i];
  }

  for (int i = 0; i < columns * columns_b; ++i) {
    float_result[i] = 0.0f;
  }

  float * b_values = new float[*num_elements_b];
  for (int i = 0; i < *num_elements_b; ++i) {
    b_values[i] = b_double_values[i];
  }


  for (int row_index = 0; row_index < rows; ++row_index) {
    int start_column = a_positions[row_index];
    int end_column = a_positions[row_index + 1];

    int start_column_b = b_positions[row_index];
    int end_column_b = b_positions[row_index + 1];

    for (int col1_index = start_column; col1_index <= end_column; ++col1_index) {
      int prev_col_index = col1_index - 1;
      int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
      float value1 = (col1_index < end_column) ? a_values[col1_index] : 0.0f;

      int col1 = (col1_index < end_column) ? a_index[col1_index] : columns;

      for (int col2_index = start_column_b; col2_index <= end_column_b; ++col2_index) {
        // std::cout << col1_index << " " << start_column << " " << end_column << " " << col2_index << std::endl;
        int prev_col_b_index = col2_index - 1;
        int prev_col2 = (prev_col_b_index >= start_column_b) ? b_index[prev_col_b_index] : -1;

        int col2 = (col2_index < end_column_b) ? b_index[col2_index] : columns_b;

        
        float value2 = (col2_index < end_column_b) ? b_values[col2_index] : 0.0f;
        
        if (col2 < columns_b) {
          for (int left = prev_col + 1; left < col1; ++left) {
            float_result[col2 * columns + left] += value2 * value2;
          }
        }
        if (col1 < columns) {
          for (int left = prev_col2 + 1; left < col2; ++left) {
            float_result[left * columns + col1] += value1 * value1;
          }
        }

        if (col1 < columns && col2 < columns_b) {
          float_result[col2 * columns + col1] += (value1 - value2) * (value1 - value2);
        }
        // std::cout << "Done" << std::endl;
        // if (col1 + col2 == 1) {
        //     std::cout << "D" << row_index << " " << col1 << " " << col2 << " " << value1 - value2 << std::endl;
        // }
      }
    }
  }

  for (int i = 0; i < columns * columns_b; ++i) {
    result[i] = std::sqrt(float_result[i]);
  }
  
  free(float_result);
  free(a_values);
  free(b_values);
}


extern "C" void matrix_Pearson_sparse_distance_same_block_cpu(
  int *a_index,
  int *a_positions,
  double *a_double_values,
  int *b_index,
  int *b_positions,
  double *b_double_values,
  double *result,
  int *num_rows,
  int *num_columns,
  int *num_columns_b,
  int *num_elements_a,
  int *num_elements_b
){
  int rows = *num_rows;
  int columns = *num_columns;

  float * a_values = new float[*num_elements_a];

  float * float_result = new float[columns * columns];
  

  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = a_double_values[i];
  }

  for (int i = 0; i < columns * columns; ++i) {
    float_result[i] = 0.0f;
  }

  float * squares = new float[columns];
  for (int i = 0; i < columns; ++i) {
    squares[i] = 0.0f;
  }


  for (int row_index = 0; row_index < rows; ++row_index) {
    int start_column = a_positions[row_index];
    int end_column = a_positions[row_index + 1];

    for (int col1_index = start_column; col1_index < end_column; ++col1_index) {

      int prev_col_index = col1_index - 1;
      int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
      int col1 = a_index[col1_index];
      float value1 = a_values[col1_index];
      squares[col1] += value1 * value1;

      for (int col2_index = col1_index + 1; col2_index < end_column; ++col2_index) {
        

        int next_col_index = col2_index + 1;
        int next_col = (next_col_index < end_column) ? a_index[next_col_index] : columns;

        int col2 = a_index[col2_index];

        
        float value2 = a_values[col2_index];

        // for (int left = prev_col + 1; left < col1; ++left) {
        //   float_result[left * columns + col2] += value2 * value2;
        //   float_result[col2 * columns + left] += value2 * value2;
        //   // if (left + col2 == 1) {
        //   //   std::cout << "L" << " " << prev_col << " " << row_index << " left " << left << " col2 " << col2 << " col1 " << col1 << " "  << value2 << std::endl;
        //   // }
        // }

        // for (int right = col2 + 1; right < next_col; ++right) {
        //   float_result[right * columns + col1] += value1 * value1;
        //   float_result[col1 * columns + right] += value1 * value1;

        //   // if (right + col1 == 1) {
        //   //   std::cout << "R" << row_index << " " << right << " " << col1 << " " << col2 << " " << value1 << std::endl;
        //   // }
        // }

        float_result[col1 * columns + col2] += value1 * value2;
        float_result[col2 * columns + col1] += value1 * value2;
        // if (col1 + col2 == 1) {
        //     std::cout << "D" << row_index << " " << col1 << " " << col2 << " " << value1 - value2 << std::endl;
        // }
      }
    }
  }
  for (int i = 0; i < columns * columns; ++i) {
    int row_index = i / columns;
    int column_index = i % columns;
    if (row_index != column_index) {
        result[i] = 1.0f - float_result[i] / std::sqrt(squares[row_index]) / std::sqrt(squares[column_index]);
    } else {
        result[i] = 0.0f;
    }
  }
  
  free(float_result);
  free(squares);
  free(a_values);
}

extern "C" void  matrix_Pearson_sparse_distance_different_blocks_cpu(
  int *a_index,
  int *a_positions,
  double *a_double_values,
  int *b_index,
  int *b_positions,
  double *b_double_values,
  double *result,
  int *num_rows,
  int *num_columns,
  int *num_columns_b,
  int *num_elements_a,
  int *num_elements_b
){
  std::cerr << "Called" << std::endl;
  int rows = *num_rows;
  int columns = *num_columns;
  int columns_b = *num_columns_b;

  float * a_values = new float[*num_elements_a];
  float * float_result = new float[columns * columns_b];
  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = a_double_values[i];
  }

  for (int i = 0; i < columns * columns_b; ++i) {
    float_result[i] = 0.0f;
  }

  float * b_values = new float[*num_elements_b];
  for (int i = 0; i < *num_elements_b; ++i) {
    b_values[i] = b_double_values[i];
  }

  float * squares_a = new float[columns];
  for (int i = 0; i < columns; ++i) {
    squares_a[i] = 0.0f;
  }

  float * squares_b = new float[columns_b];
  for (int i = 0; i < columns_b; ++i) {
    squares_b[i] = 0.0f;
  }


  for (int row_index = 0; row_index < rows; ++row_index) {
    int start_column = a_positions[row_index];
    int end_column = a_positions[row_index + 1];

    int start_column_b = b_positions[row_index];
    int end_column_b = b_positions[row_index + 1];

    for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
      float value1 = a_values[col1_index];

      int col1 = a_index[col1_index];

      for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
        // std::cout << col1_index << " " << start_column << " " << end_column << " " << col2_index << std::endl;
        int col2 = b_index[col2_index];       
        float value2 = b_values[col2_index];

        float_result[col2 * columns + col1] += value1 * value2;

      }
    }

    for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
      float value1 = a_values[col1_index];

      int col1 = a_index[col1_index];

      squares_a[col1] += value1 * value1;
    }
    for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
      float value2 = b_values[col2_index];
      int col2 = b_index[col2_index];

      // printf(
      //         "ROW: %d %d %d %d %d %f\n",
      //         row_index,
      //         start_column_b,
      //         col2_index,
      //         end_column_b,
      //         col2,
      //         value2
      //       );
      squares_b[col2] += value2 * value2;
    }
  }

  for (int i = 0; i < 5; ++i) {
    std::cerr << squares_a[i] << " " << squares_b[i] << std::endl;
  }

  // for (int i = 0; i < *num_elements_b; ++i) {
  //   std::cerr << "B: " << i << " " << b_values[i] << " " << b_index[i] << std::endl;
  // }

  for (int i = 0; i < columns * columns_b; ++i) {
    int row_index = i / columns;
    int column_index = i % columns;
    result[i] = 1.0f - float_result[i] / std::sqrt(squares_b[row_index]) / std::sqrt(squares_a[column_index]);
  }
  
  free(float_result);
  free(a_values);
  free(b_values);
  free(squares_a);
  free(squares_b);
}

extern "C" void matrix_Kendall_sparse_distance_same_block_cpu(
  int *a_index,
  int *a_positions,
  double *a_double_values,
  int *b_index,
  int *b_positions,
  double *b_double_values,
  double *result,
  int *num_rows,
  int *num_columns,
  int *num_columns_b,
  int *num_elements_a,
  int *num_elements_b
  /*double *result,
  int *num_rows,
  int *num_columns,
  int *num_elements_a*/
) {
  int rows = *num_rows;
  int columns = *num_columns;

  unsigned int *h_result = new unsigned int[columns * columns];
  std::memset(h_result, 0, columns * columns * sizeof(unsigned int));

  float *a_values = new float[*num_elements_a];
  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = static_cast<float>(a_double_values[i]);
  }

  for (int row_index = 0; row_index < rows; ++row_index) {
    int start_column = a_positions[row_index];
    int end_column = a_positions[row_index + 1];

    for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
      for (int col2_index = col1_index + 1; col2_index < end_column; ++col2_index) {
        int col1 = a_index[col1_index];
        int col2 = a_index[col2_index];

        float value1 = a_values[col1_index];
        float value2 = a_values[col2_index];
        if ((value1 - value2) * (value1 - value2) < 0) {
          h_result[col1 * columns + col2] += 1;
          h_result[col2 * columns + col1] += 1;
        }
      }
    }
  }
  for (int i = 0; i < columns * columns; ++i) {
    result[i] = static_cast<double>(h_result[i]);
  }

  delete[] h_result;
  delete[] a_values;
}


extern "C" void matrix_Kendall_sparse_distance_different_blocks_cpu(
  int *a_index,
  int *a_positions,
  double *a_double_values,
  int *b_index,
  int *b_positions,
  double *b_double_values,
  double *result,
  int *num_rows,
  int *num_columns,
  int *num_columns_b,
  int *num_elements_a,
  int *num_elements_b
) {
  
  int rows = *num_rows;
  int columns = *num_columns;
  int columns_b = *num_columns_b;

  unsigned int *h_result = new unsigned int[columns * columns_b];
  std::memset(h_result, 0, columns * columns_b * sizeof(unsigned int));
  float *a_values = new float[*num_elements_a];
  float *b_values = new float[*num_elements_b];

  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = static_cast<float>(a_double_values[i]);
  }

  for (int i = 0; i < *num_elements_b; ++i) {
    b_values[i] = static_cast<float>(b_double_values[i]);
  }

  for (int row_index = 0; row_index < rows; ++row_index) {
    int start_column_a = a_positions[row_index];
    int end_column_a = a_positions[row_index + 1];

    int start_column_b = b_positions[row_index];
    int end_column_b = b_positions[row_index + 1];
    for (int col1_index_a = start_column_a; col1_index_a < end_column_a; ++col1_index_a) {
      for (int col1_index_b = start_column_b; col1_index_b < end_column_b; ++col1_index_b) {
        int col1_a = a_index[col1_index_a];
        int col1_b = b_index[col1_index_b];

        float value1_a = a_values[col1_index_a];
        float value1_b = b_values[col1_index_b];

        for (int col2_index_a = col1_index_a + 1; col2_index_a < end_column_a; ++col2_index_a) {
          for (int col2_index_b = col1_index_b + 1; col2_index_b < end_column_b; ++col2_index_b) {
            int col2_a = a_index[col2_index_a];
            int col2_b = b_index[col2_index_b];

            float value2_a = a_values[col2_index_a];
            float value2_b = b_values[col2_index_b];

            if ((value1_a - value2_a) * (value1_b - value2_b) < 0) {
              h_result[col1_b * columns + col1_a] += 1;
              h_result[col1_a * columns_b + col1_b] += 1;
            }
          }
        }
      }
    }
  }

  for (int i = 0; i < columns * columns_b; ++i) {
    result[i] = static_cast<double>(h_result[i]);
  }

  delete[] h_result;
  delete[] a_values;
  delete[] b_values;
}

