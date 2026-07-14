#include <iostream>
#include <fstream>
#include <R.h>
#include <Rinternals.h>
#include <stdio.h>
#include <cstring>
#include <math.h>
#include <cassert>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <cblas.h>
#include "pc_runtime_cpu.h"
#include "pc_linalg_cpu.h"
#include "pc_corr_core_cpu.h"



// ==================== Pearson helpers ====================

static float* csr_to_dense_cpu(int* index, int* positions, double* values, int rows, int columns) {
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


static void center_columns_cpu(float* array, int n, int m) {
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


// ============== .Call fast path: CPU dense + sparse, full matrix =============
// CPU analogue of the GPU C_dense_block / C_sparse_block. Runs the WHOLE matrix
// through the existing (validated, cblas+OpenMP) same-block CPU drivers in one
// .Call with R-owned input buffers and an allocVector output — eliminating the
// .C argument duplication (esp. the m x m double output) and the per-block
// matrix coercion of the block loop. There is no host<->device transfer on CPU,
// so this captures the full win. metric: 0 euc,1 cos,2 pear,3 manh,4 spear,5 ken.

typedef void (*dense_drv_cpu_t)(double*, double*, double*, int*, int*, int*);


extern "C" SEXP C_dense_block_cpu(SEXP a, SEXP n_, SEXP m_, SEXP metric_) {
    int n = INTEGER(n_)[0], m = INTEGER(m_)[0], metric = INTEGER(metric_)[0];
    dense_drv_cpu_t fn = NULL;
    switch (metric) {
        case 0: fn = matrix_Euclidean_distance_same_block_cpu; break;
        case 1: fn = matrix_Cosine_distance_same_block_cpu;    break;
        case 2: fn = matrix_Pearson_distance_same_block_cpu;   break;
        case 3: fn = matrix_Manhattan_distance_same_block_cpu; break;
        case 4: fn = matrix_Spearman_distance_same_block_cpu;  break;
        case 5: fn = matrix_Kendall_distance_same_block_cpu;   break;
    }
    if (!fn) return R_NilValue;
    double* da = REAL(a);
    SEXP out = PROTECT(allocVector(REALSXP, (R_xlen_t)m * m));
    int N = n, M = m;
    fn(da, da, REAL(out), &N, &M, &M);
    UNPROTECT(1);
    return out;
}


typedef void (*sparse_drv_cpu_t)(int*, int*, double*, int*, int*, double*, double*,
                                 int*, int*, int*, int*, int*);


extern "C" SEXP C_sparse_block_cpu(SEXP ai, SEXP ap, SEXP ax, SEXP n_, SEXP m_,
                                   SEXP nnz_, SEXP metric_, SEXP layout_) {
    int n = INTEGER(n_)[0], m = INTEGER(m_)[0], nnz = INTEGER(nnz_)[0];
    int metric = INTEGER(metric_)[0];
    SEXP out = PROTECT(allocVector(REALSXP, (R_xlen_t)m * m));
    double* res = REAL(out);
    int N = n, M = m, NNZ = nnz;

    if (metric <= 2) {                       // euclidean/cosine/pearson: densify CSC->dense
        int* aidx = INTEGER(ai);
        int* apos = INTEGER(ap);
        double* axx = REAL(ax);
        double* dense = (double*)calloc((size_t)n * m, sizeof(double));
        #pragma omp parallel for schedule(static)
        for (int col = 0; col < m; ++col)
            for (int j = apos[col]; j < apos[col + 1]; ++j)
                dense[(size_t)col * n + aidx[j]] = axx[j];
        dense_drv_cpu_t fn = (metric == 0) ? matrix_Euclidean_distance_same_block_cpu
                           : (metric == 1) ? matrix_Cosine_distance_same_block_cpu
                                           : matrix_Pearson_distance_same_block_cpu;
        fn(dense, dense, res, &N, &M, &M);
        free(dense);
    } else {                                 // manh/spear/kendall: per_cell_pair (CSC)
        sparse_drv_cpu_t fn = NULL;
        switch (metric) {
            case 3: fn = matrix_Manhattan_sparse_per_cell_pair_distance_same_block_cpu; break;
            case 4: fn = matrix_Spearman_sparse_per_cell_pair_distance_same_block_cpu;  break;
            case 5: fn = matrix_Kendall_sparse_per_cell_pair_distance_same_block_cpu;   break;
        }
        if (!fn) { UNPROTECT(1); return R_NilValue; }
        int* aidx = INTEGER(ai);
        int* apos = INTEGER(ap);
        double* axx = REAL(ax);
        fn(aidx, apos, axx, aidx, apos, axx, res, &N, &M, &M, &NNZ, &NNZ);
    }
    UNPROTECT(1);
    return out;
}
