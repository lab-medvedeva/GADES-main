#include <time.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cublas_v2.h>
#include <R.h>
#include <Rinternals.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "pc_runtime.cuh"
#include "pc_linalg.cuh"
#include "pc_corr_core.cuh"



// ============== .Call fast path: dense Euclidean/Cosine/Pearson ==============
// The .C() ABI duplicates every argument — for a 10000x10000 block that means
// copying the 800 MB output twice. These .Call drivers write straight into the
// R-allocated output (no duplication) and OpenMP-parallelize the host double<->
// float conversions. The float GPU compute path is byte-for-byte the same as
// pc_drive_* (same cublasSgemm + finalize kernels), so results are identical.

// Pinned host-buffer helpers + persistent pinned staging pool moved to
// pc_runtime (pc_pinned_f/pc_host_d2f/pc_host_f2d/pc_rt_log,
// pc_pin/pc_d2f_pin/pc_d2f_pin_b/pc_srcblk_pin, g_pin_in/out/in_b) — ADR-0002.
static void pc_call_euclidean_same(const double* da, double* dc, int n, int m) {
    bool L = pc_rt_log(); double t0=L?omp_get_wtime():0;
    size_t sz = (size_t)n * m, osz = (size_t)m * m;
    float* h = pc_d2f_pin(da, sz);
    double t1=L?omp_get_wtime():0;
    float *d_A, *d_D;
    cudaMalloc(&d_A, sz * sizeof(float));
    cudaMalloc(&d_D, osz * sizeof(float));
    cudaMemcpy(d_A, h, sz * sizeof(float), cudaMemcpyHostToDevice);
    if(L) cudaDeviceSynchronize(); double t2=L?omp_get_wtime():0;
    pc_euclidean_same_block_device(d_A, n, m, d_D);
    if(L) cudaDeviceSynchronize(); double t3=L?omp_get_wtime():0;
    float* outf = pc_pin(&g_pin_out, &g_pin_out_sz, osz);
    cudaMemcpy(outf, d_D, osz * sizeof(float), cudaMemcpyDeviceToHost);
    if(L) cudaDeviceSynchronize(); double t4=L?omp_get_wtime():0;
    pc_host_f2d(outf, dc, osz);
    double t5=L?omp_get_wtime():0;
    cudaFree(d_A); cudaFree(d_D);
    if(L) Rprintf("[rt euc n=%d m=%d |W|=%.0e out=%.0e] d2f=%.0f H2D=%.0f gemm=%.0f D2H=%.0f f2d=%.0f | total=%.0f ms, transfer+conv=%.0f%%\n",
        n,m,(double)sz,(double)osz,(t1-t0)*1e3,(t2-t1)*1e3,(t3-t2)*1e3,(t4-t3)*1e3,(t5-t4)*1e3,(t5-t0)*1e3,
        100.0*((t1-t0)+(t2-t1)+(t4-t3)+(t5-t4))/((t5-t0)>0?(t5-t0):1));
}


static void pc_call_euclidean_diff(const double* da, const double* db,
                                   double* dc, int n, int m, int m_b) {
    size_t szA = (size_t)n * m, szB = (size_t)n * m_b, osz = (size_t)m * m_b;
    float* hA = pc_host_d2f(da, szA);
    float* hB = pc_host_d2f(db, szB);
    float *d_A, *d_B, *d_D;
    cudaMalloc(&d_A, szA * sizeof(float));
    cudaMalloc(&d_B, szB * sizeof(float));
    cudaMalloc(&d_D, osz * sizeof(float));
    cudaMemcpy(d_A, hA, szA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, hB, szB * sizeof(float), cudaMemcpyHostToDevice);
    cudaFreeHost(hA); cudaFreeHost(hB);
    pc_euclidean_different_blocks_device(d_A, d_B, n, m, m_b, d_D);
    float* outf = pc_pinned_f(osz);
    cudaMemcpy(outf, d_D, osz * sizeof(float), cudaMemcpyDeviceToHost);
    pc_host_f2d(outf, dc, osz);
    cudaFreeHost(outf); cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
}


static void pc_call_cosine_same(const double* da, double* dc, int n, int m,
                                bool center) {
    bool L = pc_rt_log(); double t0=L?omp_get_wtime():0;
    size_t sz = (size_t)n * m, osz = (size_t)m * m;
    float* h = pc_d2f_pin(da, sz);
    double t1=L?omp_get_wtime():0;
    float *d_A, *d_D;
    cudaMalloc(&d_A, sz * sizeof(float));
    cudaMalloc(&d_D, osz * sizeof(float));
    cudaMemcpy(d_A, h, sz * sizeof(float), cudaMemcpyHostToDevice);
    if(L) cudaDeviceSynchronize(); double t2=L?omp_get_wtime():0;
    if (center) pc_center_columns_device(d_A, n, m);
    pc_cosine_same_block_device(d_A, n, m, d_D);
    if(L) cudaDeviceSynchronize(); double t3=L?omp_get_wtime():0;
    float* outf = pc_pin(&g_pin_out, &g_pin_out_sz, osz);
    cudaMemcpy(outf, d_D, osz * sizeof(float), cudaMemcpyDeviceToHost);
    if(L) cudaDeviceSynchronize(); double t4=L?omp_get_wtime():0;
    pc_host_f2d(outf, dc, osz);
    double t5=L?omp_get_wtime():0;
    cudaFree(d_A); cudaFree(d_D);
    if(L) Rprintf("[rt %s n=%d m=%d |W|=%.0e out=%.0e] d2f=%.0f H2D=%.0f gemm=%.0f D2H=%.0f f2d=%.0f | total=%.0f ms, transfer+conv=%.0f%%\n",
        center?"pear":"cos",n,m,(double)sz,(double)osz,(t1-t0)*1e3,(t2-t1)*1e3,(t3-t2)*1e3,(t4-t3)*1e3,(t5-t4)*1e3,(t5-t0)*1e3,
        100.0*((t1-t0)+(t2-t1)+(t4-t3)+(t5-t4))/((t5-t0)>0?(t5-t0):1));
}


static void pc_call_cosine_diff(const double* da, const double* db, double* dc,
                                int n, int m, int m_b, bool center) {
    size_t szA = (size_t)n * m, szB = (size_t)n * m_b, osz = (size_t)m * m_b;
    float* hA = pc_host_d2f(da, szA);
    float* hB = pc_host_d2f(db, szB);
    float *d_A, *d_B, *d_D;
    cudaMalloc(&d_A, szA * sizeof(float));
    cudaMalloc(&d_B, szB * sizeof(float));
    cudaMalloc(&d_D, osz * sizeof(float));
    cudaMemcpy(d_A, hA, szA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, hB, szB * sizeof(float), cudaMemcpyHostToDevice);
    cudaFreeHost(hA); cudaFreeHost(hB);
    if (center) {
        pc_center_columns_device(d_A, n, m);
        pc_center_columns_device(d_B, n, m_b);
    }
    pc_cosine_different_blocks_device(d_A, d_B, n, m, m_b, d_D);
    float* outf = pc_pinned_f(osz);
    cudaMemcpy(outf, d_D, osz * sizeof(float), cudaMemcpyDeviceToHost);
    pc_host_f2d(outf, dc, osz);
    cudaFreeHost(outf); cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
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


static void pc_call_manhattan_same(const double* da, double* dc, int n, int m) {
    size_t sz = (size_t)n * m, osz = (size_t)m * m;
    float* h = pc_host_d2f(da, sz);
    float *d_A, *d_D;
    cudaMalloc(&d_A, sz * sizeof(float));
    cudaMalloc(&d_D, osz * sizeof(float));
    cudaMemcpy(d_A, h, sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaFreeHost(h);
    pc_manhattan_same_block_device(d_A, n, m, d_D);
    float* outf = pc_pinned_f(osz);
    cudaMemcpy(outf, d_D, osz * sizeof(float), cudaMemcpyDeviceToHost);
    pc_host_f2d(outf, dc, osz);
    cudaFreeHost(outf); cudaFree(d_A); cudaFree(d_D);
}


// Spearman = cosine on per-column ranks. Rank on host (matches the existing
// driver), then take the cuBLAS cosine path (was the slow atomic kernel before).
static void pc_call_spearman_same(const double* da, double* dc, int n, int m) {
    size_t sz = (size_t)n * m, osz = (size_t)m * m;
    float* h = pc_host_d2f(da, sz);
    rank_columns(h, n, m);
    float *d_A, *d_D;
    cudaMalloc(&d_A, sz * sizeof(float));
    cudaMalloc(&d_D, osz * sizeof(float));
    cudaMemcpy(d_A, h, sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaFreeHost(h);
    pc_cosine_same_block_device(d_A, n, m, d_D);
    float* outf = pc_pinned_f(osz);
    cudaMemcpy(outf, d_D, osz * sizeof(float), cudaMemcpyDeviceToHost);
    pc_host_f2d(outf, dc, osz);
    cudaFreeHost(outf); cudaFree(d_A); cudaFree(d_D);
}


static void pc_call_kendall_same(const double* da, double* dc, int n, int m) {
    size_t sz = (size_t)n * m, osz = (size_t)m * m;
    float* h = pc_host_d2f(da, sz);
    float* d_A; int* d_disc;
    cudaMalloc(&d_A, sz * sizeof(float));
    cudaMalloc(&d_disc, osz * sizeof(int));
    cudaMemcpy(d_A, h, sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaFreeHost(h);
    pc_kendall_same_block_device(d_A, n, m, d_disc);
    int* hdisc = (int*)malloc(osz * sizeof(int));
    cudaMemcpy(hdisc, d_disc, osz * sizeof(int), cudaMemcpyDeviceToHost);
    double norm = (double)n * (n - 1);
    // Upper triangle holds counts; mirror to lower, zero diagonal (same layout
    // as matrix_Kendall_distance_same_block; symmetric, so reshape-agnostic).
    #pragma omp parallel for schedule(static)
    for (long long row = 0; row < m; ++row)
        for (long long col = 0; col < m; ++col) {
            size_t idx = (size_t)row * m + col;
            if (row < col)      dc[idx] = (double)hdisc[idx] * 2.0 / norm;
            else if (row > col) dc[idx] = (double)hdisc[(size_t)col * m + row] * 2.0 / norm;
            else                dc[idx] = 0.0;
        }
    free(hdisc); cudaFree(d_A); cudaFree(d_disc);
}


// Unified .Call entry: one dense distance matrix (m x m) over a resident float
// copy. metric: 0 euclidean, 1 cosine, 2 pearson, 3 manhattan, 4 spearman,
// 5 kendall. Returns R_NilValue if it won't fit on the GPU (R falls back to the
// block loop).
extern "C" SEXP C_dense_block(SEXP a, SEXP n_, SEXP m_, SEXP metric_) {
    int n = INTEGER(n_)[0], m = INTEGER(m_)[0], metric = INTEGER(metric_)[0];

    size_t need = ((size_t)n * m + (size_t)m * m) * sizeof(float);
    if (metric == 5) {                       // kendall variant A may add scratch
        size_t per = (size_t)4 * (n < 1 ? 1 : n) * sizeof(float);
        if (per > 96 * 1024) need += (size_t)2 * 1024 * 1024 * 1024;
    }
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess ||
        need + (need / 10) > free_b) {
        return R_NilValue;
    }

    const double* da = REAL(a);
    SEXP out = PROTECT(allocVector(REALSXP, (R_xlen_t)m * m));
    double* dc = REAL(out);
    switch (metric) {
        case 0: pc_call_euclidean_same(da, dc, n, m);        break;
        case 1: pc_call_cosine_same(da, dc, n, m, false);    break;
        case 2: pc_call_cosine_same(da, dc, n, m, true);     break;
        case 3: pc_call_manhattan_same(da, dc, n, m);        break;
        case 4: pc_call_spearman_same(da, dc, n, m);         break;
        case 5: pc_call_kendall_same(da, dc, n, m);          break;
        default: UNPROTECT(1); return R_NilValue;
    }
    UNPROTECT(1);
    return out;
}


// ---- Honest C-side batched dense round-trip (euclidean/cosine/pearson) --------
// Loads the full n x m input onto the GPU ONCE (persistent pinned d2f), then tiles
// the OUTPUT in batch x batch blocks: per tile a single cuBLAS GEMM (reusing the
// same validated same/diff device drivers on sub-column pointers) + finalize +
// D2H of just that tile. GPU memory is bounded by O(n*m + batch^2) — the full m x m
// is never resident on device — so it scales past the full-matrix fast path while
// keeping the input upload + double<->float conversions out of the per-tile R loop
// (which is what made the .C block loop ~10x slower). write=0 => discard (benchmark
// round-trip); write=1 => assemble the m x m double output (column-major, mirrored).
extern "C" SEXP C_dense_block_batched(SEXP a, SEXP n_, SEXP m_, SEXP metric_, SEXP batch_, SEXP write_) {
    int n = INTEGER(n_)[0], m = INTEGER(m_)[0], metric = INTEGER(metric_)[0];
    int batch = INTEGER(batch_)[0], write = INTEGER(write_)[0];
    if (metric > 3) return R_NilValue;                 // euclidean/cosine/pearson (GEMM) + manhattan (tiled L1)
    if (batch < 1) batch = m;
    if (batch > m) batch = m;

    // Honest batching: only TWO column-blocks of the input (n x batch each) + one
    // output tile are ever resident -> GPU memory O(n*batch + batch^2), independent
    // of m. Guard against a single column-block + tile not fitting.
    size_t need = ((size_t)2 * n * batch + (size_t)batch * batch) * sizeof(float);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || need + need/10 > free_b)
        return R_NilValue;                             // won't fit -> R block-loop fallback

    bool L = pc_rt_log(); double t0 = L ? omp_get_wtime() : 0;
    // input read directly from the SEXP (double OR integer) -> no R as.double copy
    float *d_Ai, *d_Aj, *d_tile;
    cudaMalloc(&d_Ai, (size_t)n * batch * sizeof(float));
    cudaMalloc(&d_Aj, (size_t)n * batch * sizeof(float));
    cudaMalloc(&d_tile, (size_t)batch * batch * sizeof(float));
    SEXP out = R_NilValue; double* dc = nullptr;
    if (write) { out = PROTECT(allocVector(REALSXP, (R_xlen_t)m * m)); dc = REAL(out); }

    double up_s = 0, gemm_s = 0, d2h_s = 0, f2d_s = 0;
    for (int bi = 0; bi < m; bi += batch) {
        int bis = (bi + batch <= m) ? batch : (m - bi);
        double u0 = L ? omp_get_wtime() : 0;
        float* hAi = pc_srcblk_pin(a, (size_t)n * bi, (size_t)n * bis, &g_pin_in, &g_pin_in_sz);   // upload only this column-block
        cudaMemcpy(d_Ai, hAi, (size_t)n * bis * sizeof(float), cudaMemcpyHostToDevice);
        if (metric == 2) pc_center_columns_device(d_Ai, n, bis);        // pearson: center the block
        if (L) { cudaDeviceSynchronize(); up_s += omp_get_wtime() - u0; }
        for (int bj = bi; bj < m; bj += batch) {
            int bjs = (bj + batch <= m) ? batch : (m - bj);
            float* d_Bj = d_Ai;                                         // diagonal tile reuses d_Ai
            if (bj != bi) {
                double u1 = L ? omp_get_wtime() : 0;
                float* hAj = pc_srcblk_pin(a, (size_t)n * bj, (size_t)n * bjs, &g_pin_in_b, &g_pin_in_b_sz);
                cudaMemcpy(d_Aj, hAj, (size_t)n * bjs * sizeof(float), cudaMemcpyHostToDevice);
                if (metric == 2) pc_center_columns_device(d_Aj, n, bjs);
                if (L) { cudaDeviceSynchronize(); up_s += omp_get_wtime() - u1; }
                d_Bj = d_Aj;
            }
            double s0 = L ? omp_get_wtime() : 0;
            if (bi == bj) {
                if      (metric == 0) pc_euclidean_same_block_device(d_Ai, n, bis, d_tile);
                else if (metric == 3) pc_manhattan_tile_device(d_Ai, d_Ai, n, bis, bis, d_tile);
                else                  pc_cosine_same_block_device   (d_Ai, n, bis, d_tile);
            } else {
                if      (metric == 0) pc_euclidean_different_blocks_device(d_Ai, d_Bj, n, bis, bjs, d_tile);
                else if (metric == 3) pc_manhattan_tile_device(d_Ai, d_Bj, n, bis, bjs, d_tile);
                else                  pc_cosine_different_blocks_device   (d_Ai, d_Bj, n, bis, bjs, d_tile);
            }
            if (L) { cudaDeviceSynchronize(); gemm_s += omp_get_wtime() - s0; }
            double s1 = L ? omp_get_wtime() : 0;
            float* outf = pc_pin(&g_pin_out, &g_pin_out_sz, (size_t)bis * bjs);
            cudaMemcpy(outf, d_tile, (size_t)bis * bjs * sizeof(float), cudaMemcpyDeviceToHost);
            if (L) { cudaDeviceSynchronize(); d2h_s += omp_get_wtime() - s1; }
            if (write) {                               // place tile (col-major bis x bjs) + mirror
                double s2 = L ? omp_get_wtime() : 0;
                #pragma omp parallel for schedule(static)
                for (long long c = 0; c < bjs; ++c)
                    for (int r = 0; r < bis; ++r) {
                        double v = (double)outf[(size_t)c * bis + r];
                        dc[(size_t)(bj + c) * m + (bi + r)] = v;
                        dc[(size_t)(bi + r) * m + (bj + c)] = v;   // symmetric mirror
                    }
                if (L) f2d_s += omp_get_wtime() - s2;
            }
        }
    }
    cudaFree(d_Ai); cudaFree(d_Aj); cudaFree(d_tile);
    if (L) Rprintf("[rt-batched metric=%d n=%d m=%d batch=%d] upload=%.0f gemm=%.0f D2H=%.0f f2d=%.0f | total=%.0f ms (GPU mem O(n*batch+batch^2))\n",
        metric, n, m, batch, up_s*1e3, gemm_s*1e3, d2h_s*1e3, f2d_s*1e3, (omp_get_wtime()-t0)*1e3);
    if (write) { UNPROTECT(1); return out; }
    return ScalarLogical(1);                            // discard mode
}


// PCCsc_to_dense_kernel and PCCsc_block_to_dense_kernel moved to pc_linalg (ADR-0002).

// euclidean/cosine/pearson on a sparse CSC matrix: densify on the GPU, then take
// the fast dense cuBLAS path (pinned download + OpenMP, same code as C_dense_block).
static void pc_call_sparse_densify(const int* ai, const int* ap, const double* ax,
                                   int n, int m, int nnz, double* dc, int metric) {
    bool L = pc_rt_log(); double t0=L?omp_get_wtime():0;
    float* hx = pc_pin(&g_pin_in, &g_pin_in_sz, nnz);       // persistent pinned (was per-call malloc)
    #pragma omp parallel for schedule(static)
    for (long long k = 0; k < (long long)nnz; ++k) hx[k] = (float)ax[k];
    int *d_i, *d_p; float* d_x;
    cudaMalloc(&d_i, (size_t)nnz * sizeof(int));
    cudaMalloc(&d_p, (size_t)(m + 1) * sizeof(int));
    cudaMalloc(&d_x, (size_t)nnz * sizeof(float));
    cudaMemcpy(d_i, ai, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, ap, (size_t)(m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, hx, (size_t)nnz * sizeof(float), cudaMemcpyHostToDevice);
    float* d_A;
    cudaMalloc(&d_A, (size_t)n * m * sizeof(float));
    cudaMemset(d_A, 0, (size_t)n * m * sizeof(float));
    int t = 128, b = (m + t - 1) / t;
    PCCsc_to_dense_kernel<<<b, t>>>(d_i, d_p, d_x, n, m, d_A);
    cudaFree(d_i); cudaFree(d_p); cudaFree(d_x);
    if(L){ cudaDeviceSynchronize(); }
    double t1=L?omp_get_wtime():0;
    if (metric == 2) pc_center_columns_device(d_A, n, m);   // pearson = centered cosine
    float* d_D;
    cudaMalloc(&d_D, (size_t)m * m * sizeof(float));
    if (metric == 0) pc_euclidean_same_block_device(d_A, n, m, d_D);
    else             pc_cosine_same_block_device(d_A, n, m, d_D);
    if(L){ cudaDeviceSynchronize(); }
    double t2=L?omp_get_wtime():0;
    float* outf = pc_pin(&g_pin_out, &g_pin_out_sz, (size_t)m * m);   // persistent pinned
    cudaMemcpy(outf, d_D, (size_t)m * m * sizeof(float), cudaMemcpyDeviceToHost);
    pc_host_f2d(outf, dc, (size_t)m * m);
    cudaFree(d_A); cudaFree(d_D);
    if(L) Rprintf("[rt-sparse-densify m=%d n=%d nnz=%d metric=%d] densify+upload=%.0f gemm=%.0f D2H+f2d=%.0f | total=%.0f ms\n",
        m,n,nnz,metric,(t1-t0)*1e3,(t2-t1)*1e3,(omp_get_wtime()-t2)*1e3,(omp_get_wtime()-t0)*1e3);
}


// ---- Honest C-side batched SPARSE round-trip (euclidean/cosine/pearson) -------
// Sparse analogue of C_dense_block_batched: the CSC arrays are uploaded once
// (nnz-sized, small), then per output tile ONLY the two needed column-blocks are
// densified on the GPU (n x batch each) and fed to the same GEMM device drivers.
// GPU memory is O(nnz + n*batch + batch^2), independent of m -> scales to any m
// without ever densifying the full n x m matrix. write=0 discard, write=1 assemble.
extern "C" SEXP C_sparse_block_batched(SEXP ai, SEXP ap, SEXP ax, SEXP n_, SEXP m_,
                                       SEXP nnz_, SEXP metric_, SEXP batch_, SEXP write_) {
    int n = INTEGER(n_)[0], m = INTEGER(m_)[0], nnz = INTEGER(nnz_)[0];
    int metric = INTEGER(metric_)[0], batch = INTEGER(batch_)[0], write = INTEGER(write_)[0];
    if (metric > 2) return R_NilValue;
    if (batch < 1) batch = m; if (batch > m) batch = m;

    size_t need = (size_t)nnz * (sizeof(int) + sizeof(float)) + (size_t)(m + 1) * sizeof(int)
                + (size_t)2 * n * batch * sizeof(float) + (size_t)batch * batch * sizeof(float);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || need + need/10 > free_b)
        return R_NilValue;

    const int* aidx = INTEGER(ai); const int* apos = INTEGER(ap); const double* axx = REAL(ax);
    float* hx = pc_pin(&g_pin_in, &g_pin_in_sz, nnz);      // CSC values -> float (persistent pinned)
    #pragma omp parallel for schedule(static)
    for (long long k = 0; k < (long long)nnz; ++k) hx[k] = (float)axx[k];
    int *d_i, *d_p; float *d_x;
    cudaMalloc(&d_i, (size_t)nnz * sizeof(int));
    cudaMalloc(&d_p, (size_t)(m + 1) * sizeof(int));
    cudaMalloc(&d_x, (size_t)nnz * sizeof(float));
    cudaMemcpy(d_i, aidx, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, apos, (size_t)(m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, hx,   (size_t)nnz * sizeof(float), cudaMemcpyHostToDevice);
    float *d_Ai, *d_Aj, *d_tile;
    cudaMalloc(&d_Ai, (size_t)n * batch * sizeof(float));
    cudaMalloc(&d_Aj, (size_t)n * batch * sizeof(float));
    cudaMalloc(&d_tile, (size_t)batch * batch * sizeof(float));
    SEXP out = R_NilValue; double* dc = nullptr;
    if (write) { out = PROTECT(allocVector(REALSXP, (R_xlen_t)m * m)); dc = REAL(out); }
    int tb = 128;
    for (int bi = 0; bi < m; bi += batch) {
        int bis = (bi + batch <= m) ? batch : (m - bi);
        cudaMemset(d_Ai, 0, (size_t)n * bis * sizeof(float));
        PCCsc_block_to_dense_kernel<<<(bis + tb - 1)/tb, tb>>>(d_i, d_p, d_x, n, bi, bis, d_Ai);
        if (metric == 2) pc_center_columns_device(d_Ai, n, bis);
        for (int bj = bi; bj < m; bj += batch) {
            int bjs = (bj + batch <= m) ? batch : (m - bj);
            float* d_Bj = d_Ai;
            if (bj != bi) {
                cudaMemset(d_Aj, 0, (size_t)n * bjs * sizeof(float));
                PCCsc_block_to_dense_kernel<<<(bjs + tb - 1)/tb, tb>>>(d_i, d_p, d_x, n, bj, bjs, d_Aj);
                if (metric == 2) pc_center_columns_device(d_Aj, n, bjs);
                d_Bj = d_Aj;
            }
            if (bi == bj) {
                if (metric == 0) pc_euclidean_same_block_device(d_Ai, n, bis, d_tile);
                else             pc_cosine_same_block_device(d_Ai, n, bis, d_tile);
            } else {
                if (metric == 0) pc_euclidean_different_blocks_device(d_Ai, d_Bj, n, bis, bjs, d_tile);
                else             pc_cosine_different_blocks_device(d_Ai, d_Bj, n, bis, bjs, d_tile);
            }
            float* outf = pc_pin(&g_pin_out, &g_pin_out_sz, (size_t)bis * bjs);
            cudaMemcpy(outf, d_tile, (size_t)bis * bjs * sizeof(float), cudaMemcpyDeviceToHost);
            if (write) {
                #pragma omp parallel for schedule(static)
                for (long long c = 0; c < bjs; ++c)
                    for (int r = 0; r < bis; ++r) {
                        double v = (double)outf[(size_t)c * bis + r];
                        dc[(size_t)(bj + c) * m + (bi + r)] = v;
                        dc[(size_t)(bi + r) * m + (bj + c)] = v;
                    }
            }
        }
    }
    cudaFree(d_i); cudaFree(d_p); cudaFree(d_x);
    cudaFree(d_Ai); cudaFree(d_Aj); cudaFree(d_tile);
    if (write) { UNPROTECT(1); return out; }
    return ScalarLogical(1);
}


// ============== .Call fast path: ALL sparse metrics, full matrix =============
// Always fed the full-matrix CSC (Csparse @i/@p/@x). euclidean/cosine/pearson
// densify on the GPU and take the fast dense path (pinned + OpenMP + cuBLAS);
// manhattan/spearman/kendall use the sparsity-aware per_cell_pair drivers (CSC,
// double result on-device). Either way no .C duplication and no per-block
// coercion. metric: 0 euc,1 cos,2 pear,3 manh,4 spear,5 kendall. Returns
// R_NilValue if it won't fit on the GPU (R falls back to the block loop).
typedef void (*sparse_drv_t)(int*, int*, double*, int*, int*, double*, double*,
                             int*, int*, int*, int*, int*);


extern "C" SEXP C_sparse_block(SEXP ai, SEXP ap, SEXP ax, SEXP n_, SEXP m_,
                               SEXP nnz_, SEXP metric_, SEXP layout_) {
    int n = INTEGER(n_)[0], m = INTEGER(m_)[0], nnz = INTEGER(nnz_)[0];
    int metric = INTEGER(metric_)[0];

    // GPU-memory guard (drivers gpuErrchk-exit on OOM, which would kill R).
    size_t need = (size_t)m * m * 16 + (size_t)nnz * 8 + (size_t)(m + 1) * 8;
    if (metric <= 2) need += (size_t)n * m * sizeof(float);       // densify dense
    if (metric == 5) need += (size_t)2 * 1024 * 1024 * 1024;      // kendall scratch
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess ||
        need + (need / 10) > free_b) {
        return R_NilValue;
    }

    int* aidx = INTEGER(ai);
    int* apos = INTEGER(ap);
    double* axx = REAL(ax);
    SEXP out = PROTECT(allocVector(REALSXP, (R_xlen_t)m * m));
    double* res = REAL(out);

    if (metric <= 2) {                       // euclidean / cosine / pearson
        pc_call_sparse_densify(aidx, apos, axx, n, m, nnz, res, metric);
    } else {                                 // manhattan / spearman / kendall (pcp)
        sparse_drv_t fn = NULL;
        switch (metric) {
            case 3: fn = matrix_Manhattan_sparse_per_cell_pair_distance_same_block; break;
            case 4: fn = matrix_Spearman_sparse_per_cell_pair_distance_same_block;  break;
            case 5: fn = matrix_Kendall_sparse_per_cell_pair_distance_same_block;   break;
        }
        if (!fn) { UNPROTECT(1); return R_NilValue; }
        int N = n, M = m, NNZ = nnz;
        fn(aidx, apos, axx, aidx, apos, axx, res, &N, &M, &M, &NNZ, &NNZ);
    }
    UNPROTECT(1);
    return out;
}
