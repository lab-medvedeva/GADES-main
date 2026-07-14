#ifndef PC_LINALG_CUH
#define PC_LINALG_CUH

// ============================================================================
// pc_linalg — metric-agnostic numeric utilities shared by the metric paths:
// per-column reductions, centering, gram normalization, CSR/CSC->dense scatter,
// and Spearman rank transforms. These are NOT runtime state (see pc_runtime).
//
// Kernels are `static __global__` and host helpers `static`, so every TU that
// includes this header gets its own inlined copy — no cross-TU device symbols,
// hence NO -rdc=true / device linking.
// ============================================================================

#include <cstddef>
#include <vector>
#include <algorithm>
#include <utility>
#include <cuda_runtime.h>

// ---- per-column reductions -------------------------------------------------
static __global__ void PCCol_sqnorm_kernel(const float* A, int n, int m, float* norms) {
    extern __shared__ float sdata_sq[];
    int col = blockIdx.x;
    if (col >= m) return;
    int tid = threadIdx.x;
    const float* c = A + (size_t)n * col;
    float acc = 0.0f;
    for (int r = tid; r < n; r += blockDim.x) {
        float v = c[r]; acc += v * v;
    }
    sdata_sq[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata_sq[tid] += sdata_sq[tid + s];
        __syncthreads();
    }
    if (tid == 0) norms[col] = sqrtf(sdata_sq[0]);
}

static __global__ void PCCol_sum_kernel(const float* A, int n, int m, float* sums) {
    extern __shared__ float sdata_sum[];
    int col = blockIdx.x;
    if (col >= m) return;
    int tid = threadIdx.x;
    const float* c = A + (size_t)n * col;
    float acc = 0.0f;
    for (int r = tid; r < n; r += blockDim.x) acc += c[r];
    sdata_sum[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata_sum[tid] += sdata_sum[tid + s];
        __syncthreads();
    }
    if (tid == 0) sums[col] = sdata_sum[0];
}

// Squared per-column norms (no sqrt) — used by Euclidean finalize where we
// need ||col||² directly.
static __global__ void PCCol_sq_kernel(const float* A, int n, int m, float* sq) {
    extern __shared__ float sdata_raw[];
    int col = blockIdx.x;
    if (col >= m) return;
    int tid = threadIdx.x;
    const float* c = A + (size_t)n * col;
    float acc = 0.0f;
    for (int r = tid; r < n; r += blockDim.x) {
        float v = c[r]; acc += v * v;
    }
    sdata_raw[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata_raw[tid] += sdata_raw[tid + s];
        __syncthreads();
    }
    if (tid == 0) sq[col] = sdata_raw[0];
}

// ---- centering -------------------------------------------------------------
static __global__ void PCCenter_columns_kernel(float* A, int n, int m, const float* sums) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y;
    if (r >= n || c >= m) return;
    A[r + (size_t)n * c] -= sums[c] / (float)n;
}

// ---- gram normalization (cosine / pearson) ---------------------------------
// Same-block: D is col-major m x m (symmetric).
static __global__ void PCNormalize_gram_same_kernel(int m, float* G, const float* norms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= m) return;
    size_t idx = (size_t)i + (size_t)j * m;
    if (i == j) G[idx] = 0.0f;
    else        G[idx] = 1.0f - G[idx] / (norms[i] * norms[j]);
}

// Different-blocks: D is col-major m x m_b, x_norms[m], y_norms[m_b].
static __global__ void PCNormalize_gram_xy_kernel(int m, int m_b, float* G,
                                                  const float* x_norms,
                                                  const float* y_norms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= m_b) return;
    size_t idx = (size_t)i + (size_t)j * m;
    G[idx] = 1.0f - G[idx] / (x_norms[i] * y_norms[j]);
}

// Center columns of d_A in place (n x m).
static void pc_center_columns_device(float* d_A, int n, int m) {
    float* d_sums;
    cudaMalloc(&d_sums, m * sizeof(float));
    int t = 256;
    PCCol_sum_kernel<<<m, t, t * sizeof(float)>>>(d_A, n, m, d_sums);
    dim3 tb(128, 1);
    dim3 gb((n + 127) / 128, m);
    PCCenter_columns_kernel<<<gb, tb>>>(d_A, n, m, d_sums);
    cudaFree(d_sums);
}

// ---- CSR / CSC -> dense scatter --------------------------------------------
// Scatter CSR (row-major) into col-major dense d_A (n x m). d_A must be
// pre-zeroed. One thread per row of A; loops over that row's nonzeros.
static __global__ void PCCsr_to_dense_kernel(const int* d_idx, const int* d_ptr,
                                             const float* d_val, int n, int m,
                                             float* d_A) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    int start = d_ptr[row];
    int end = d_ptr[row + 1];
    for (int j = start; j < end; ++j) {
        int col = d_idx[j];
        d_A[(size_t)col * n + row] = d_val[j];
    }
}

// Densify CSR (host-side arrays) into a freshly-allocated device buffer.
// Caller owns the returned d_A and must cudaFree it. h_val stays on host.
static float* pc_sparse_to_dense_device(const int* a_index,
                                        const int* a_positions,
                                        const double* a_values, int n, int m,
                                        int nnz) {
    std::vector<float> val_f(nnz);
    for (int i = 0; i < nnz; ++i) val_f[i] = (float)a_values[i];

    int* d_idx;
    int* d_ptr;
    float* d_val;
    cudaMalloc(&d_idx, nnz * sizeof(int));
    cudaMalloc(&d_ptr, (n + 1) * sizeof(int));
    cudaMalloc(&d_val, nnz * sizeof(float));
    cudaMemcpy(d_idx, a_index,     nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr, a_positions, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val_f.data(),nnz * sizeof(float), cudaMemcpyHostToDevice);

    float* d_A;
    cudaMalloc(&d_A, (size_t)n * m * sizeof(float));
    cudaMemset(d_A, 0, (size_t)n * m * sizeof(float));

    int t = 128;
    int b = (n + t - 1) / t;
    PCCsr_to_dense_kernel<<<b, t>>>(d_idx, d_ptr, d_val, n, m, d_A);

    cudaFree(d_idx); cudaFree(d_ptr); cudaFree(d_val);
    return d_A;
}

// CSC -> column-major dense float scatter (one thread per cell column). d_A must
// be pre-zeroed. CSC: d_p[col]..d_p[col+1] index d_i (gene rows) / d_x (values).
static __global__ void PCCsc_to_dense_kernel(const int* d_i, const int* d_p,
                                             const float* d_x, int n, int m, float* d_A) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= m) return;
    int start = d_p[col], end = d_p[col + 1];
    for (int j = start; j < end; ++j) d_A[(size_t)col * n + d_i[j]] = d_x[j];
}
// Densify ONLY a column range [col_start, col_start+col_count) of a CSC into a
// dense n x col_count block (col-major). d_block must be pre-zeroed. Used by the
// honest sparse batched path: only the column-blocks of the current tile are ever
// densified -> GPU memory stays O(nnz + n*batch + batch^2), independent of m.
static __global__ void PCCsc_block_to_dense_kernel(const int* d_i, const int* d_p,
                                                   const float* d_x, int n,
                                                   int col_start, int col_count, float* d_block) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= col_count) return;
    int col = col_start + c;
    int start = d_p[col], end = d_p[col + 1];
    for (int j = start; j < end; ++j) d_block[(size_t)c * n + d_i[j]] = d_x[j];
}

// ---- Spearman rank transforms ----------------------------------------------
static void rank_columns(float* array, int n, int m) {
    float mean_rank = (n + 1) / 2.0f;
    // Columns rank independently -> parallelize across them; each thread keeps
    // its own scratch (a column-shared buffer would race). This is the spearman
    // host bottleneck (one sort of n elements per column).
    #pragma omp parallel
    {
        std::vector<std::pair<float, int>> col_data(n);
        #pragma omp for schedule(dynamic)
        for (int j = 0; j < m; ++j) {
            float* col = array + (size_t)j * n;
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

#endif // PC_LINALG_CUH
