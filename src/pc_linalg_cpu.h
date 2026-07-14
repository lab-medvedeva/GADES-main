#ifndef PC_LINALG_CPU_H
#define PC_LINALG_CPU_H

// ============================================================================
// pc_linalg_cpu — CPU-side metric-agnostic numeric utilities (CPU analog of
// pc_linalg): per-column norms, centering, CSR->dense densify, and Spearman
// rank transforms. Shared across the cosine/euclidean/pearson/spearman CPU
// drivers. Not runtime policy (see pc_runtime_cpu).
// ============================================================================

#include <cstddef>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>

// ---- per-column reductions / centering -------------------------------------
static void pc_col_sqnorms(const float* A, int n, int m, float* norms) {
    #pragma omp parallel for
    for (int c = 0; c < m; ++c) {
        const float* col = A + (size_t)n * c;
        float s = 0.0f;
        for (int r = 0; r < n; ++r) s += col[r] * col[r];
        norms[c] = std::sqrt(s);
    }
}

static void pc_center_columns(float* A, int n, int m) {
    #pragma omp parallel for
    for (int c = 0; c < m; ++c) {
        float* col = A + (size_t)n * c;
        float s = 0.0f;
        for (int r = 0; r < n; ++r) s += col[r];
        float mean = s / (float)n;
        for (int r = 0; r < n; ++r) col[r] -= mean;
    }
}

// Squared per-column norms (no sqrt).
static void pc_col_sq(const float* A, int n, int m, float* sq) {
    #pragma omp parallel for
    for (int c = 0; c < m; ++c) {
        const float* col = A + (size_t)n * c;
        float s = 0.0f;
        for (int r = 0; r < n; ++r) s += col[r] * col[r];
        sq[c] = s;
    }
}

// ---- CSR -> dense densify ---------------------------------------------------
// Densify CSR (host arrays) into a fresh float* buffer (n x m col-major).
// Caller owns the returned pointer and must delete[] it.
static float* pc_sparse_to_dense_cpu(const int* a_index,
                                     const int* a_positions,
                                     const double* a_values, int n, int m,
                                     int nnz) {
    float* dense = new float[(size_t)n * m];
    std::memset(dense, 0, (size_t)n * m * sizeof(float));
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < n; ++row) {
        int start = a_positions[row];
        int end = a_positions[row + 1];
        for (int j = start; j < end; ++j) {
            int col = a_index[j];
            dense[(size_t)col * n + row] = (float)a_values[j];
        }
    }
    return dense;
}

// ---- Spearman rank transforms ----------------------------------------------
// Rank columns and center (subtract mean rank = (n+1)/2) so that the existing
// Cosine logic (cosine-style correlation) produces Pearson-on-ranks = Spearman.
static void rank_columns_cpu(float* array, int n, int m) {
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
// average-tie ranks (rank - mean_rank) so Pearson logic gives Spearman.
// Returns a dense column-major ranked array (rows x columns).
static float* csr_to_ranked_dense_cpu(int* index, int* positions, double* values, int rows, int columns) {
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

#endif // PC_LINALG_CPU_H
