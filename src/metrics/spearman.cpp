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



// ==================== Spearman ====================

// rank_columns_cpu moved to pc_linalg_cpu.

// csr_to_ranked_dense_cpu moved to pc_linalg_cpu.

extern "C" void matrix_Spearman_distance_same_block_cpu(
    double* a, double* b, double* c, int* n, int* m, int* m_b
) {
    int array_size = *n * *m;
    float* array_new = new float[array_size];
    for (int i = 0; i < array_size; ++i) {
        array_new[i] = static_cast<float>(a[i]);
    }
    rank_columns_cpu(array_new, *n, *m);

    float* h_scalar = new float[(*m) * (*m)];
    std::memset(h_scalar, 0, (*m) * (*m) * sizeof(float));
    float* h_prod1 = new float[(*m) * (*m)];
    std::memset(h_prod1, 0, (*m) * (*m) * sizeof(float));
    float* h_prod2 = new float[(*m) * (*m)];
    std::memset(h_prod2, 0, (*m) * (*m) * sizeof(float));

    #pragma omp parallel for schedule(dynamic)
    for (int col1_num = 0; col1_num < *m; ++col1_num) {
        for (int col2_num = col1_num; col2_num < *m; ++col2_num) {
            float* col1 = array_new + *n * col1_num;
            float* col2 = array_new + *n * col2_num;
            float scalar = 0.0f, p1 = 0.0f, p2 = 0.0f;
            for (int row = 0; row < *n; ++row) {
                scalar += col1[row] * col2[row];
                p1 += col1[row] * col1[row];
                p2 += col2[row] * col2[row];
            }
            h_scalar[col1_num * *m + col2_num] = scalar;
            h_prod1[col1_num * *m + col2_num] = p1;
            h_prod2[col1_num * *m + col2_num] = p2;
            h_scalar[col2_num * *m + col1_num] = scalar;
            h_prod1[col2_num * *m + col1_num] = p1;
            h_prod2[col2_num * *m + col1_num] = p2;
        }
    }

    for (int i = 0; i < (*m) * (*m); ++i) {
        c[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);
    }

    free(h_prod1);
    free(h_prod2);
    free(h_scalar);
    free(array_new);
}


extern "C" void matrix_Spearman_distance_different_blocks_cpu(
    double* a, double* b, double* c, int* n, int* m, int* m_b
) {
    int array_size = *n * *m;
    float* array_new = new float[array_size];
    for (int i = 0; i < array_size; ++i) {
        array_new[i] = static_cast<float>(a[i]);
    }
    rank_columns_cpu(array_new, *n, *m);

    int array2_size = *n * (*m_b);
    float* array2_new = new float[array2_size];
    for (int i = 0; i < array2_size; ++i) {
        array2_new[i] = static_cast<float>(b[i]);
    }
    rank_columns_cpu(array2_new, *n, *m_b);

    float* d_array = new float[array_size];
    float* d_array2 = new float[array2_size];
    std::memcpy(d_array, array_new, array_size * sizeof(float));
    std::memcpy(d_array2, array2_new, array2_size * sizeof(float));

    float* h_scalar = new float[(*m) * (*m_b)];
    std::memset(h_scalar, 0, (*m) * (*m_b) * sizeof(float));
    float* h_prod1 = new float[(*m) * (*m_b)];
    std::memset(h_prod1, 0, (*m) * (*m_b) * sizeof(float));
    float* h_prod2 = new float[(*m) * (*m_b)];
    std::memset(h_prod2, 0, (*m) * (*m_b) * sizeof(float));

    parallel_for_cols(*m, [&](int col1_start, int col1_end) {
        for (int row = 0; row < *n; row++) {
            for (int col1_num = col1_start; col1_num < col1_end; ++col1_num) {
                float* col1 = d_array + *n * col1_num;
                for (int col2_num = 0; col2_num < *m_b; ++col2_num) {
                    float* col2 = d_array2 + *n * col2_num;
                    float num = col1[row] * col2[row];
                    float sum1 = col1[row] * col1[row];
                    float sum2 = col2[row] * col2[row];
                    h_scalar[col2_num * *m + col1_num] += num;
                    h_prod1[col2_num * *m + col1_num] += sum1;
                    h_prod2[col2_num * *m + col1_num] += sum2;
                }
            }
        }
    });

    for (int i = 0; i < (*m) * (*m_b); ++i) {
        c[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);
    }

    delete[] h_prod1;
    delete[] h_prod2;
    delete[] h_scalar;
    delete[] d_array;
    delete[] d_array2;
    delete[] array_new;
    delete[] array2_new;
}


extern "C" void matrix_Spearman_sparse_distance_same_block_cpu(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
    int rows = *num_rows;
    int columns = *num_columns;

    float* dense = csr_to_ranked_dense_cpu(a_index, a_positions, a_double_values, rows, columns);

    float* h_scalar = new float[columns * columns];
    std::memset(h_scalar, 0, columns * columns * sizeof(float));
    float* h_prod1 = new float[columns * columns];
    std::memset(h_prod1, 0, columns * columns * sizeof(float));
    float* h_prod2 = new float[columns * columns];
    std::memset(h_prod2, 0, columns * columns * sizeof(float));

    #pragma omp parallel for schedule(dynamic)
    for (int col1_num = 0; col1_num < columns; ++col1_num) {
        for (int col2_num = col1_num; col2_num < columns; ++col2_num) {
            float* col1 = dense + rows * col1_num;
            float* col2 = dense + rows * col2_num;
            float scalar = 0.0f, p1 = 0.0f, p2 = 0.0f;
            for (int row = 0; row < rows; ++row) {
                scalar += col1[row] * col2[row];
                p1 += col1[row] * col1[row];
                p2 += col2[row] * col2[row];
            }
            h_scalar[col1_num * columns + col2_num] = scalar;
            h_prod1[col1_num * columns + col2_num] = p1;
            h_prod2[col1_num * columns + col2_num] = p2;
            h_scalar[col2_num * columns + col1_num] = scalar;
            h_prod1[col2_num * columns + col1_num] = p1;
            h_prod2[col2_num * columns + col1_num] = p2;
        }
    }

    for (int i = 0; i < columns * columns; ++i) {
        result[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);
    }

    free(h_prod1);
    free(h_prod2);
    free(h_scalar);
    delete[] dense;
}


extern "C" void matrix_Spearman_sparse_distance_different_blocks_cpu(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b
) {
    int rows = *num_rows;
    int columns = *num_columns;
    int columns_b = *num_columns_b;

    float* dense_a = csr_to_ranked_dense_cpu(a_index, a_positions, a_double_values, rows, columns);
    float* dense_b = csr_to_ranked_dense_cpu(b_index, b_positions, b_double_values, rows, columns_b);

    float* h_scalar = new float[columns * columns_b];
    std::memset(h_scalar, 0, columns * columns_b * sizeof(float));
    float* h_prod1 = new float[columns * columns_b];
    std::memset(h_prod1, 0, columns * columns_b * sizeof(float));
    float* h_prod2 = new float[columns * columns_b];
    std::memset(h_prod2, 0, columns * columns_b * sizeof(float));

    parallel_for_cols(columns, [&](int col1_start, int col1_end) {
        for (int row = 0; row < rows; row++) {
            for (int col1_num = col1_start; col1_num < col1_end; ++col1_num) {
                float* col1 = dense_a + rows * col1_num;
                for (int col2_num = 0; col2_num < columns_b; ++col2_num) {
                    float* col2 = dense_b + rows * col2_num;
                    float num = col1[row] * col2[row];
                    float sum1 = col1[row] * col1[row];
                    float sum2 = col2[row] * col2[row];
                    h_scalar[col2_num * columns + col1_num] += num;
                    h_prod1[col2_num * columns + col1_num] += sum1;
                    h_prod2[col2_num * columns + col1_num] += sum2;
                }
            }
        }
    });

    for (int i = 0; i < columns * columns_b; ++i) {
        result[i] = 1.0 - h_scalar[i] / sqrtf(h_prod1[i]) / sqrtf(h_prod2[i]);
    }

    delete[] h_prod1;
    delete[] h_prod2;
    delete[] h_scalar;
    delete[] dense_a;
    delete[] dense_b;
}


// ─── Spearman per_cell_pair ───
//
// Rank-transform every cell once (centered ranks), then reduce to a Cosine-
// style dot product on shifted ranks. Stored value v_a[g] is
// (centered_rank_a[g] - zr_a) for nnz positions; zr_a is the centered rank
// the implicit/explicit zeros of cell a share. The merge over nnz∩nnz yields
//     dot_v_I = Σ_{g∈I} v_a[g]·v_b[g]
// and the full centered-rank dot is
//     dot = dot_v_I - n_genes · zr_a · zr_b
// (cross-terms over A_only/B_only collapse because v is 0 outside nnz).
// Per-cell norm_sq = Σ_nnz c[g]² + (n_genes - nnz)·zr² (over ALL positions).

static void spearman_per_cell_pair_preprocess(
    const int* csc_p, const int* /*csc_i*/, const double* csc_x_in,
    int n_genes, int n_cells,
    float* v_out, float* zr_out, float* norm_sq_out)
{
    float mean_rank = (n_genes + 1) / 2.0f;
    #pragma omp parallel
    {
        std::vector<std::pair<float, int>> entries;
        #pragma omp for schedule(dynamic)
        for (int c = 0; c < n_cells; ++c) {
            int start = csc_p[c];
            int end   = csc_p[c + 1];
            int nnz   = end - start;

            entries.resize(nnz);
            for (int k = 0; k < nnz; ++k) {
                entries[k] = { static_cast<float>(csc_x_in[start + k]), k };
            }
            std::sort(entries.begin(), entries.end());

            int neg_count = 0, explicit_zero_count = 0;
            for (int k = 0; k < nnz; ++k) {
                if (entries[k].first < 0.0f) neg_count++;
                else if (entries[k].first == 0.0f) explicit_zero_count++;
            }
            int total_zeros = (n_genes - nnz) + explicit_zero_count;

            float zr = 0.0f;
            if (total_zeros > 0) {
                zr = (neg_count + 1 + neg_count + total_zeros) / 2.0f - mean_rank;
            }
            zr_out[c] = zr;

            float active_sq = 0.0f;

            // Negative entries: sorted positions 0..neg_count-1
            int i = 0;
            int global_pos = 0;
            while (i < neg_count) {
                int eq = i + 1;
                while (eq < neg_count && entries[eq].first == entries[i].first) ++eq;
                float rank = (global_pos + 1 + global_pos + (eq - i)) / 2.0f - mean_rank;
                float vshift = rank - zr;
                for (int k = i; k < eq; ++k) {
                    v_out[start + entries[k].second] = vshift;
                    active_sq += rank * rank;
                }
                global_pos += (eq - i);
                i = eq;
            }

            // Explicit zeros — rank == zr → v = 0
            for (int k = neg_count; k < neg_count + explicit_zero_count; ++k) {
                v_out[start + entries[k].second] = 0.0f;
                active_sq += zr * zr;
            }

            // Positive entries
            global_pos = neg_count + total_zeros;
            i = neg_count + explicit_zero_count;
            while (i < nnz) {
                int eq = i + 1;
                while (eq < nnz && entries[eq].first == entries[i].first) ++eq;
                float rank = (global_pos + 1 + global_pos + (eq - i)) / 2.0f - mean_rank;
                float vshift = rank - zr;
                for (int k = i; k < eq; ++k) {
                    v_out[start + entries[k].second] = vshift;
                    active_sq += rank * rank;
                }
                global_pos += (eq - i);
                i = eq;
            }

            int implicit_zeros = n_genes - nnz;
            norm_sq_out[c] = active_sq + (float)implicit_zeros * zr * zr;
        }
    }
}


static float spearman_per_cell_pair_merge(
    const int* a_i, const float* a_v, int ia, int ea, float zr_a, float norm_sq_a,
    const int* b_i, const float* b_v, int ib, int eb, float zr_b, float norm_sq_b,
    int n_genes)
{
    float dot_v_I = 0.0f;
    while (ia < ea && ib < eb) {
        if      (a_i[ia] < b_i[ib]) ++ia;
        else if (b_i[ib] < a_i[ia]) ++ib;
        else { dot_v_I += a_v[ia] * b_v[ib]; ++ia; ++ib; }
    }
    float dot = dot_v_I - (float)n_genes * zr_a * zr_b;
    return 1.0f - dot / sqrtf(norm_sq_a * norm_sq_b);
}


extern "C" void matrix_Spearman_sparse_per_cell_pair_distance_same_block_cpu(
    int* csc_i, int* csc_p, double* csc_x_double,
    int* /*b*/, int* /*b*/, double* /*b*/,
    double* result, int* num_rows, int* num_columns,
    int* /*num_columns_b*/, int* num_elements_a, int* /*num_elements_b*/)
{
    int n_genes = *num_rows;
    int n_cells = *num_columns;
    int nnz = *num_elements_a;

    float* v = new float[nnz];
    float* zr = new float[n_cells];
    float* norm_sq = new float[n_cells];

    spearman_per_cell_pair_preprocess(csc_p, csc_i, csc_x_double,
                                       n_genes, n_cells, v, zr, norm_sq);

    #pragma omp parallel for schedule(dynamic)
    for (int ca = 0; ca < n_cells; ++ca) {
        for (int cb = ca + 1; cb < n_cells; ++cb) {
            float d = spearman_per_cell_pair_merge(
                csc_i, v, csc_p[ca], csc_p[ca + 1], zr[ca], norm_sq[ca],
                csc_i, v, csc_p[cb], csc_p[cb + 1], zr[cb], norm_sq[cb],
                n_genes);
            result[ca * n_cells + cb] = d;
            result[cb * n_cells + ca] = d;
        }
        result[ca * n_cells + ca] = 0.0;
    }

    delete[] v;
    delete[] zr;
    delete[] norm_sq;
}


extern "C" void matrix_Spearman_sparse_per_cell_pair_distance_different_blocks_cpu(
    int* a_csc_i, int* a_csc_p, double* a_xd,
    int* b_csc_i, int* b_csc_p, double* b_xd,
    double* result, int* num_rows, int* num_columns,
    int* num_columns_b, int* num_elements_a, int* num_elements_b)
{
    int n_genes = *num_rows;
    int n_cells_a = *num_columns;
    int n_cells_b = *num_columns_b;
    int nnz_a = *num_elements_a;
    int nnz_b = *num_elements_b;

    float* a_v = new float[nnz_a];
    float* a_zr = new float[n_cells_a];
    float* a_nsq = new float[n_cells_a];
    float* b_v = new float[nnz_b];
    float* b_zr = new float[n_cells_b];
    float* b_nsq = new float[n_cells_b];

    spearman_per_cell_pair_preprocess(a_csc_p, a_csc_i, a_xd,
                                       n_genes, n_cells_a, a_v, a_zr, a_nsq);
    spearman_per_cell_pair_preprocess(b_csc_p, b_csc_i, b_xd,
                                       n_genes, n_cells_b, b_v, b_zr, b_nsq);

    #pragma omp parallel for schedule(dynamic)
    for (int ca = 0; ca < n_cells_a; ++ca) {
        for (int cb = 0; cb < n_cells_b; ++cb) {
            float d = spearman_per_cell_pair_merge(
                a_csc_i, a_v, a_csc_p[ca], a_csc_p[ca + 1], a_zr[ca], a_nsq[ca],
                b_csc_i, b_v, b_csc_p[cb], b_csc_p[cb + 1], b_zr[cb], b_nsq[cb],
                n_genes);
            result[cb * n_cells_a + ca] = d;
        }
    }

    delete[] a_v;  delete[] a_zr;  delete[] a_nsq;
    delete[] b_v;  delete[] b_zr;  delete[] b_nsq;
}
