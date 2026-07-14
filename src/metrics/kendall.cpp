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


//=============================================
// O(n log n) discordant count for two dense length-n vectors via Fenwick
// inversion count (Knight 1966). tau-naive: discordant iff (a_i-a_j)*(b_i-b_j) < 0,
// i.e. number of pairs ordered strictly oppositely in a and b. Caller provides
// scratch idx[n], bs[n], fen[n+1] (reused across calls to avoid per-pair alloc).
static long long dense_kendall_disc(const float* a, const float* b, int n,
                                    int* idx, float* bs, int* fen) {
  if (n < 2) return 0;
  for (int i = 0; i < n; ++i) idx[i] = i;
  std::sort(idx, idx + n, [&](int x, int y) {
    if (a[x] != a[y]) return a[x] < a[y];
    return b[x] < b[y];
  });
  for (int i = 0; i < n; ++i) bs[i] = b[i];
  std::sort(bs, bs + n);
  int B = (int)(std::unique(bs, bs + n) - bs);          // distinct b values
  for (int i = 0; i <= B; ++i) fen[i] = 0;
  long long disc = 0;
  int inserted = 0, p = 0;
  while (p < n) {                                       // process equal-a groups
    int q = p;
    while (q < n && a[idx[q]] == a[idx[p]]) ++q;
    for (int t = p; t < q; ++t) {                       // count inserted with b strictly greater
      int r = (int)(std::lower_bound(bs, bs + B, b[idx[t]]) - bs) + 1;
      int le = 0; for (int i = r; i > 0; i -= i & -i) le += fen[i];
      disc += inserted - le;
    }
    for (int t = p; t < q; ++t) {
      int r = (int)(std::lower_bound(bs, bs + B, b[idx[t]]) - bs) + 1;
      for (int i = r; i <= B; i += i & -i) ++fen[i];
      ++inserted;
    }
    p = q;
  }
  return disc;
}


// Kendall_distance_matrix for same block (O(m^2 * n log n) via Fenwick).
extern "C" void matrix_Kendall_distance_same_block_cpu(double * a, double * b, double * c, int * n, int * m, int * m_b) {
  int N = *n, M = *m;
  int array_size = N * M;
  float * d_array = new float[array_size];
  for (int i = 0; i < array_size; ++i) d_array[i] = a[i];

  unsigned int * h_result = new unsigned int[(size_t)M * M];
  std::memset(h_result, 0, (size_t)M * M * sizeof(unsigned int));

  #pragma omp parallel
  {
    std::vector<int> idx(N), fen(N + 1);
    std::vector<float> bs(N);
    #pragma omp for schedule(dynamic)
    for (int col1_num = 0; col1_num < M; ++col1_num) {
      const float * col1 = d_array + (size_t)N * col1_num;
      for (int col2_num = col1_num + 1; col2_num < M; ++col2_num) {
        const float * col2 = d_array + (size_t)N * col2_num;
        long long disc = dense_kendall_disc(col1, col2, N, idx.data(), bs.data(), fen.data());
        h_result[(size_t)col1_num * M + col2_num] = (unsigned int)disc;
        h_result[(size_t)col2_num * M + col1_num] = (unsigned int)disc;
      }
    }
  }

  for (size_t i = 0; i < (size_t)M * M; ++i) c[i] = h_result[i] * 2.0 / N / (N - 1);
  delete[] h_result;
  delete[] d_array;
}


// Kendall_distance_matrix for different blocks (O(m*m_b * n log n) via Fenwick).
extern "C" void  matrix_Kendall_distance_different_blocks_cpu(double* a, double* b, double* c, int* n, int* m, int* m_b){
  int N = *n, M = *m, MB = *m_b;
  float * d_array  = new float[(size_t)N * M];
  for (size_t i = 0; i < (size_t)N * M; ++i)  d_array[i]  = a[i];
  float * d_array2 = new float[(size_t)N * MB];
  for (size_t i = 0; i < (size_t)N * MB; ++i) d_array2[i] = b[i];

  unsigned int * h_result = new unsigned int[(size_t)M * MB];
  std::memset(h_result, 0, (size_t)M * MB * sizeof(unsigned int));

  #pragma omp parallel
  {
    std::vector<int> idx(N), fen(N + 1);
    std::vector<float> bs(N);
    #pragma omp for schedule(dynamic)
    for (int col1_num = 0; col1_num < M; ++col1_num) {
      const float * col1 = d_array + (size_t)N * col1_num;
      for (int col2_num = 0; col2_num < MB; ++col2_num) {
        const float * col2 = d_array2 + (size_t)N * col2_num;
        long long disc = dense_kendall_disc(col1, col2, N, idx.data(), bs.data(), fen.data());
        h_result[(size_t)col2_num * M + col1_num] = (unsigned int)disc;
      }
    }
  }

  for (size_t i = 0; i < (size_t)M * MB; ++i) c[i] = h_result[i] * 2.0 / N / (N - 1);
  delete[] h_result;
  delete[] d_array;
  delete[] d_array2;
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
) {
  int rows = *num_rows;
  int columns = *num_columns;

  float *a_values = new float[*num_elements_a];
  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = static_cast<float>(a_double_values[i]);
  }

  int result_size = columns * columns;
  int nt = get_num_threads(rows);
  std::vector<int*> locals(nt);
  for (int t = 0; t < nt; ++t) {
    locals[t] = new int[result_size];
    std::memset(locals[t], 0, result_size * sizeof(int));
  }

  // Two-pointer merge over CSR rows: for each row pair, merge nonzero
  // columns from both rows, compute diffs, count discordant column pairs.
  parallel_for_with_id(rows, nt, [&](int t, int row_start, int row_end) {
    int* local = locals[t];
    std::vector<int> pos_cols, neg_cols;

    for (int r1 = row_start; r1 < row_end; ++r1) {
      int r1_start = a_positions[r1];
      int r1_end   = a_positions[r1 + 1];

      for (int r2 = r1 + 1; r2 < rows; ++r2) {
        int r2_start = a_positions[r2];
        int r2_end   = a_positions[r2 + 1];

        pos_cols.clear();
        neg_cols.clear();

        // Two-pointer merge of R1 and R2 nonzero columns
        int i = r1_start, j = r2_start;
        while (i < r1_end && j < r2_end) {
          int c1 = a_index[i], c2 = a_index[j];
          if (c1 == c2) {
            float diff = a_values[i] - a_values[j];
            if (diff > 0) pos_cols.push_back(c1);
            else if (diff < 0) neg_cols.push_back(c1);
            ++i; ++j;
          } else if (c1 < c2) {
            // R1 nonzero, R2 zero => diff = a_values[i]
            if (a_values[i] > 0) pos_cols.push_back(c1);
            else if (a_values[i] < 0) neg_cols.push_back(c1);
            ++i;
          } else {
            // R1 zero, R2 nonzero => diff = -a_values[j]
            if (a_values[j] < 0) pos_cols.push_back(c2);
            else if (a_values[j] > 0) neg_cols.push_back(c2);
            ++j;
          }
        }
        while (i < r1_end) {
          if (a_values[i] > 0) pos_cols.push_back(a_index[i]);
          else if (a_values[i] < 0) neg_cols.push_back(a_index[i]);
          ++i;
        }
        while (j < r2_end) {
          if (a_values[j] < 0) pos_cols.push_back(a_index[j]);
          else if (a_values[j] > 0) neg_cols.push_back(a_index[j]);
          ++j;
        }

        // Each (pos, neg) pair is discordant
        for (int p : pos_cols) {
          for (int n : neg_cols) {
            local[p * columns + n] += 1;
            local[n * columns + p] += 1;
          }
        }
      }
    }
  });

  int *disconcordant = new int[result_size];
  std::memset(disconcordant, 0, result_size * sizeof(int));
  for (int t = 0; t < nt; ++t) {
    for (int i = 0; i < result_size; ++i) disconcordant[i] += locals[t][i];
    delete[] locals[t];
  }

  for (int i = 0; i < result_size; ++i) {
    result[i] = static_cast<double>(disconcordant[i]) * 2.0f / rows / (rows - 1);
  }

  delete[] disconcordant;
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
  int result_size = columns * columns_b;

  float *a_values = new float[*num_elements_a];
  float *b_values = new float[*num_elements_b];

  for (int i = 0; i < *num_elements_a; ++i) {
    a_values[i] = static_cast<float>(a_double_values[i]);
  }

  for (int i = 0; i < *num_elements_b; ++i) {
    b_values[i] = static_cast<float>(b_double_values[i]);
  }

  int nt = get_num_threads(rows);
  std::vector<int*> locals(nt);
  for (int t = 0; t < nt; ++t) {
    locals[t] = new int[result_size];
    std::memset(locals[t], 0, result_size * sizeof(int));
  }

  // Two-pointer merge: for each row pair, compute diffs for block A columns
  // and block B columns separately, then count cross-discordant pairs.
  parallel_for_with_id(rows, nt, [&](int tid, int row_start, int row_end) {
    int* local = locals[tid];
    // Diffs for block A columns (indexed 0..columns-1)
    std::vector<int> a_pos_cols, a_neg_cols;
    // Diffs for block B columns (indexed 0..columns_b-1)
    std::vector<int> b_pos_cols, b_neg_cols;

    for (int r1 = row_start; r1 < row_end; ++r1) {
      int r1a_start = a_positions[r1], r1a_end = a_positions[r1 + 1];
      int r1b_start = b_positions[r1], r1b_end = b_positions[r1 + 1];

      for (int r2 = r1 + 1; r2 < rows; ++r2) {
        int r2a_start = a_positions[r2], r2a_end = a_positions[r2 + 1];
        int r2b_start = b_positions[r2], r2b_end = b_positions[r2 + 1];

        // Merge block A columns for rows r1 and r2
        a_pos_cols.clear(); a_neg_cols.clear();
        {
          int i = r1a_start, j = r2a_start;
          while (i < r1a_end && j < r2a_end) {
            int c1 = a_index[i], c2 = a_index[j];
            if (c1 == c2) {
              float diff = a_values[i] - a_values[j];
              if (diff > 0) a_pos_cols.push_back(c1);
              else if (diff < 0) a_neg_cols.push_back(c1);
              ++i; ++j;
            } else if (c1 < c2) {
              if (a_values[i] > 0) a_pos_cols.push_back(c1);
              else if (a_values[i] < 0) a_neg_cols.push_back(c1);
              ++i;
            } else {
              if (a_values[j] < 0) a_pos_cols.push_back(c2);
              else if (a_values[j] > 0) a_neg_cols.push_back(c2);
              ++j;
            }
          }
          while (i < r1a_end) {
            if (a_values[i] > 0) a_pos_cols.push_back(a_index[i]);
            else if (a_values[i] < 0) a_neg_cols.push_back(a_index[i]);
            ++i;
          }
          while (j < r2a_end) {
            if (a_values[j] < 0) a_pos_cols.push_back(a_index[j]);
            else if (a_values[j] > 0) a_neg_cols.push_back(a_index[j]);
            ++j;
          }
        }

        // Merge block B columns for rows r1 and r2
        b_pos_cols.clear(); b_neg_cols.clear();
        {
          int i = r1b_start, j = r2b_start;
          while (i < r1b_end && j < r2b_end) {
            int c1 = b_index[i], c2 = b_index[j];
            if (c1 == c2) {
              float diff = b_values[i] - b_values[j];
              if (diff > 0) b_pos_cols.push_back(c1);
              else if (diff < 0) b_neg_cols.push_back(c1);
              ++i; ++j;
            } else if (c1 < c2) {
              if (b_values[i] > 0) b_pos_cols.push_back(c1);
              else if (b_values[i] < 0) b_neg_cols.push_back(c1);
              ++i;
            } else {
              if (b_values[j] < 0) b_pos_cols.push_back(c2);
              else if (b_values[j] > 0) b_neg_cols.push_back(c2);
              ++j;
            }
          }
          while (i < r1b_end) {
            if (b_values[i] > 0) b_pos_cols.push_back(b_index[i]);
            else if (b_values[i] < 0) b_neg_cols.push_back(b_index[i]);
            ++i;
          }
          while (j < r2b_end) {
            if (b_values[j] < 0) b_pos_cols.push_back(b_index[j]);
            else if (b_values[j] > 0) b_neg_cols.push_back(b_index[j]);
            ++j;
          }
        }

        // Cross-discordant: A_pos × B_neg and A_neg × B_pos
        // result layout: result[col_b * columns + col_a]
        for (int ac : a_pos_cols) {
          for (int bc : b_neg_cols) {
            local[bc * columns + ac] += 1;
          }
        }
        for (int ac : a_neg_cols) {
          for (int bc : b_pos_cols) {
            local[bc * columns + ac] += 1;
          }
        }
      }
    }
  });

  int *disconcordant = new int[result_size];
  std::memset(disconcordant, 0, result_size * sizeof(int));
  for (int t = 0; t < nt; ++t) {
    for (int i = 0; i < result_size; ++i) disconcordant[i] += locals[t][i];
    delete[] locals[t];
  }

  for (int i = 0; i < result_size; ++i) {
    result[i] = static_cast<double>(disconcordant[i]) * 2.0f / rows / (rows - 1);
  }

  delete[] disconcordant;
  delete[] a_values;
  delete[] b_values;
}


// ==================== Per-cell-pair sparse Kendall ====================
//
// Alternative sparse Kendall using CSC layout (CsparseMatrix from R).
// 1 thread per (cell_a, cell_b) pair with double two-pointer merge.
// No atomics, no sweep-line, signed-correct via n_signflip * n_inactive.
// See plans/PER_CELL_PAIR.md for algorithm details.

// Shared helper: compute discordant count for one cell pair via double merge.
// Takes two separate CSC arrays (a = cell_a, b = cell_b). For same_block,
// both point into the same underlying CSC.
static long long kendall_per_cell_pair_merge(
    const int* a_csc_i, const float* a_csc_x, int ia, int ea,
    const int* b_csc_i, const float* b_csc_x, int ib, int eb,
    int n_genes)
{
  // O(k log k) discordant count via Fenwick inversion count (Knight's algorithm),
  // k = nnz of the two cells' union. Replaces the former O(k^2) double-merge.
  // Active-pair discordances are strict inversions in (a asc, b desc); the
  // all-zero block uses the closed form n_signflip * n_inactive.
  // tau-naive convention: discordant iff (a_i-a_j)*(b_i-b_j) < 0.

  // Merge the union of nonzero genes into value arrays (av, bv).
  std::vector<float> av, bv;
  av.reserve((ea - ia) + (eb - ib));
  bv.reserve((ea - ia) + (eb - ib));
  int n_signflip = 0;

  int oia = ia, oib = ib;
  while (oia < ea || oib < eb) {
    float a_i, b_i;
    if (oia < ea && (oib >= eb || a_csc_i[oia] < b_csc_i[oib])) {
      a_i = a_csc_x[oia]; b_i = 0.0f; ++oia;
    } else if (oib < eb && (oia >= ea || b_csc_i[oib] < a_csc_i[oia])) {
      a_i = 0.0f; b_i = b_csc_x[oib]; ++oib;
    } else {
      a_i = a_csc_x[oia]; b_i = b_csc_x[oib]; ++oia; ++oib;
    }
    av.push_back(a_i); bv.push_back(b_i);
    if (a_i * b_i < 0.0f) ++n_signflip;
  }

  int k = (int)av.size();
  long long n_inactive = (long long)n_genes - k;
  long long discordant = (long long)n_signflip * n_inactive;

  if (k >= 2) {
    // Order by a asc, then b asc; the b tie-break keeps equal-a groups from
    // counting as inversions.
    std::vector<int> idx(k);
    for (int i = 0; i < k; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int x, int y) {
      if (av[x] != av[y]) return av[x] < av[y];
      return bv[x] < bv[y];
    });

    std::vector<float> bs(bv);
    std::sort(bs.begin(), bs.end());
    bs.erase(std::unique(bs.begin(), bs.end()), bs.end());
    int B = (int)bs.size();

    std::vector<int> fen(B + 1, 0);
    auto upd = [&](int p) { for (; p <= B; p += p & -p) ++fen[p]; };
    auto qry = [&](int p) { int s = 0; for (; p > 0; p -= p & -p) s += fen[p]; return s; };
    auto rankb = [&](float v) {
      return int(std::lower_bound(bs.begin(), bs.end(), v) - bs.begin()) + 1;
    };

    // Process in equal-a groups: count discordances against already-inserted
    // points (strictly smaller a), then insert the group.
    int inserted = 0, p = 0;
    while (p < k) {
      int q = p;
      while (q < k && av[idx[q]] == av[idx[p]]) ++q;
      for (int t = p; t < q; ++t) {
        int r = rankb(bv[idx[t]]);
        discordant += (long long)(inserted - qry(r)); // inserted points with strictly larger b
      }
      for (int t = p; t < q; ++t) { upd(rankb(bv[idx[t]])); ++inserted; }
      p = q;
    }
  }

  return discordant;
}


extern "C" void matrix_Kendall_sparse_per_cell_pair_distance_same_block_cpu(
    int* csc_i,
    int* csc_p,
    double* csc_x_double,
    int* /*b_index*/,
    int* /*b_positions*/,
    double* /*b_values*/,
    double* result,
    int* num_rows,
    int* num_columns,
    int* /*num_columns_b*/,
    int* num_elements_a,
    int* /*num_elements_b*/)
{
  int n_genes = *num_rows;
  int n_cells = *num_columns;
  int nnz = *num_elements_a;

  float* csc_x = new float[nnz];
  for (int k = 0; k < nnz; ++k) csc_x[k] = static_cast<float>(csc_x_double[k]);

  double norm = (double)n_genes * (n_genes - 1);

  #pragma omp parallel for schedule(dynamic)
  for (int cell_a = 0; cell_a < n_cells; ++cell_a) {
    for (int cell_b = cell_a + 1; cell_b < n_cells; ++cell_b) {
      long long disc = kendall_per_cell_pair_merge(
          csc_i, csc_x, csc_p[cell_a], csc_p[cell_a + 1],
          csc_i, csc_x, csc_p[cell_b], csc_p[cell_b + 1],
          n_genes);
      double d = (double)disc * 2.0 / norm;
      result[cell_a * n_cells + cell_b] = d;
      result[cell_b * n_cells + cell_a] = d;
    }
    result[cell_a * n_cells + cell_a] = 0.0;
  }

  delete[] csc_x;
}


extern "C" void matrix_Kendall_sparse_per_cell_pair_distance_different_blocks_cpu(
    int* a_csc_i,
    int* a_csc_p,
    double* a_csc_x_double,
    int* b_csc_i,
    int* b_csc_p,
    double* b_csc_x_double,
    double* result,
    int* num_rows,
    int* num_columns,
    int* num_columns_b,
    int* num_elements_a,
    int* num_elements_b)
{
  int n_genes = *num_rows;
  int n_cells_a = *num_columns;
  int n_cells_b = *num_columns_b;
  int nnz_a = *num_elements_a;
  int nnz_b = *num_elements_b;

  float* a_csc_x = new float[nnz_a];
  float* b_csc_x = new float[nnz_b];
  for (int k = 0; k < nnz_a; ++k) a_csc_x[k] = static_cast<float>(a_csc_x_double[k]);
  for (int k = 0; k < nnz_b; ++k) b_csc_x[k] = static_cast<float>(b_csc_x_double[k]);

  double norm = (double)n_genes * (n_genes - 1);

  #pragma omp parallel for schedule(dynamic)
  for (int cell_a = 0; cell_a < n_cells_a; ++cell_a) {
    for (int cell_b = 0; cell_b < n_cells_b; ++cell_b) {
      long long disc = kendall_per_cell_pair_merge(
          a_csc_i, a_csc_x,
          a_csc_p[cell_a], a_csc_p[cell_a + 1],
          b_csc_i, b_csc_x,
          b_csc_p[cell_b], b_csc_p[cell_b + 1],
          n_genes);
      double d = (double)disc * 2.0 / norm;
      result[cell_b * n_cells_a + cell_a] = d;
    }
  }

  delete[] a_csc_x;
  delete[] b_csc_x;
}
