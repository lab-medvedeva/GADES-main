// Python-facing C API for the CPU backend.
// Forward-declares the extern "C" functions from main.cpp and provides
// a clean dispatch layer callable via ctypes.

extern "C" {

// Dense distance drivers (main.cpp)
void matrix_Euclidean_distance_same_block_cpu(double*, double*, double*, int*, int*, int*);
void matrix_Euclidean_distance_different_blocks_cpu(double*, double*, double*, int*, int*, int*);
void matrix_Cosine_distance_same_block_cpu(double*, double*, double*, int*, int*, int*);
void matrix_Cosine_distance_different_blocks_cpu(double*, double*, double*, int*, int*, int*);
void matrix_Pearson_distance_same_block_cpu(double*, double*, double*, int*, int*, int*);
void matrix_Pearson_distance_different_blocks_cpu(double*, double*, double*, int*, int*, int*);
void matrix_Manhattan_distance_same_block_cpu(double*, double*, double*, int*, int*, int*);
void matrix_Manhattan_distance_different_blocks_cpu(double*, double*, double*, int*, int*, int*);
void matrix_Spearman_distance_same_block_cpu(double*, double*, double*, int*, int*, int*);
void matrix_Spearman_distance_different_blocks_cpu(double*, double*, double*, int*, int*, int*);
void matrix_Kendall_distance_same_block_cpu(double*, double*, double*, int*, int*, int*);
void matrix_Kendall_distance_different_blocks_cpu(double*, double*, double*, int*, int*, int*);

// Sparse per_cell_pair drivers (main.cpp) — CSC format
void matrix_Euclidean_sparse_per_cell_pair_distance_same_block_cpu(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);
void matrix_Euclidean_sparse_per_cell_pair_distance_different_blocks_cpu(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);
void matrix_Cosine_sparse_per_cell_pair_distance_same_block_cpu(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);
void matrix_Cosine_sparse_per_cell_pair_distance_different_blocks_cpu(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);
void matrix_Pearson_sparse_per_cell_pair_distance_same_block_cpu(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);
void matrix_Pearson_sparse_per_cell_pair_distance_different_blocks_cpu(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);
void matrix_Manhattan_sparse_per_cell_pair_distance_same_block_cpu(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);
void matrix_Manhattan_sparse_per_cell_pair_distance_different_blocks_cpu(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);
void matrix_Spearman_sparse_per_cell_pair_distance_same_block_cpu(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);
void matrix_Spearman_sparse_per_cell_pair_distance_different_blocks_cpu(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);
void matrix_Kendall_sparse_per_cell_pair_distance_same_block_cpu(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);
void matrix_Kendall_sparse_per_cell_pair_distance_different_blocks_cpu(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);

} // extern "C" forward declarations


// metric codes: 0=euclidean 1=cosine 2=pearson 3=manhattan 4=spearman 5=kendall

typedef void (*dense_fn_t)(double*, double*, double*, int*, int*, int*);
typedef void (*sparse_fn_t)(int*, int*, double*, int*, int*, double*,
                            double*, int*, int*, int*, int*, int*);

static dense_fn_t dense_same_table[] = {
    matrix_Euclidean_distance_same_block_cpu,
    matrix_Cosine_distance_same_block_cpu,
    matrix_Pearson_distance_same_block_cpu,
    matrix_Manhattan_distance_same_block_cpu,
    matrix_Spearman_distance_same_block_cpu,
    matrix_Kendall_distance_same_block_cpu,
};

static dense_fn_t dense_diff_table[] = {
    matrix_Euclidean_distance_different_blocks_cpu,
    matrix_Cosine_distance_different_blocks_cpu,
    matrix_Pearson_distance_different_blocks_cpu,
    matrix_Manhattan_distance_different_blocks_cpu,
    matrix_Spearman_distance_different_blocks_cpu,
    matrix_Kendall_distance_different_blocks_cpu,
};

static sparse_fn_t sparse_same_table[] = {
    matrix_Euclidean_sparse_per_cell_pair_distance_same_block_cpu,
    matrix_Cosine_sparse_per_cell_pair_distance_same_block_cpu,
    matrix_Pearson_sparse_per_cell_pair_distance_same_block_cpu,
    matrix_Manhattan_sparse_per_cell_pair_distance_same_block_cpu,
    matrix_Spearman_sparse_per_cell_pair_distance_same_block_cpu,
    matrix_Kendall_sparse_per_cell_pair_distance_same_block_cpu,
};

static sparse_fn_t sparse_diff_table[] = {
    matrix_Euclidean_sparse_per_cell_pair_distance_different_blocks_cpu,
    matrix_Cosine_sparse_per_cell_pair_distance_different_blocks_cpu,
    matrix_Pearson_sparse_per_cell_pair_distance_different_blocks_cpu,
    matrix_Manhattan_sparse_per_cell_pair_distance_different_blocks_cpu,
    matrix_Spearman_sparse_per_cell_pair_distance_different_blocks_cpu,
    matrix_Kendall_sparse_per_cell_pair_distance_different_blocks_cpu,
};


extern "C" {

int gades_dense_cpu(double* a, double* out, int n, int m, int metric) {
    if (metric < 0 || metric > 5) return -1;
    int N = n, M = m;
    dense_same_table[metric](a, a, out, &N, &M, &M);
    return 0;
}

int gades_dense_pairwise_cpu(double* a, double* b, double* out,
                                 int n, int m_a, int m_b, int metric) {
    if (metric < 0 || metric > 5) return -1;
    int N = n, MA = m_a, MB = m_b;
    dense_diff_table[metric](a, b, out, &N, &MA, &MB);
    return 0;
}

int gades_sparse_cpu(int* indices, int* indptr, double* data,
                         double* out, int n, int m, int nnz, int metric) {
    if (metric < 0 || metric > 5) return -1;
    int N = n, M = m, NNZ = nnz;
    sparse_same_table[metric](indices, indptr, data,
                              indices, indptr, data,
                              out, &N, &M, &M, &NNZ, &NNZ);
    return 0;
}

int gades_sparse_pairwise_cpu(int* a_i, int* a_p, double* a_x,
                                  int* b_i, int* b_p, double* b_x,
                                  double* out, int n, int m_a, int m_b,
                                  int nnz_a, int nnz_b, int metric) {
    if (metric < 0 || metric > 5) return -1;
    int N = n, MA = m_a, MB = m_b, NNZA = nnz_a, NNZB = nnz_b;
    sparse_diff_table[metric](a_i, a_p, a_x,
                              b_i, b_p, b_x,
                              out, &N, &MA, &MB, &NNZA, &NNZB);
    return 0;
}

} // extern "C"
