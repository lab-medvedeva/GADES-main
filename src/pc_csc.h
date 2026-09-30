#ifndef PC_CSC_H
#define PC_CSC_H

// CSC layout contract, shared by the GPU and CPU backends (no CUDA here).

#include <vector>
#include <algorithm>
#ifdef PC_NO_R
#include <cstdio>
#define REprintf(...) fprintf(stderr, __VA_ARGS__)
#else
#include <R.h>
#endif

// ---- CSC contract check ----------------------------------------------------
//
// Every sparsity-aware kernel assumes the row indices inside each column are
// sorted and unique: the rank metrics merge two columns' nonzero lists, and the
// distance metrics account for the zero gaps *between* consecutive nonzeros.
// Neither is meaningful otherwise, and the failure is silent -- Kendall returns
// values above 1 for a metric bounded by 1, and nothing crashes.
//
// The invariant belongs here, at the boundary that requires it, rather than in
// each language wrapper: R's Matrix canonicalises on construction and so never
// trips it, but SciPy does not re-sort after fancy row indexing, and any future
// caller starts out unaware. Checking is one linear pass over nnz -- negligible
// beside an O(m^2 k log k) kernel -- so check always and repair only when the
// check fails (see pc_csc_canonical).
//
// Returns the offending column, or -1 when the layout is valid.
inline long long pc_csc_first_unsorted_column(const int* col_ptr, const int* row_idx,
                                              int n_cols)
{
    for (int c = 0; c < n_cols; ++c) {
        for (int k = col_ptr[c] + 1; k < col_ptr[c + 1]; ++k) {
            if (row_idx[k] <= row_idx[k - 1]) return (long long)c;
        }
    }
    return -1;
}

// A usable view of a CSC matrix: the caller's arrays when they are already
// canonical, a repaired copy when they are not.
//
// Repair rather than refuse, because this ships as an end-to-end package: a
// caller who hands over a matrix straight out of fancy indexing should get the
// right answer, not homework. The common path pays only the linear check; the
// sort runs solely when it is needed, and says so, because silently reordering
// someone's data is how the original bug stayed hidden.
//
// The caller's arrays are never modified in place -- they may be R's own SEXP
// storage or a NumPy buffer.
struct PcCscView {
    const int* row_idx;
    const double* values;
    bool repaired;
    std::vector<int> idx_owned;
    std::vector<double> val_owned;
};

inline PcCscView pc_csc_canonical(const int* col_ptr, const int* row_idx,
                                  const double* values, int n_cols, int nnz,
                                  const char* where)
{
    PcCscView view;
    view.row_idx = row_idx;
    view.values = values;
    view.repaired = false;

    if (pc_csc_first_unsorted_column(col_ptr, row_idx, n_cols) < 0) return view;

    view.idx_owned.assign(row_idx, row_idx + nnz);
    view.val_owned.assign(values, values + nnz);

    std::vector<int> order;
    long long duplicates = 0;
    for (int c = 0; c < n_cols; ++c) {
        int lo = col_ptr[c], hi = col_ptr[c + 1], len = hi - lo;
        if (len < 2) continue;
        order.resize(len);
        for (int k = 0; k < len; ++k) order[k] = k;
        const int* src = row_idx + lo;
        std::sort(order.begin(), order.end(),
                  [src](int a, int b) { return src[a] < src[b]; });
        for (int k = 0; k < len; ++k) {
            view.idx_owned[lo + k] = row_idx[lo + order[k]];
            view.val_owned[lo + k] = values[lo + order[k]];
            if (k && view.idx_owned[lo + k] == view.idx_owned[lo + k - 1]) ++duplicates;
        }
    }

    view.row_idx = view.idx_owned.data();
    view.values = view.val_owned.data();
    view.repaired = true;

    REprintf("%s: CSC row indices were not sorted within columns; sorted a "
             "private copy (%d nonzeros).\n"
             "  The sparse kernels walk columns in index order, so this would "
             "otherwise have produced silently wrong distances.\n"
             "  Canonicalise upstream to avoid the copy "
             "(R: as(x, \"CsparseMatrix\"); SciPy: x.sum_duplicates()).\n",
             where, nnz);
    if (duplicates) {
        REprintf("  WARNING: %lld duplicate row indices remain; sorting cannot "
                 "merge them and the result will still be wrong. Call "
                 "sum_duplicates() upstream.\n", duplicates);
    }
    return view;
}


#endif // PC_CSC_H
