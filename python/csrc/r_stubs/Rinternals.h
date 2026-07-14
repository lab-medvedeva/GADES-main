#ifndef GADES_RINTERNALS_STUB_H
#define GADES_RINTERNALS_STUB_H

#include <stdlib.h>
#include <stdint.h>

typedef void* SEXP;

#define REALSXP    14
#define INTSXP     13
#define LGLSXP     10

typedef long R_xlen_t;

#define R_NilValue ((SEXP)0)

static inline double* REAL(SEXP x)    { return (double*)x; }
static inline int*    INTEGER(SEXP x) { return (int*)x; }
static inline int*    LOGICAL(SEXP x) { return (int*)x; }
static inline int     TYPEOF(SEXP x)  { (void)x; return REALSXP; }
static inline R_xlen_t XLENGTH(SEXP x) { (void)x; return 0; }

static inline SEXP PROTECT(SEXP x)    { return x; }
static inline void UNPROTECT(int n)   { (void)n; }

static inline SEXP allocVector(int type, R_xlen_t n) {
    size_t sz = (type == REALSXP) ? sizeof(double) : sizeof(int);
    return (SEXP)calloc((size_t)n, sz);
}

static inline SEXP ScalarLogical(int val) {
    int* p = (int*)malloc(sizeof(int));
    if (p) *p = val;
    return (SEXP)p;
}

static inline SEXP ScalarReal(double val) {
    double* p = (double*)malloc(sizeof(double));
    if (p) *p = val;
    return (SEXP)p;
}

#endif
