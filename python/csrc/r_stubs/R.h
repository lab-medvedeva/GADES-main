#ifndef GADES_R_STUB_H
#define GADES_R_STUB_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define Rprintf printf
#define REprintf(...) fprintf(stderr, __VA_ARGS__)
#define error(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); abort(); } while(0)
#define warning(...) fprintf(stderr, __VA_ARGS__)

#define R_PosInf INFINITY
#define R_NegInf (-INFINITY)
#define R_NaN NAN

#endif
