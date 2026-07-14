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
#include "pcp_dispatch.cuh"   // host-only pcp launch-config policy




// ===== Dense Kendall via O(n log n) inversion count (Knight), 1 thread / cell pair =====
// tau-naive: discordant iff (a_i-a_j)*(b_i-b_j) < 0 = strict inversions in (a asc, b desc).
__device__ inline void dk_sort_ab(float* va, float* vb, float* ta, float* tb, int k) {
  float* sa = va; float* sb = vb; float* da = ta; float* db = tb;
  for (int w = 1; w < k; w <<= 1) {
    for (int lo = 0; lo < k; lo += 2 * w) {
      int mid = min(lo + w, k), hi = min(lo + 2 * w, k), i = lo, j = mid, o = lo;
      while (i < mid && j < hi) {
        bool tl = (sa[i] < sa[j]) || (sa[i] == sa[j] && sb[i] <= sb[j]);
        if (tl) { da[o] = sa[i]; db[o] = sb[i]; ++i; } else { da[o] = sa[j]; db[o] = sb[j]; ++j; }
        ++o;
      }
      while (i < mid) { da[o] = sa[i]; db[o] = sb[i]; ++i; ++o; }
      while (j < hi)  { da[o] = sa[j]; db[o] = sb[j]; ++j; ++o; }
    }
    float* t; t = sa; sa = da; da = t; t = sb; sb = db; db = t;
  }
  if (sa != va) for (int i = 0; i < k; ++i) { va[i] = sa[i]; vb[i] = sb[i]; }
}

__device__ inline long long dk_count_inv(float* x, float* tmp, int k) {
  long long inv = 0; float* s = x; float* d = tmp;
  for (int w = 1; w < k; w <<= 1) {
    for (int lo = 0; lo < k; lo += 2 * w) {
      int mid = min(lo + w, k), hi = min(lo + 2 * w, k), i = lo, j = mid, o = lo;
      while (i < mid && j < hi) {
        if (s[i] <= s[j]) d[o++] = s[i++];
        else { d[o++] = s[j++]; inv += (mid - i); }
      }
      while (i < mid) d[o++] = s[i++];
      while (j < hi)  d[o++] = s[j++];
    }
    float* t = s; s = d; d = t;
  }
  return inv;
}

// scratch holds 4 float arrays of size cap (>= n): va, vb, ta, tb.
__device__ inline long long dk_disc(const float* c1, const float* c2, int n, float* scratch, int cap) {
  if (n < 2) return 0;
  float* va = scratch; float* vb = scratch + cap; float* ta = scratch + 2 * cap;
  for (int i = 0; i < n; ++i) { va[i] = c1[i]; vb[i] = c2[i]; }
  dk_sort_ab(va, vb, ta, scratch + 3 * cap, n);
  return dk_count_inv(vb, ta, n);   // vb in a-order; ta as temp
}

__global__ void Rkendall_dense_pcp_same_block(
    const float* __restrict__ array, int n, int m, int cap,
    float* scratch_arena, int* __restrict__ disc_out) {
  long long P = (long long)m * m;
  int tid = blockIdx.x * blockDim.x + threadIdx.x, nthreads = gridDim.x * blockDim.x;
  float* my = scratch_arena + (size_t)tid * 4 * cap;
  for (long long idx = tid; idx < P; idx += nthreads) {
    int a = (int)(idx / m), b = (int)(idx % m);
    if (a >= b) continue;
    disc_out[a * m + b] = (int)dk_disc(array + (size_t)n * a, array + (size_t)n * b, n, my, cap);
  }
}

__global__ void Rkendall_dense_pcp_different_blocks(
    const float* __restrict__ A, const float* __restrict__ B, int n, int m, int m_b, int cap,
    float* scratch_arena, int* __restrict__ disc_out) {
  long long P = (long long)m * m_b;
  int tid = blockIdx.x * blockDim.x + threadIdx.x, nthreads = gridDim.x * blockDim.x;
  float* my = scratch_arena + (size_t)tid * 4 * cap;
  for (long long idx = tid; idx < P; idx += nthreads) {
    int a = (int)(idx / m_b), b = (int)(idx % m_b);
    disc_out[(size_t)b * m + a] = (int)dk_disc(A + (size_t)n * a, B + (size_t)n * b, n, my, cap);
  }
}


// ===== Dense Kendall variant B: 1 warp / cell pair, cooperative merge sort in shared mem =====
__device__ inline void dk_warp_sort_ab(float* a0, float* b0, float* a1, float* b1, int k, int lane) {
  float *sa = a0, *sb = b0, *da = a1, *db = b1;
  for (int w = 1; w < k; w <<= 1) {
    int step = 2 * w, nmerges = (k + step - 1) / step;
    for (int mm = lane; mm < nmerges; mm += 32) {
      int lo = mm * step, mid = min(lo + w, k), hi = min(lo + step, k), i = lo, j = mid, o = lo;
      while (i < mid && j < hi) {
        bool tl = (sa[i] < sa[j]) || (sa[i] == sa[j] && sb[i] <= sb[j]);
        if (tl) { da[o] = sa[i]; db[o] = sb[i]; ++i; } else { da[o] = sa[j]; db[o] = sb[j]; ++j; }
        ++o;
      }
      while (i < mid) { da[o] = sa[i]; db[o] = sb[i]; ++i; ++o; }
      while (j < hi)  { da[o] = sa[j]; db[o] = sb[j]; ++j; ++o; }
    }
    __syncwarp();
    float* t; t = sa; sa = da; da = t; t = sb; sb = db; db = t;
  }
  if (sa != a0) { for (int t = lane; t < k; t += 32) { a0[t] = sa[t]; b0[t] = sb[t]; } __syncwarp(); }
}

__device__ inline long long dk_warp_count_inv(float* x0, float* x1, int k, int lane) {
  float *s = x0, *d = x1; long long inv = 0;
  for (int w = 1; w < k; w <<= 1) {
    int step = 2 * w, nmerges = (k + step - 1) / step;
    for (int mm = lane; mm < nmerges; mm += 32) {
      int lo = mm * step, mid = min(lo + w, k), hi = min(lo + step, k), i = lo, j = mid, o = lo;
      while (i < mid && j < hi) { if (s[i] <= s[j]) d[o++] = s[i++]; else { d[o++] = s[j++]; inv += (mid - i); } }
      while (i < mid) d[o++] = s[i++];
      while (j < hi)  d[o++] = s[j++];
    }
    __syncwarp();
    float* t = s; s = d; d = t;
  }
  unsigned long long u = (unsigned long long)inv;
  for (int off = 16; off > 0; off >>= 1) {
    unsigned int ulo = __shfl_down_sync(0xffffffff, (unsigned int)(u & 0xffffffffu), off);
    unsigned int uhi = __shfl_down_sync(0xffffffff, (unsigned int)(u >> 32), off);
    u += ((unsigned long long)uhi << 32) | ulo;
  }
  return (long long)u; // valid on lane 0
}

__global__ void Rkendall_dense_warpB_same_block(
    const float* __restrict__ array, int n, int m, int cap, int* __restrict__ disc_out) {
  extern __shared__ float smemB[];
  int lane = threadIdx.x, wib = threadIdx.y;
  float* sw = smemB + (size_t)wib * 4 * cap;
  float* sa = sw; float* sb = sw + cap; float* ta = sw + 2 * cap; float* tb = sw + 3 * cap;
  long long P = (long long)m * m;
  long long gw = (long long)blockIdx.x * blockDim.y + wib, stride = (long long)gridDim.x * blockDim.y;
  for (long long idx = gw; idx < P; idx += stride) {
    int a = (int)(idx / m), b = (int)(idx % m);
    if (a >= b) continue;
    const float* c1 = array + (size_t)n * a; const float* c2 = array + (size_t)n * b;
    for (int i = lane; i < n; i += 32) { sa[i] = c1[i]; sb[i] = c2[i]; }
    __syncwarp();
    long long disc = 0;
    if (n >= 2) { dk_warp_sort_ab(sa, sb, ta, tb, n, lane); disc = dk_warp_count_inv(sb, ta, n, lane); }
    if (lane == 0) disc_out[a * m + b] = (int)disc;
    __syncwarp();
  }
}

__global__ void Rkendall_dense_warpB_different_blocks(
    const float* __restrict__ A, const float* __restrict__ B, int n, int m, int m_b, int cap, int* __restrict__ disc_out) {
  extern __shared__ float smemB[];
  int lane = threadIdx.x, wib = threadIdx.y;
  float* sw = smemB + (size_t)wib * 4 * cap;
  float* sa = sw; float* sb = sw + cap; float* ta = sw + 2 * cap; float* tb = sw + 3 * cap;
  long long P = (long long)m * m_b;
  long long gw = (long long)blockIdx.x * blockDim.y + wib, stride = (long long)gridDim.x * blockDim.y;
  for (long long idx = gw; idx < P; idx += stride) {
    int a = (int)(idx / m_b), b = (int)(idx % m_b);
    const float* c1 = A + (size_t)n * a; const float* c2 = B + (size_t)n * b;
    for (int i = lane; i < n; i += 32) { sa[i] = c1[i]; sb[i] = c2[i]; }
    __syncwarp();
    long long disc = 0;
    if (n >= 2) { dk_warp_sort_ab(sa, sb, ta, tb, n, lane); disc = dk_warp_count_inv(sb, ta, n, lane); }
    if (lane == 0) disc_out[(size_t)b * m + a] = (int)disc;
    __syncwarp();
  }
}


// ===== Dense Kendall variant G: 1 warp / pair, cooperative merge in GLOBAL scratch =====
// Same cooperative sort/inversion as variant B, but the per-warp sa/sb/ta/tb live in a
// global-scratch slab (4*cap floats) instead of shared. Used when n is too large to fit
// shared even for one warp -- replaces the old 1-thread-per-pair variant A so the heavy
// per-pair O(n log n) work is shared across 32 lanes instead of serialized on one.
__global__ void Rkendall_dense_warpG_same_block(
    const float* __restrict__ array, int n, int m, int cap,
    float* gscratch, int* __restrict__ disc_out) {
  int lane = threadIdx.x, wib = threadIdx.y, W = blockDim.y;
  long long gwid = (long long)blockIdx.x * W + wib;
  float* sw = gscratch + gwid * 4 * (size_t)cap;
  float* sa = sw; float* sb = sw + cap; float* ta = sw + 2 * cap; float* tb = sw + 3 * cap;
  long long P = (long long)m * m;
  long long stride = (long long)gridDim.x * W;
  for (long long idx = gwid; idx < P; idx += stride) {
    int a = (int)(idx / m), b = (int)(idx % m);
    if (a >= b) continue;
    const float* c1 = array + (size_t)n * a; const float* c2 = array + (size_t)n * b;
    for (int i = lane; i < n; i += 32) { sa[i] = c1[i]; sb[i] = c2[i]; }
    __syncwarp();
    long long disc = 0;
    if (n >= 2) { dk_warp_sort_ab(sa, sb, ta, tb, n, lane); disc = dk_warp_count_inv(sb, ta, n, lane); }
    if (lane == 0) disc_out[a * m + b] = (int)disc;
    __syncwarp();
  }
}

__global__ void Rkendall_dense_warpG_different_blocks(
    const float* __restrict__ A, const float* __restrict__ B, int n, int m, int m_b, int cap,
    float* gscratch, int* __restrict__ disc_out) {
  int lane = threadIdx.x, wib = threadIdx.y, W = blockDim.y;
  long long gwid = (long long)blockIdx.x * W + wib;
  float* sw = gscratch + gwid * 4 * (size_t)cap;
  float* sa = sw; float* sb = sw + cap; float* ta = sw + 2 * cap; float* tb = sw + 3 * cap;
  long long P = (long long)m * m_b;
  long long stride = (long long)gridDim.x * W;
  for (long long idx = gwid; idx < P; idx += stride) {
    int a = (int)(idx / m_b), b = (int)(idx % m_b);
    const float* c1 = A + (size_t)n * a; const float* c2 = B + (size_t)n * b;
    for (int i = lane; i < n; i += 32) { sa[i] = c1[i]; sb[i] = c2[i]; }
    __syncwarp();
    long long disc = 0;
    if (n >= 2) { dk_warp_sort_ab(sa, sb, ta, tb, n, lane); disc = dk_warp_count_inv(sb, ta, n, lane); }
    if (lane == 0) disc_out[(size_t)b * m + a] = (int)disc;
    __syncwarp();
  }
}


// Device opt-in shared-memory budget (cached), NOT hardcoded 96 KiB. Same query the
// sparse per-pair dispatch uses; declared here so the dense kendall drivers (above the
// sparse helper) can call it.
static size_t dk_smem_budget() {
  static size_t cached = 0;
  if (cached) return cached;
  int dev = 0; cudaGetDevice(&dev);
  int v = 0;
  cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (v <= 0) cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlock, dev);
  if (v <= 0) v = 48 * 1024;
  cached = (size_t)v;
  return cached;
}


// Shared dispatch for dense Kendall (used by the .C drivers and the .Call fast path).
// cap = n is uniform over all pairs (no poisoning), so the choice is global: if one
// warp's 4*n buffer fits the device budget -> variant B (shared, up to 16 warps/block);
// else variant G (warp-cooperative GLOBAL scratch, 16 warps) instead of 1-thread/pair.
// Caller has already zeroed d_disc.
static int dk_warps_target() {           // variant B (shared): occupancy-bound
  int t = 16; const char* ew = getenv("HOBO_PCP_WARPS");
  if (ew && ew[0]) { int v = atoi(ew); if (v >= 1 && v <= 32) t = v; }
  return t;
}

// variant G (warp-cooperative GLOBAL merge) is GLOBAL-bandwidth-bound, not
// occupancy-bound: measured W=1 beats W=16 by ~8% (more warps/block just contend
// for L2/global bandwidth; the grid is already scratch-saturated). So default 1
// warp/block here, separate from the shared-path target. Env HOBO_DK_WARPS_G.
static int dk_warps_g_target() {
  int t = 1; const char* ew = getenv("HOBO_DK_WARPS_G");
  if (ew && ew[0]) { int v = atoi(ew); if (v >= 1 && v <= 32) t = v; }
  return t;
}

static void dk_run_same(const float* d_array, int N, int M, int* d_disc) {
  int cap = N < 1 ? 1 : N;
  size_t per = (size_t)4 * cap * sizeof(float);
  long long P = (long long)M * M;
  size_t budget = dk_smem_budget();
  size_t usable = budget > 2048 ? budget - 2048 : budget / 2;
  if (per <= usable) {                                   // variant B: warp + shared
    int Wmax = (int)(usable / per); if (Wmax < 1) Wmax = 1;
    int W = dk_warps_target(); if (W > Wmax) W = Wmax; if (W < 1) W = 1;
    size_t smem = (size_t)W * per;
    cudaFuncSetAttribute(Rkendall_dense_warpB_same_block, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    int blocks = (int)((P + W - 1) / W); if (blocks > 65535) blocks = 65535; if (blocks < 1) blocks = 1;
    dim3 threads(32, W);
    Rkendall_dense_warpB_same_block<<<blocks, threads, smem>>>((float*)d_array, N, M, cap, d_disc);
  } else if (getenv("HOBO_DK_LEGACY")) {                 // legacy variant A: 1 thread/pair (for A/B bench)
    long long want = (long long)(((size_t)2 * 1024 * 1024 * 1024) / per);
    if (want > P) want = P; if (want < 128) want = 128;
    int nb = (int)((want + 127) / 128); if (nb > 65535) nb = 65535;
    float* d_scratch; cudaMalloc(&d_scratch, (size_t)nb * 128 * 4 * cap * sizeof(float));
    Rkendall_dense_pcp_same_block<<<nb, 128>>>((float*)d_array, N, M, cap, d_scratch, d_disc);
    cudaDeviceSynchronize(); cudaFree(d_scratch);
  } else {                                               // variant G: warp-cooperative global
    int W = dk_warps_g_target();
    const size_t sbudget = (size_t)2 * 1024 * 1024 * 1024;
    size_t per_warp_g = (size_t)4 * cap * sizeof(float);
    long long want_warps = (long long)(sbudget / per_warp_g); if (want_warps < W) want_warps = W;
    long long want_blocks = (want_warps + W - 1) / W;
    long long cover = (P + W - 1) / W;
    long long blocks = want_blocks < cover ? want_blocks : cover;
    if (blocks > 65535) blocks = 65535; if (blocks < 1) blocks = 1;
    float* d_scratch; cudaMalloc(&d_scratch, (size_t)blocks * W * 4 * cap * sizeof(float));
    dim3 threads(32, W);
    Rkendall_dense_warpG_same_block<<<(int)blocks, threads>>>((float*)d_array, N, M, cap, d_scratch, d_disc);
    cudaDeviceSynchronize();
    cudaFree(d_scratch);
  }
  gpuErrchk(cudaPeekAtLastError());
}

static void dk_run_diff(const float* dA, const float* dB, int N, int M, int MB, int* d_disc) {
  int cap = N < 1 ? 1 : N;
  size_t per = (size_t)4 * cap * sizeof(float);
  long long P = (long long)M * MB;
  size_t budget = dk_smem_budget();
  size_t usable = budget > 2048 ? budget - 2048 : budget / 2;
  if (per <= usable) {
    int Wmax = (int)(usable / per); if (Wmax < 1) Wmax = 1;
    int W = dk_warps_target(); if (W > Wmax) W = Wmax; if (W < 1) W = 1;
    size_t smem = (size_t)W * per;
    cudaFuncSetAttribute(Rkendall_dense_warpB_different_blocks, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    int blocks = (int)((P + W - 1) / W); if (blocks > 65535) blocks = 65535; if (blocks < 1) blocks = 1;
    dim3 threads(32, W);
    Rkendall_dense_warpB_different_blocks<<<blocks, threads, smem>>>((float*)dA, (float*)dB, N, M, MB, cap, d_disc);
  } else {
    int W = dk_warps_g_target();
    const size_t sbudget = (size_t)2 * 1024 * 1024 * 1024;
    size_t per_warp_g = (size_t)4 * cap * sizeof(float);
    long long want_warps = (long long)(sbudget / per_warp_g); if (want_warps < W) want_warps = W;
    long long want_blocks = (want_warps + W - 1) / W;
    long long cover = (P + W - 1) / W;
    long long blocks = want_blocks < cover ? want_blocks : cover;
    if (blocks > 65535) blocks = 65535; if (blocks < 1) blocks = 1;
    float* d_scratch; cudaMalloc(&d_scratch, (size_t)blocks * W * 4 * cap * sizeof(float));
    dim3 threads(32, W);
    Rkendall_dense_warpG_different_blocks<<<(int)blocks, threads>>>((float*)dA, (float*)dB, N, M, MB, cap, d_scratch, d_disc);
    cudaDeviceSynchronize();
    cudaFree(d_scratch);
  }
  gpuErrchk(cudaPeekAtLastError());
}


__global__ void Rkendall_gpu_atomic_float(float* array, const int n, const int m, unsigned int* result) {
  
  int row1 = blockIdx.y * blockDim.y + threadIdx.y;
  int row2 = blockIdx.x * blockDim.x + threadIdx.x;

  if (row1 >= row2 || row1 >= n || row2 >= n) {
    return;
  }

  //if (row2 % 5000 == 0 && row1 % 500 == 0) {
  //  printf("%d %d %d\n", row1, row2, m);
  //}
  for (int col1_num = 0; col1_num < m; ++col1_num) {
      for (int col2_num = col1_num + 1; col2_num < m; ++col2_num) {
          float* col1 = array + n * col1_num;
          float* col2 = array + n * col2_num;

          if ((col1[row1] - col1[row2]) * (col2[row1] - col2[row2]) < 0){
            atomicAdd(result + col1_num * m + col2_num, 1);
//              atomicAdd(result + col2_num * m + col1_num, 1);
          }
      }
  }
}


__global__ void Rkendall_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, unsigned int* result) {
  
  int row1 = blockIdx.y * blockDim.y + threadIdx.y;
  int row2 = blockIdx.x * blockDim.x + threadIdx.x;

  if (row1 >= row2 || row2 >= n || row1 >= n) {
    return;
  }
  for (int col1_num = 0; col1_num < m; ++col1_num) {
      for (int col2_num = 0; col2_num < m_b; ++col2_num) {
          float* col1 = array + n * col1_num;
          float* col2 = array2 + n * col2_num;

          if ((col1[row1] - col1[row2]) * (col2[row1] - col2[row2]) < 0){
              atomicAdd(result + col2_num * m + col1_num, 1);
          }
      }
  }
}


//' Driver Function for calculation of Kendall matrix for same block.
//'
//' Allocates Memory required for the operation. Then,
//' efficiently calculate the distance matrix using the kernel, 
//' which is translated to appropriate R tables.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//' 
extern "C" void matrix_Kendall_distance_same_block(double* a, double * b /* not used */, double* c, int* n, int* m, int* m_b){
  int N = *n, M = *m;
  size_t asz = (size_t)N * M;
  std::vector<float> af(asz);
  for (size_t i = 0; i < asz; ++i) af[i] = (float)a[i];

  float* d_array; int* d_disc;
  cudaMalloc(&d_array, asz * sizeof(float));
  cudaMalloc(&d_disc, (size_t)M * M * sizeof(int));
  cudaMemcpy(d_array, af.data(), asz * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_disc, 0, (size_t)M * M * sizeof(int));

  dk_run_same(d_array, N, M, d_disc);            // variant B (shared) or G (warp-global)

  std::vector<int> h((size_t)M * M);
  cudaMemcpy(h.data(), d_disc, (size_t)M * M * sizeof(int), cudaMemcpyDeviceToHost);  // syncs
  double norm = (double)N * (N - 1);
  for (int row = 0; row < M; ++row)
    for (int col = 0; col < M; ++col) {
      size_t idx = (size_t)row * M + col;
      if (row < col)      c[idx] = (double)h[idx] * 2.0 / norm;
      else if (row > col) c[idx] = c[(size_t)col * M + row];
      else                c[idx] = 0.0;
    }
  cudaFree(d_array); cudaFree(d_disc);
}



//' Driver Function for calculation of Kendall matrix for different block.
//'
//' Allocates Memory required for the operation. Then,
//' efficiently calculate the distance matrix using the kernel, 
//' which is translated to appropriate R tables.
//'
//' @param a,b,c double pointers pointing to memory.
//' @param n,m,m_b Boundary values.
//' @export
//' 
extern "C" void matrix_Kendall_distance_different_blocks(double* a, double* b, double* c, int* n, int* m, int* m_b){
  int N = *n, M = *m, MB = *m_b;
  std::vector<float> af((size_t)N * M), bf((size_t)N * MB);
  for (size_t i = 0; i < (size_t)N * M; ++i)  af[i] = (float)a[i];
  for (size_t i = 0; i < (size_t)N * MB; ++i) bf[i] = (float)b[i];

  float* dA; float* dB; int* d_disc;
  cudaMalloc(&dA, (size_t)N * M  * sizeof(float));
  cudaMalloc(&dB, (size_t)N * MB * sizeof(float));
  cudaMalloc(&d_disc, (size_t)M * MB * sizeof(int));
  cudaMemcpy(dA, af.data(), (size_t)N * M  * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, bf.data(), (size_t)N * MB * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_disc, 0, (size_t)M * MB * sizeof(int));

  dk_run_diff(dA, dB, N, M, MB, d_disc);          // variant B (shared) or G (warp-global)

  std::vector<int> h((size_t)M * MB);
  cudaMemcpy(h.data(), d_disc, (size_t)M * MB * sizeof(int), cudaMemcpyDeviceToHost);  // syncs
  double norm = (double)N * (N - 1);
  for (size_t i = 0; i < (size_t)M * MB; ++i) c[i] = (double)h[i] * 2.0 / norm;

  cudaFree(dA); cudaFree(dB); cudaFree(d_disc);
}


__global__ void RkendallSparseCorr_gpu_atomic_float_same_block(
    int* a_index, int* a_positions, float* a_values,
    int* concordant, int rows, int columns) 
{
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int row_jndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_index >= row_jndex || row_jndex >= rows) {
    return;
  }
  //printf("%d\n", columns);
  //if (row_index % 200 == 0 && row_jndex % 5000 == 0) { 
  //    printf("%d %d\n", row_index, row_jndex);
  //}
  int start_column = a_positions[row_index];
  int end_column = a_positions[row_index + 1];

  int start_column_b = a_positions[row_jndex];
  int end_column_b = a_positions[row_jndex + 1];

  bool left_thresholds[5000];
  //printf("%d\n", end_column_b - start_column_b);
  for (int i = 0; i < end_column_b - start_column_b; ++i) {
    left_thresholds[i] = false;
  }
  bool left_threshold_selected = false;
  bool right_threshold_selected = false;
  int right_down1_threshold = start_column_b;
  int left_down1_threshold = start_column_b;
  int left_down2_threshold = start_column_b;
  int right_down2_threshold = start_column_b;
  bool left_activated = false;
  bool right_activated = false;
  for (int col1_index = start_column; col1_index < end_column; ++col1_index) {
    int prev_col_index = col1_index - 1;
    int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
    int col1 = a_index[col1_index];
    float value1 = a_values[col1_index];
    

    while (right_down1_threshold < end_column_b && a_index[right_down1_threshold] < col1) {
      right_down1_threshold += 1;
    }

    if (right_down1_threshold < end_column_b && a_index[right_down1_threshold] == col1) {
      left_activated = true;
    }
    if (right_down1_threshold < end_column_b && a_index[right_down1_threshold] == col1) {
      left_down2_threshold = right_down1_threshold + 1;
      right_activated = true;
    } else {
      left_down2_threshold = right_down1_threshold;
    }

    right_down2_threshold = left_down2_threshold;
    for (int col2_index = col1_index; col2_index < end_column; ++col2_index) {
      int col2 = a_index[col2_index];
      float value2 = a_values[col2_index];
      int next_col_index = col2_index + 1;
      int next_col = (next_col_index < end_column) ? a_index[next_col_index] : columns;

      while (right_down2_threshold < end_column_b && a_index[right_down2_threshold] < next_col) {
          right_down2_threshold += 1;
      }
      if (left_down1_threshold < end_column_b && !left_thresholds[left_down1_threshold - start_column_b]) {
        left_thresholds[left_down1_threshold - start_column_b] = true;
        for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
            for (int right = left + 1; right < right_down1_threshold; ++right) {
              float product = a_values[left] * a_values[right];
              if (product < 0) {
                atomicAdd(concordant + a_index[left] * columns + a_index[right], 1);
                // atomicAdd(concordant + a_index[right] * columns + a_index[left], 1);
                // disconcordant[a_index[left] * columns + a_index[right]] += 1;
                // disconcordant[a_index[right] * columns + a_index[left]] += 1;
              } 
              //else {
                //atomicAdd(concordant + a_index[left] * columns + a_index[right], 1);
                //atomicAdd(concordant + a_index[right] * columns + a_index[left], 1);
                // concordant[a_index[left] * columns + a_index[right]] += 1;
                // concordant[a_index[right] * columns + a_index[left]] += 1;
              //}
            }
        }
      }
      
      if (left_down2_threshold < end_column_b && !left_thresholds[left_down2_threshold - start_column_b]) {
        left_thresholds[left_down2_threshold - start_column_b] = true;
        for (int left = left_down2_threshold; left < right_down2_threshold; left++) {
            for (int right = left + 1; right < right_down2_threshold; ++right) {
              float product = a_values[left] * a_values[right];
              if (product < 0) {
                atomicAdd(concordant + a_index[left] * columns + a_index[right], 1);
                // atomicAdd(concordant + a_index[right] * columns + a_index[left], 1);
                // disconcordant[a_index[left] * columns + a_index[right]] += 1;
                // disconcordant[a_index[right] * columns + a_index[left]] += 1;
              } 
              //else {
              //  atomicAdd(concordant + a_index[left] * columns + a_index[right], 1);
              //  atomicAdd(concordant + a_index[right] * columns + a_index[left], 1);
                // concordant[a_index[left] * columns + a_index[right]] += 1;
                // concordant[a_index[right] * columns + a_index[left]] += 1;
              //}
            }
        }
      }
      
      for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
            for (int right = left_down2_threshold; right < right_down2_threshold; ++right) {
              float product = a_values[left] * a_values[right];
              if (product < 0) {
                atomicAdd(concordant + a_index[left] * columns + a_index[right], 1);
                // atomicAdd(concordant + a_index[right] * columns + a_index[left], 1);
                // disconcordant[a_index[left] * columns + a_index[right]] += 1;
                // disconcordant[a_index[right] * columns + a_index[left]] += 1;
              } 
              //else {
              //  atomicAdd(concordant + a_index[left] * columns + a_index[right], 1);
              //  atomicAdd(concordant + a_index[right] * columns + a_index[left], 1);
                // concordant[a_index[left] * columns + a_index[right]] += 1;
                // concordant[a_index[right] * columns + a_index[left]] += 1;
              //}
            }
        }
      
      float left_value = (left_activated) ? a_values[right_down1_threshold] : 0;
      float right_value = (right_activated) ? a_values[left_down2_threshold - 1] : 0;
      
      float left_diff = left_value - a_values[col1_index];
      float right_diff = right_value - a_values[col2_index];
      float product = left_diff * right_diff;
      //if (product > 0) {
      //  atomicAdd(concordant + col1 * columns + col2, 1);
      //  atomicAdd(concordant + col2 * columns + col1, 1);
      //} else 
      if (product < 0) {
        atomicAdd(concordant + col1 * columns + col2, 1);
        // atomicAdd(concordant + col2 * columns + col1, 1);
      }
      
      for (int right = left_down2_threshold; right < right_down2_threshold; ++right) {
        product = left_diff * a_values[right];
        if (product < 0) {
          atomicAdd(concordant + col1 * columns + a_index[right], 1);
          //atomicAdd(concordant + a_index[right] * columns + col1, 1);
        } 
        //else if (product > 0) {
        //  atomicAdd(concordant + col1 * columns + a_index[right], 1);
        //  atomicAdd(concordant + a_index[right] * columns + col1, 1);
        //}
      }
          
      for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
        // std::cout << a_index[left] << " " << a_index[col2_index] << std::endl;
        product = right_diff * a_values[left];
        if (product < 0) {
          atomicAdd(concordant + a_index[left] * columns + col2, 1);
          //atomicAdd(concordant + col2 * columns + a_index[left], 1);
        }
        //else if (product > 0) {
        //  atomicAdd(concordant + a_index[left]* columns + col2, 1);
        //  atomicAdd(concordant + col2 * columns + a_index[left], 1);
        //}
      }

      right_activated = false;
      while (left_down2_threshold < end_column_b && a_index[left_down2_threshold] <= next_col) {
          if (a_index[left_down2_threshold] == next_col) {
            right_activated = true;
          }
          left_down2_threshold += 1;
      }

    }
    
    while (left_down1_threshold < end_column_b && a_index[left_down1_threshold] <= col1) {
      left_down1_threshold += 1;
    }
  }
}


// CSR(genes x cells) -> CSC(genes x cells): transpose of storage. Gene indices
// within each output cell column come out ascending (genes scanned in order),
// which the per_cell_pair merge requires.
static void kendall_csr_genes_to_csc_cells(
    const int* row_ptr, const int* col_idx, const double* vals,
    int n_genes, int n_cells, int nnz,
    std::vector<int>& csc_p, std::vector<int>& csc_i, std::vector<double>& csc_x)
{
  csc_p.assign(n_cells + 1, 0);
  csc_i.resize(nnz); csc_x.resize(nnz);
  for (int k = 0; k < nnz; ++k) csc_p[col_idx[k] + 1]++;
  for (int c = 0; c < n_cells; ++c) csc_p[c + 1] += csc_p[c];
  std::vector<int> next(csc_p.begin(), csc_p.end());
  for (int g = 0; g < n_genes; ++g)
    for (int k = row_ptr[g]; k < row_ptr[g + 1]; ++k) {
      int c = col_idx[k]; int dst = next[c]++;
      csc_i[dst] = g; csc_x[dst] = vals[k];
    }
}


// per_cell_pair drivers (defined later) that the default path delegates to.
extern "C" void matrix_Kendall_sparse_per_cell_pair_distance_same_block(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);

extern "C" void matrix_Kendall_sparse_per_cell_pair_distance_different_blocks(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);


// The legacy per-gene-pair GPU kernel (RkendallSparseCorr_gpu_atomic_float_*) was
// numerically incorrect (diverged from ground truth); the default sparse Kendall
// GPU path now delegates to the validated per_cell_pair implementation. The
// default input is CSR-by-genes (a_positions over genes, a_index = cell ids);
// transpose it to the CSC-by-cells layout per_cell_pair expects.
extern "C" void matrix_Kendall_sparse_distance_same_block(
    int* a_index, int* a_positions, double* a_double_values,
    int* /*b_index*/, int* /*b_positions*/, double* /*b_double_values*/,
    double* result, int* num_rows, int* num_columns, int* /*num_columns_b*/,
    int* num_elements_a, int* /*num_elements_b*/)
{
  int n_genes = *num_rows, n_cells = *num_columns, nnz = *num_elements_a;
  std::vector<int> csc_p, csc_i; std::vector<double> csc_x;
  kendall_csr_genes_to_csc_cells(a_positions, a_index, a_double_values,
                                 n_genes, n_cells, nnz, csc_p, csc_i, csc_x);
  matrix_Kendall_sparse_per_cell_pair_distance_same_block(
      csc_i.data(), csc_p.data(), csc_x.data(), nullptr, nullptr, nullptr,
      result, num_rows, num_columns, nullptr, num_elements_a, nullptr);
}


__global__ void RkendallSparseCorr_gpu_atomic_float_different_blocks(
    int* a_index, int* a_positions, float* a_values,
    int* b_index, int* b_positions, float* b_values,
    int* concordant, int rows, int columns, int columns_b) 
{
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int row_jndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_index >= row_jndex || row_jndex >= rows) {
    return;
  }
  //if (row_index % 200 == 0 && row_jndex % 2000 == 0) { 
  //    printf("%d %d\n", row_index, row_jndex);
  //}

  int start_column = a_positions[row_index];
  int end_column = a_positions[row_index + 1];
  int start_column_down = a_positions[row_jndex];
  int end_column_down = a_positions[row_jndex + 1];

  int start_column_b = b_positions[row_index];
  int end_column_b = b_positions[row_index + 1];
  int start_column_down_b = b_positions[row_jndex];
  int end_column_down_b = b_positions[row_jndex + 1];
  bool left_threshold_selected = false;
  bool right_threshold_selected = false;
  int right_down1_threshold = start_column_down;
  int left_down1_threshold = start_column_down;
  bool left_activated = false;
  bool right_activated = false;
  // std::cout << "BEFORE THRESHOLD " << left_down1_threshold << " " << right_down1_threshold << " " << left_down2_threshold << " " << right_down2_threshold << std::endl;

  for (int col1_index = start_column; col1_index <= end_column; ++col1_index) {
    int prev_col_index = col1_index - 1;
    int prev_col = (prev_col_index >= start_column) ? a_index[prev_col_index] : -1;
    int col1 = (col1_index < end_column) ? a_index[col1_index]: columns;
    float value1 = (col1_index < end_column) ? a_values[col1_index]: 0;
    
    while (right_down1_threshold < end_column_down && a_index[right_down1_threshold] < col1) {
      right_down1_threshold += 1;
    }

    if (right_down1_threshold < end_column_down && a_index[right_down1_threshold] == col1) {
      left_activated = true;
    }

    int left_down2_threshold = start_column_down_b;
    int right_down2_threshold = start_column_down_b;
    right_activated = false;
    for (int col2_index = start_column_b; col2_index < end_column_b; ++col2_index) {
      int col2 = b_index[col2_index];
      float value2 = b_values[col2_index];
      int next_col_index = col2_index + 1;
      int next_col = (next_col_index < end_column_b) ? b_index[next_col_index] : columns_b;

      while (right_down2_threshold < end_column_down_b && b_index[right_down2_threshold] < next_col) {
          right_down2_threshold += 1;
      }
      
      for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
            for (int right = left_down2_threshold; right < right_down2_threshold; ++right) {
              float product = a_values[left] * b_values[right];
              if (product < 0) {
                atomicAdd(
                  concordant + b_index[right] * columns + a_index[left], 1
                );
              }
              //else {
              //  atomicAdd(
              //    concordant + b_index[right] * columns + a_index[left], 1
              //  );
              //}
            }
        }
      // std::cout << "COL" << col1 << " " << col2 << std::endl;
      // std::cout << "THRESHOLD " << left_down1_threshold << " " << right_down1_threshold << " " << left_down2_threshold << " " << right_down2_threshold << std::endl;
      
      float left_value = (left_activated) ? a_values[right_down1_threshold] : 0;
      float right_value = (right_activated) ? b_values[left_down2_threshold - 1] : 0;
      
      
      float right_diff = right_value - value2;
      
      float left_diff = left_value - value1;
      float product = left_diff * right_diff;
      if (product < 0) {
        atomicAdd(concordant + col2 * columns + col1, 1);
      }
      
      for (int right = left_down2_threshold; right < right_down2_threshold; ++right) {
        product = left_diff * b_values[right];
        if (product < 0) {
          atomicAdd(concordant + b_index[right] * columns + col1, 1);
        } 
        //else if (product > 0) {
        //  atomicAdd(concordant + b_index[right] * columns + col1, 1);
        //}
      }
          
      for (int left = left_down1_threshold; left < right_down1_threshold; left++) {
        product = right_diff * a_values[left];
        if (product < 0) {
          atomicAdd(concordant + col2 * columns + a_index[left], 1);
        } 
        //else if (product > 0) {
        //  atomicAdd(concordant + col2 * columns + a_index[left], 1);
        //}
      }

      right_activated = false;
      while (left_down2_threshold < end_column_down_b && b_index[left_down2_threshold] <= next_col) {
          if (b_index[left_down2_threshold] == next_col) {
            right_activated = true;
          }
          left_down2_threshold += 1;
      }

    }
    
    while (left_down1_threshold < end_column_down && a_index[left_down1_threshold] <= col1) {
      left_down1_threshold += 1;
    }
  }
}


extern "C" void matrix_Kendall_sparse_distance_different_blocks(
    int* a_index, int* a_positions, double* a_double_values,
    int* b_index, int* b_positions, double* b_double_values,
    double* result, int* num_rows, int* num_columns, int* num_columns_b,
    int* num_elements_a, int* num_elements_b) {

  // Delegate to the validated per_cell_pair path (see same_block note). Both
  // blocks arrive as CSR-by-genes; transpose each to CSC-by-cells.
  int n_genes = *num_rows, n_cells_a = *num_columns, n_cells_b = *num_columns_b;
  int nnz_a = *num_elements_a, nnz_b = *num_elements_b;
  std::vector<int> ap, ai, bp, bi; std::vector<double> ax, bx;
  kendall_csr_genes_to_csc_cells(a_positions, a_index, a_double_values,
                                 n_genes, n_cells_a, nnz_a, ap, ai, ax);
  kendall_csr_genes_to_csc_cells(b_positions, b_index, b_double_values,
                                 n_genes, n_cells_b, nnz_b, bp, bi, bx);
  matrix_Kendall_sparse_per_cell_pair_distance_different_blocks(
      ai.data(), ap.data(), ax.data(), bi.data(), bp.data(), bx.data(),
      result, num_rows, num_columns, num_columns_b, num_elements_a, num_elements_b);
}


// ==================== Per-cell-pair sparse Kendall ====================
//
// Alternative sparse Kendall using CSC layout (CsparseMatrix from R).
// 1 thread per (cell_a, cell_b) pair with double two-pointer merge.
// No atomics, no sweep-line, signed-correct via n_signflip * n_inactive.
// See plans/PER_CELL_PAIR.md for algorithm details.

__global__ void RkendallSparseCorr_gpu_per_cell_pair_same_block(
    const int* __restrict__ csc_p,
    const int* __restrict__ csc_i,
    const float* __restrict__ csc_x,
    int n_genes,
    int n_cells,
    int* __restrict__ discordant_out)
{
  int cell_a = blockIdx.y * blockDim.y + threadIdx.y;
  int cell_b = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_a >= cell_b || cell_b >= n_cells) return;

  int ia = csc_p[cell_a], ea = csc_p[cell_a + 1];
  int ib = csc_p[cell_b], eb = csc_p[cell_b + 1];

  int discordant = 0;
  int n_active   = 0;
  int n_signflip = 0;

  while (ia < ea || ib < eb) {
    float a_i, b_i;
    int next_ia = ia, next_ib = ib;

    if (ia < ea && (ib >= eb || csc_i[ia] < csc_i[ib])) {
      a_i = csc_x[ia]; b_i = 0.0f;
      next_ia = ia + 1;
    } else if (ib < eb && (ia >= ea || csc_i[ib] < csc_i[ia])) {
      a_i = 0.0f; b_i = csc_x[ib];
      next_ib = ib + 1;
    } else {
      a_i = csc_x[ia]; b_i = csc_x[ib];
      next_ia = ia + 1;
      next_ib = ib + 1;
    }

    ++n_active;
    if (a_i * b_i < 0.0f) ++n_signflip;

    int ja = next_ia, jb = next_ib;
    while (ja < ea || jb < eb) {
      float a_j, b_j;
      if (ja < ea && (jb >= eb || csc_i[ja] < csc_i[jb])) {
        a_j = csc_x[ja]; b_j = 0.0f; ++ja;
      } else if (jb < eb && (ja >= ea || csc_i[jb] < csc_i[ja])) {
        a_j = 0.0f; b_j = csc_x[jb]; ++jb;
      } else {
        a_j = csc_x[ja]; b_j = csc_x[jb]; ++ja; ++jb;
      }
      if ((a_i - a_j) * (b_i - b_j) < 0.0f) ++discordant;
    }

    ia = next_ia;
    ib = next_ib;
  }

  int n_inactive = n_genes - n_active;
  discordant += n_signflip * n_inactive;

  discordant_out[cell_a * n_cells + cell_b] = discordant;
}


__global__ void RkendallSparseCorr_gpu_per_cell_pair_different_blocks(
    const int* __restrict__ a_csc_p,
    const int* __restrict__ a_csc_i,
    const float* __restrict__ a_csc_x,
    const int* __restrict__ b_csc_p,
    const int* __restrict__ b_csc_i,
    const float* __restrict__ b_csc_x,
    int n_genes,
    int n_cells_a,
    int n_cells_b,
    int* __restrict__ discordant_out)
{
  int cell_a = blockIdx.y * blockDim.y + threadIdx.y;
  int cell_b = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_a >= n_cells_a || cell_b >= n_cells_b) return;

  int ia = a_csc_p[cell_a], ea = a_csc_p[cell_a + 1];
  int ib = b_csc_p[cell_b], eb = b_csc_p[cell_b + 1];

  int discordant = 0;
  int n_active   = 0;
  int n_signflip = 0;

  while (ia < ea || ib < eb) {
    float a_i, b_i;
    int next_ia = ia, next_ib = ib;

    if (ia < ea && (ib >= eb || a_csc_i[ia] < b_csc_i[ib])) {
      a_i = a_csc_x[ia]; b_i = 0.0f;
      next_ia = ia + 1;
    } else if (ib < eb && (ia >= ea || b_csc_i[ib] < a_csc_i[ia])) {
      a_i = 0.0f; b_i = b_csc_x[ib];
      next_ib = ib + 1;
    } else {
      a_i = a_csc_x[ia]; b_i = b_csc_x[ib];
      next_ia = ia + 1;
      next_ib = ib + 1;
    }

    ++n_active;
    if (a_i * b_i < 0.0f) ++n_signflip;

    int ja = next_ia, jb = next_ib;
    while (ja < ea || jb < eb) {
      float a_j, b_j;
      if (ja < ea && (jb >= eb || a_csc_i[ja] < b_csc_i[jb])) {
        a_j = a_csc_x[ja]; b_j = 0.0f; ++ja;
      } else if (jb < eb && (ja >= ea || b_csc_i[jb] < a_csc_i[ja])) {
        a_j = 0.0f; b_j = b_csc_x[jb]; ++jb;
      } else {
        a_j = a_csc_x[ja]; b_j = b_csc_x[jb]; ++ja; ++jb;
      }
      if ((a_i - a_j) * (b_i - b_j) < 0.0f) ++discordant;
    }

    ia = next_ia;
    ib = next_ib;
  }

  int n_inactive = n_genes - n_active;
  discordant += n_signflip * n_inactive;

  discordant_out[cell_b * n_cells_a + cell_a] = discordant;
}


__global__ void FinalizeKendallPerCellPair(
    int n_cells, int n_genes,
    const int* __restrict__ discordant_in,
    double* __restrict__ result_out)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_cells || col >= n_cells) return;

  int idx = row * n_cells + col;
  if (row < col) {
    double d = (double)discordant_in[idx];
    result_out[idx] = d * 2.0 / ((double)n_genes * (n_genes - 1));
  } else if (row > col) {
    int sym = col * n_cells + row;
    double d = (double)discordant_in[sym];
    result_out[idx] = d * 2.0 / ((double)n_genes * (n_genes - 1));
  } else {
    result_out[idx] = 0.0;
  }
}


// Bare per_cell_pair delegates to the best O(k log k) variant: B (cooperative
// merge-sort in shared memory) when the per-warp scratch fits the smem budget,
// else A (global scratch arena) for large feature counts. Mirrors the dense
// driver's B/A selection. (pcp_max_col_nnz / pcp_dispatch_cfg etc. now in pcp_dispatch.cuh.)

extern "C" void matrix_Kendall_sparse_per_cell_pair_a_distance_same_block(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);

extern "C" void matrix_Kendall_sparse_per_cell_pair_b_distance_same_block(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);

extern "C" void matrix_Kendall_sparse_per_cell_pair_a_distance_different_blocks(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);

extern "C" void matrix_Kendall_sparse_per_cell_pair_b_distance_different_blocks(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);

extern "C" void matrix_Kendall_sparse_per_cell_pair_dispatch_distance_same_block(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);

extern "C" void matrix_Kendall_sparse_per_cell_pair_dispatch_distance_different_blocks(
    int*, int*, double*, int*, int*, double*, double*, int*, int*, int*, int*, int*);


extern "C" void matrix_Kendall_sparse_per_cell_pair_distance_same_block(
    int* csc_i_in,
    int* csc_p_in,
    double* csc_x_in,
    int* /*b_index*/,
    int* /*b_positions*/,
    double* /*b_values*/,
    double* result,
    int* num_rows,
    int* num_columns,
    int* /*num_columns_b*/,
    int* num_elements,
    int* /*num_elements_b*/)
{
  // Per-pair dispatch: each cell pair is routed individually — cheap pairs to
  // the warp+shared merge, rare oversized pairs to the 1-thread global-scratch
  // fallback. The shared budget is queried from the device, not hardcoded.
  // This decouples the variant choice from the tile's global max_k, removing
  // the variant-A poisoning where one fat column dragged the whole tile slow.
  matrix_Kendall_sparse_per_cell_pair_dispatch_distance_same_block(
      csc_i_in, csc_p_in, csc_x_in, nullptr, nullptr, nullptr,
      result, num_rows, num_columns, nullptr, num_elements, nullptr);
}


extern "C" void matrix_Kendall_sparse_per_cell_pair_distance_different_blocks(
    int* a_csc_i_in,
    int* a_csc_p_in,
    double* a_csc_x_in,
    int* b_csc_i_in,
    int* b_csc_p_in,
    double* b_csc_x_in,
    double* result,
    int* num_rows,
    int* num_columns,
    int* num_columns_b,
    int* num_elements_a,
    int* num_elements_b)
{
  matrix_Kendall_sparse_per_cell_pair_dispatch_distance_different_blocks(
      a_csc_i_in, a_csc_p_in, a_csc_x_in, b_csc_i_in, b_csc_p_in, b_csc_x_in,
      result, num_rows, num_columns, num_columns_b, num_elements_a, num_elements_b);
}


// ==================== Per-cell-pair sparse Kendall: O(k log k) variants ====================
//
// Three implementations of the same tau-naive discordant count, to compare
// experimentally:
//   A      -- 1 thread per pair, O(k log k) merge-sort inversion count, scratch
//             in a global arena (persistent-thread grid).
//   B      -- 1 warp per pair, the same merge-sort/count done cooperatively in
//             shared memory.
//   Hybrid -- 1 thread per pair, branch on union nnz: small k -> O(k^2) with
//             zero scratch, large k -> O(k log k) global-scratch path.
// All write an int discordant matrix consumed by FinalizeKendallPerCellPair
// (same_block) or host-normalized (different_blocks).

#define PCP_HYBRID_THRESHOLD 64   // union-nnz cutoff between O(k^2) and O(k log k)

#define PCP_B_SMALL 64            // warp-B: union size at/below which the O(k^2) shared path is used


// --- O(k^2) double merge, zero scratch (reused by hybrid) ---
__device__ inline long long pcp_disc_k2(
    const int* a_i, const float* a_x, int ia, int ea,
    const int* b_i, const float* b_x, int ib, int eb,
    int n_genes)
{
  long long discordant = 0;
  int n_active = 0, n_signflip = 0;
  int oia = ia, oib = ib;
  while (oia < ea || oib < eb) {
    float av, bv; int nia = oia, nib = oib;
    if (oia < ea && (oib >= eb || a_i[oia] < b_i[oib])) { av = a_x[oia]; bv = 0.0f; nia = oia + 1; }
    else if (oib < eb && (oia >= ea || b_i[oib] < a_i[oia])) { av = 0.0f; bv = b_x[oib]; nib = oib + 1; }
    else { av = a_x[oia]; bv = b_x[oib]; nia = oia + 1; nib = oib + 1; }
    ++n_active;
    if (av * bv < 0.0f) ++n_signflip;
    int ja = nia, jb = nib;
    while (ja < ea || jb < eb) {
      float aj, bj;
      if (ja < ea && (jb >= eb || a_i[ja] < b_i[jb])) { aj = a_x[ja]; bj = 0.0f; ++ja; }
      else if (jb < eb && (ja >= ea || b_i[jb] < a_i[ja])) { aj = 0.0f; bj = b_x[jb]; ++jb; }
      else { aj = a_x[ja]; bj = b_x[jb]; ++ja; ++jb; }
      if ((av - aj) * (bv - bj) < 0.0f) ++discordant;
    }
    oia = nia; oib = nib;
  }
  discordant += (long long)n_signflip * (n_genes - n_active);
  return discordant;
}


// --- single-thread merge sort of paired arrays by (a asc, b asc); result in va,vb ---
__device__ inline void pcp_sort_ab(float* va, float* vb, float* ta, float* tb, int k) {
  float* sa = va; float* sb = vb; float* da = ta; float* db = tb;
  for (int w = 1; w < k; w <<= 1) {
    for (int lo = 0; lo < k; lo += 2 * w) {
      int mid = min(lo + w, k), hi = min(lo + 2 * w, k);
      int i = lo, j = mid, o = lo;
      while (i < mid && j < hi) {
        bool tl = (sa[i] < sa[j]) || (sa[i] == sa[j] && sb[i] <= sb[j]);
        if (tl) { da[o] = sa[i]; db[o] = sb[i]; ++i; }
        else    { da[o] = sa[j]; db[o] = sb[j]; ++j; }
        ++o;
      }
      while (i < mid) { da[o] = sa[i]; db[o] = sb[i]; ++i; ++o; }
      while (j < hi)  { da[o] = sa[j]; db[o] = sb[j]; ++j; ++o; }
    }
    float* t;
    t = sa; sa = da; da = t;
    t = sb; sb = db; db = t;
  }
  if (sa != va) for (int i = 0; i < k; ++i) { va[i] = sa[i]; vb[i] = sb[i]; }
}


// --- single-thread strict inversion count (pairs i<j with x[i] > x[j]); x consumed ---
__device__ inline long long pcp_count_inv(float* x, float* tmp, int k) {
  long long inv = 0;
  float* s = x; float* d = tmp;
  for (int w = 1; w < k; w <<= 1) {
    for (int lo = 0; lo < k; lo += 2 * w) {
      int mid = min(lo + w, k), hi = min(lo + 2 * w, k);
      int i = lo, j = mid, o = lo;
      while (i < mid && j < hi) {
        if (s[i] <= s[j]) d[o++] = s[i++];
        else { d[o++] = s[j++]; inv += (mid - i); }
      }
      while (i < mid) d[o++] = s[i++];
      while (j < hi)  d[o++] = s[j++];
    }
    float* t = s; s = d; d = t;
  }
  return inv;
}


// --- O(k log k) discordant count for one pair, single thread, global scratch ---
// scratch holds 4 float arrays of size cap: va, vb, ta, tb.
__device__ inline long long pcp_disc_klogn(
    const int* a_i, const float* a_x, int ia, int ea,
    const int* b_i, const float* b_x, int ib, int eb,
    int n_genes, float* scratch, int cap)
{
  float* va = scratch;
  float* vb = scratch + cap;
  float* ta = scratch + 2 * cap;
  float* tb = scratch + 3 * cap;

  int k = 0, n_signflip = 0;
  int oia = ia, oib = ib;
  while (oia < ea || oib < eb) {
    float av, bv;
    if (oia < ea && (oib >= eb || a_i[oia] < b_i[oib])) { av = a_x[oia]; bv = 0.0f; ++oia; }
    else if (oib < eb && (oia >= ea || b_i[oib] < a_i[oia])) { av = 0.0f; bv = b_x[oib]; ++oib; }
    else { av = a_x[oia]; bv = b_x[oib]; ++oia; ++oib; }
    va[k] = av; vb[k] = bv; ++k;
    if (av * bv < 0.0f) ++n_signflip;
  }
  long long discordant = (long long)n_signflip * (n_genes - k);
  if (k >= 2) {
    pcp_sort_ab(va, vb, ta, tb, k);
    discordant += pcp_count_inv(vb, ta, k);   // vb is in a-order; ta is free
  }
  return discordant;
}


// ---------- Variant A kernels (thread per pair, global scratch arena) ----------
__global__ void RkendallPCP_A_same_block(
    const int* __restrict__ csc_p, const int* __restrict__ csc_i, const float* __restrict__ csc_x,
    int n_genes, int n_cells, int cap, float* scratch_arena, int* __restrict__ discordant_out)
{
  long long P = (long long)n_cells * n_cells;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  float* my = scratch_arena + (size_t)tid * 4 * cap;
  for (long long idx = tid; idx < P; idx += nthreads) {
    int a = (int)(idx / n_cells), b = (int)(idx % n_cells);
    if (a >= b) continue;
    long long disc = pcp_disc_klogn(csc_i, csc_x, csc_p[a], csc_p[a + 1],
                                    csc_i, csc_x, csc_p[b], csc_p[b + 1], n_genes, my, cap);
    discordant_out[a * n_cells + b] = (int)disc;
  }
}


__global__ void RkendallPCP_A_different_blocks(
    const int* __restrict__ a_csc_p, const int* __restrict__ a_csc_i, const float* __restrict__ a_csc_x,
    const int* __restrict__ b_csc_p, const int* __restrict__ b_csc_i, const float* __restrict__ b_csc_x,
    int n_genes, int n_cells_a, int n_cells_b, int cap, float* scratch_arena, int* __restrict__ discordant_out)
{
  long long P = (long long)n_cells_a * n_cells_b;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  float* my = scratch_arena + (size_t)tid * 4 * cap;
  for (long long idx = tid; idx < P; idx += nthreads) {
    int a = (int)(idx / n_cells_b), b = (int)(idx % n_cells_b);
    long long disc = pcp_disc_klogn(a_csc_i, a_csc_x, a_csc_p[a], a_csc_p[a + 1],
                                    b_csc_i, b_csc_x, b_csc_p[b], b_csc_p[b + 1], n_genes, my, cap);
    discordant_out[b * n_cells_a + a] = (int)disc;
  }
}


// ---------- Hybrid kernels (thread per pair, branch on union nnz) ----------
__global__ void RkendallPCP_Hybrid_same_block(
    const int* __restrict__ csc_p, const int* __restrict__ csc_i, const float* __restrict__ csc_x,
    int n_genes, int n_cells, int cap, int threshold, float* scratch_arena, int* __restrict__ discordant_out)
{
  long long P = (long long)n_cells * n_cells;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  float* my = scratch_arena + (size_t)tid * 4 * cap;
  for (long long idx = tid; idx < P; idx += nthreads) {
    int a = (int)(idx / n_cells), b = (int)(idx % n_cells);
    if (a >= b) continue;
    int ia = csc_p[a], ea = csc_p[a + 1], ib = csc_p[b], eb = csc_p[b + 1];
    long long disc;
    if ((ea - ia) + (eb - ib) <= threshold)
      disc = pcp_disc_k2(csc_i, csc_x, ia, ea, csc_i, csc_x, ib, eb, n_genes);
    else
      disc = pcp_disc_klogn(csc_i, csc_x, ia, ea, csc_i, csc_x, ib, eb, n_genes, my, cap);
    discordant_out[a * n_cells + b] = (int)disc;
  }
}


__global__ void RkendallPCP_Hybrid_different_blocks(
    const int* __restrict__ a_csc_p, const int* __restrict__ a_csc_i, const float* __restrict__ a_csc_x,
    const int* __restrict__ b_csc_p, const int* __restrict__ b_csc_i, const float* __restrict__ b_csc_x,
    int n_genes, int n_cells_a, int n_cells_b, int cap, int threshold, float* scratch_arena, int* __restrict__ discordant_out)
{
  long long P = (long long)n_cells_a * n_cells_b;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  float* my = scratch_arena + (size_t)tid * 4 * cap;
  for (long long idx = tid; idx < P; idx += nthreads) {
    int a = (int)(idx / n_cells_b), b = (int)(idx % n_cells_b);
    int ia = a_csc_p[a], ea = a_csc_p[a + 1], ib = b_csc_p[b], eb = b_csc_p[b + 1];
    long long disc;
    if ((ea - ia) + (eb - ib) <= threshold)
      disc = pcp_disc_k2(a_csc_i, a_csc_x, ia, ea, b_csc_i, b_csc_x, ib, eb, n_genes);
    else
      disc = pcp_disc_klogn(a_csc_i, a_csc_x, ia, ea, b_csc_i, b_csc_x, ib, eb, n_genes, my, cap);
    discordant_out[b * n_cells_a + a] = (int)disc;
  }
}


// ---------- Variant B: warp-cooperative merge sort + inversion count ----------
// Lanes (threadIdx.x, 0..31) cooperate on one pair; warps (threadIdx.y) and
// blocks (blockIdx.x) grid-stride over pairs. Shared layout per warp: 4 arrays
// of `cap` floats (sa, sb, ta, tb).

__device__ inline void warp_sort_ab(float* a0, float* b0, float* a1, float* b1, int k, int lane) {
  float *sa = a0, *sb = b0, *da = a1, *db = b1;
  for (int w = 1; w < k; w <<= 1) {
    int step = 2 * w, nmerges = (k + step - 1) / step;
    for (int m = lane; m < nmerges; m += 32) {
      int lo = m * step, mid = min(lo + w, k), hi = min(lo + step, k);
      int i = lo, j = mid, o = lo;
      while (i < mid && j < hi) {
        bool tl = (sa[i] < sa[j]) || (sa[i] == sa[j] && sb[i] <= sb[j]);
        if (tl) { da[o] = sa[i]; db[o] = sb[i]; ++i; }
        else    { da[o] = sa[j]; db[o] = sb[j]; ++j; }
        ++o;
      }
      while (i < mid) { da[o] = sa[i]; db[o] = sb[i]; ++i; ++o; }
      while (j < hi)  { da[o] = sa[j]; db[o] = sb[j]; ++j; ++o; }
    }
    __syncwarp();
    float* t; t = sa; sa = da; da = t; t = sb; sb = db; db = t;
  }
  if (sa != a0) { for (int t = lane; t < k; t += 32) { a0[t] = sa[t]; b0[t] = sb[t]; } __syncwarp(); }
}


__device__ inline long long warp_count_inv(float* x0, float* x1, int k, int lane) {
  float *s = x0, *d = x1;
  long long inv = 0;
  for (int w = 1; w < k; w <<= 1) {
    int step = 2 * w, nmerges = (k + step - 1) / step;
    for (int m = lane; m < nmerges; m += 32) {
      int lo = m * step, mid = min(lo + w, k), hi = min(lo + step, k);
      int i = lo, j = mid, o = lo;
      while (i < mid && j < hi) {
        if (s[i] <= s[j]) d[o++] = s[i++];
        else { d[o++] = s[j++]; inv += (mid - i); }
      }
      while (i < mid) d[o++] = s[i++];
      while (j < hi)  d[o++] = s[j++];
    }
    __syncwarp();
    float* t = s; s = d; d = t;
  }
  // 64-bit warp reduction (two 32-bit shuffles per step)
  unsigned long long u = (unsigned long long)inv;
  for (int off = 16; off > 0; off >>= 1) {
    unsigned int ulo = __shfl_down_sync(0xffffffff, (unsigned int)(u & 0xffffffffu), off);
    unsigned int uhi = __shfl_down_sync(0xffffffff, (unsigned int)(u >> 32), off);
    u += ((unsigned long long)uhi << 32) | ulo;
  }
  return (long long)u; // valid on lane 0
}


// --- warp-parallel O(k^2) discordant count over points already in shared mem ---
__device__ inline long long warp_disc_k2_shared(const float* sa, const float* sb, int k, int lane) {
  long long disc = 0;
  for (int i = lane; i < k; i += 32) {
    float ai = sa[i], bi = sb[i];
    for (int j = i + 1; j < k; ++j)
      if ((ai - sa[j]) * (bi - sb[j]) < 0.0f) ++disc;
  }
  unsigned long long u = (unsigned long long)disc;
  for (int off = 16; off > 0; off >>= 1) {
    unsigned int ulo = __shfl_down_sync(0xffffffff, (unsigned int)(u & 0xffffffffu), off);
    unsigned int uhi = __shfl_down_sync(0xffffffff, (unsigned int)(u >> 32), off);
    u += ((unsigned long long)uhi << 32) | ulo;
  }
  return (long long)u; // valid on lane 0
}


__device__ inline long long pcp_disc_warp(
    const int* a_i, const float* a_x, int ia, int ea,
    const int* b_i, const float* b_x, int ib, int eb,
    int n_genes, float* smem_warp, int cap, int lane, int* s_k, int* s_nsf, int wib, int small_k)
{
  float* sa = smem_warp;
  float* sb = smem_warp + cap;
  float* ta = smem_warp + 2 * cap;
  float* tb = smem_warp + 3 * cap;

  if (lane == 0) {
    int k = 0, nsf = 0, oia = ia, oib = ib;
    while (oia < ea || oib < eb) {
      float av, bv;
      if (oia < ea && (oib >= eb || a_i[oia] < b_i[oib])) { av = a_x[oia]; bv = 0.0f; ++oia; }
      else if (oib < eb && (oia >= ea || b_i[oib] < a_i[oia])) { av = 0.0f; bv = b_x[oib]; ++oib; }
      else { av = a_x[oia]; bv = b_x[oib]; ++oia; ++oib; }
      sa[k] = av; sb[k] = bv; ++k;
      if (av * bv < 0.0f) ++nsf;
    }
    s_k[wib] = k; s_nsf[wib] = nsf;
  }
  __syncwarp();
  int k = s_k[wib], nsf = s_nsf[wib];
  long long discordant = (long long)nsf * (n_genes - k);
  if (k >= 2) {
    long long inv;
    if (k <= small_k) {
      inv = warp_disc_k2_shared(sa, sb, k, lane);   // tiny k: skip the sort
    } else {
      warp_sort_ab(sa, sb, ta, tb, k, lane);        // sb now in a-order
      inv = warp_count_inv(sb, ta, k, lane);
    }
    if (lane == 0) discordant += inv;
  }
  return discordant; // valid on lane 0
}


__global__ void RkendallPCP_B_same_block(
    const int* __restrict__ csc_p, const int* __restrict__ csc_i, const float* __restrict__ csc_x,
    int n_genes, int n_cells, int cap, int small_k, int* __restrict__ discordant_out)
{
  extern __shared__ float smem[];
  __shared__ int s_k[32];
  __shared__ int s_nsf[32];
  int lane = threadIdx.x;
  int wib = threadIdx.y;
  float* smem_warp = smem + (size_t)wib * 4 * cap;

  long long P = (long long)n_cells * n_cells;
  long long gw = (long long)blockIdx.x * blockDim.y + wib;
  long long stride = (long long)gridDim.x * blockDim.y;
  for (long long idx = gw; idx < P; idx += stride) {
    int a = (int)(idx / n_cells), b = (int)(idx % n_cells);
    if (a >= b) continue;
    long long disc = pcp_disc_warp(csc_i, csc_x, csc_p[a], csc_p[a + 1],
                                   csc_i, csc_x, csc_p[b], csc_p[b + 1],
                                   n_genes, smem_warp, cap, lane, s_k, s_nsf, wib, small_k);
    if (lane == 0) discordant_out[a * n_cells + b] = (int)disc;
    __syncwarp();
  }
}


__global__ void RkendallPCP_B_different_blocks(
    const int* __restrict__ a_csc_p, const int* __restrict__ a_csc_i, const float* __restrict__ a_csc_x,
    const int* __restrict__ b_csc_p, const int* __restrict__ b_csc_i, const float* __restrict__ b_csc_x,
    int n_genes, int n_cells_a, int n_cells_b, int cap, int small_k, int* __restrict__ discordant_out)
{
  extern __shared__ float smem[];
  __shared__ int s_k[32];
  __shared__ int s_nsf[32];
  int lane = threadIdx.x;
  int wib = threadIdx.y;
  float* smem_warp = smem + (size_t)wib * 4 * cap;

  long long P = (long long)n_cells_a * n_cells_b;
  long long gw = (long long)blockIdx.x * blockDim.y + wib;
  long long stride = (long long)gridDim.x * blockDim.y;
  for (long long idx = gw; idx < P; idx += stride) {
    int a = (int)(idx / n_cells_b), b = (int)(idx % n_cells_b);
    long long disc = pcp_disc_warp(a_csc_i, a_csc_x, a_csc_p[a], a_csc_p[a + 1],
                                   b_csc_i, b_csc_x, b_csc_p[b], b_csc_p[b + 1],
                                   n_genes, smem_warp, cap, lane, s_k, s_nsf, wib, small_k);
    if (lane == 0) discordant_out[b * n_cells_a + a] = (int)disc;
    __syncwarp();
  }
}


// ---------- Per-pair dispatch kernels (1 warp / pair) ----------
// Each warp grid-strides over cell pairs. For every pair it decides INDIVIDUALLY
// (decoupled from the tile's global max_k): if the pair's union support fits the
// device-derived shared budget (kpair <= cap_smem), the whole warp does the
// cooperative shared merge-sort/inversion count (fast); otherwise lane 0 alone
// runs the O(k log k) merge in a per-warp global-scratch slab (cap_global sized,
// big enough for the heaviest pair). The branch is uniform across the warp, so
// there is no intra-warp divergence on the dispatch itself.
__global__ void RkendallPCP_Dispatch_same_block(
    const int* __restrict__ csc_p, const int* __restrict__ csc_i, const float* __restrict__ csc_x,
    int n_genes, int n_cells, int cap_smem, int small_k,
    float* scratch_arena, int cap_global, int* __restrict__ discordant_out,
    unsigned long long* counters)
{
  extern __shared__ float smem[];
  __shared__ int s_k[32];
  __shared__ int s_nsf[32];
  int lane = threadIdx.x;
  int wib  = threadIdx.y;
  int W    = blockDim.y;
  float* smem_warp = smem + (size_t)wib * 4 * cap_smem;
  long long gwid   = (long long)blockIdx.x * W + wib;   // unique warp id in grid
  float* my_scratch = scratch_arena ? (scratch_arena + (size_t)gwid * 4 * cap_global) : nullptr;

  long long P = (long long)n_cells * n_cells;
  long long stride = (long long)gridDim.x * W;
  for (long long idx = gwid; idx < P; idx += stride) {
    int a = (int)(idx / n_cells), b = (int)(idx % n_cells);
    if (a >= b) continue;
    int ia = csc_p[a], ea = csc_p[a + 1], ib = csc_p[b], eb = csc_p[b + 1];
    int kpair = (ea - ia) + (eb - ib);
    // One warp per pair either way. Pair fits the shared budget -> cooperative
    // merge in shared (fast). Otherwise the SAME warp-cooperative merge runs in
    // the per-warp global-scratch slab (cap_global): the rare heavy pair keeps
    // all 32 lanes (only the slower global memory), instead of collapsing to a
    // single lane -- the heavy tail is where Anime's time lives, so serialising
    // it would erase the win.
    long long disc;
    if (kpair <= cap_smem) {
      disc = pcp_disc_warp(csc_i, csc_x, ia, ea, csc_i, csc_x, ib, eb,
                           n_genes, smem_warp, cap_smem, lane, s_k, s_nsf, wib, small_k);
      if (counters && lane == 0) atomicAdd(&counters[0], 1ULL);   // shared ("B") pair
    } else {
      disc = pcp_disc_warp(csc_i, csc_x, ia, ea, csc_i, csc_x, ib, eb,
                           n_genes, my_scratch, cap_global, lane, s_k, s_nsf, wib, small_k);
      if (counters && lane == 0) atomicAdd(&counters[1], 1ULL);   // global fallback ("A") pair
    }
    if (lane == 0) discordant_out[a * n_cells + b] = (int)disc;
    __syncwarp();
  }
}


__global__ void RkendallPCP_Dispatch_different_blocks(
    const int* __restrict__ a_csc_p, const int* __restrict__ a_csc_i, const float* __restrict__ a_csc_x,
    const int* __restrict__ b_csc_p, const int* __restrict__ b_csc_i, const float* __restrict__ b_csc_x,
    int n_genes, int n_cells_a, int n_cells_b, int cap_smem, int small_k,
    float* scratch_arena, int cap_global, int* __restrict__ discordant_out,
    unsigned long long* counters)
{
  extern __shared__ float smem[];
  __shared__ int s_k[32];
  __shared__ int s_nsf[32];
  int lane = threadIdx.x;
  int wib  = threadIdx.y;
  int W    = blockDim.y;
  float* smem_warp = smem + (size_t)wib * 4 * cap_smem;
  long long gwid   = (long long)blockIdx.x * W + wib;
  float* my_scratch = scratch_arena ? (scratch_arena + (size_t)gwid * 4 * cap_global) : nullptr;

  long long P = (long long)n_cells_a * n_cells_b;
  long long stride = (long long)gridDim.x * W;
  for (long long idx = gwid; idx < P; idx += stride) {
    int a = (int)(idx / n_cells_b), b = (int)(idx % n_cells_b);
    int ia = a_csc_p[a], ea = a_csc_p[a + 1], ib = b_csc_p[b], eb = b_csc_p[b + 1];
    int kpair = (ea - ia) + (eb - ib);
    long long disc;
    if (kpair <= cap_smem) {
      disc = pcp_disc_warp(a_csc_i, a_csc_x, ia, ea, b_csc_i, b_csc_x, ib, eb,
                           n_genes, smem_warp, cap_smem, lane, s_k, s_nsf, wib, small_k);
      if (counters && lane == 0) atomicAdd(&counters[0], 1ULL);   // shared ("B") pair
    } else {
      disc = pcp_disc_warp(a_csc_i, a_csc_x, ia, ea, b_csc_i, b_csc_x, ib, eb,
                           n_genes, my_scratch, cap_global, lane, s_k, s_nsf, wib, small_k);
      if (counters && lane == 0) atomicAdd(&counters[1], 1ULL);   // global fallback ("A") pair
    }
    if (lane == 0) discordant_out[b * n_cells_a + a] = (int)disc;
    __syncwarp();
  }
}


// ---------- host helpers ----------
// pcp_max_col_nnz moved to pcp_dispatch.


// pcp_grid_for_scratch moved to pcp_dispatch.


// ---------- Variant A drivers ----------
extern "C" void matrix_Kendall_sparse_per_cell_pair_a_distance_same_block(
    int* csc_i_in, int* csc_p_in, double* csc_x_in,
    int*, int*, double*, double* result,
    int* num_rows, int* num_columns, int*, int* num_elements, int*)
{
  int n_genes = *num_rows, n_cells = *num_columns, nnz = *num_elements;
  std::vector<float> csc_x_f(nnz);
  for (int k = 0; k < nnz; ++k) csc_x_f[k] = (float)csc_x_in[k];
  int cap = 2 * pcp_max_col_nnz(csc_p_in, n_cells); if (cap < 1) cap = 1;

  int *d_csc_i, *d_csc_p, *d_discordant; float *d_csc_x, *d_scratch; double* d_result;
  cudaMalloc(&d_csc_i, nnz * sizeof(int));
  cudaMalloc(&d_csc_p, (n_cells + 1) * sizeof(int));
  cudaMalloc(&d_csc_x, nnz * sizeof(float));
  cudaMalloc(&d_discordant, (size_t)n_cells * n_cells * sizeof(int));
  cudaMalloc(&d_result, (size_t)n_cells * n_cells * sizeof(double));
  cudaMemcpy(d_csc_i, csc_i_in, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csc_p, csc_p_in, (n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csc_x, csc_x_f.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_discordant, 0, (size_t)n_cells * n_cells * sizeof(int));

  int nblocks, nthreads_total;
  pcp_grid_for_scratch((long long)n_cells * n_cells, cap, &nblocks, &nthreads_total);
  cudaMalloc(&d_scratch, (size_t)nthreads_total * 4 * cap * sizeof(float));

  RkendallPCP_A_same_block<<<nblocks, 128>>>(d_csc_p, d_csc_i, d_csc_x, n_genes, n_cells, cap, d_scratch, d_discordant);
  gpuErrchk(cudaPeekAtLastError());

  dim3 ft(16, 16), fb((n_cells + 15) / 16, (n_cells + 15) / 16);
  FinalizeKendallPerCellPair<<<fb, ft>>>(n_cells, n_genes, d_discordant, d_result);
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(result, d_result, (size_t)n_cells * n_cells * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_csc_i); cudaFree(d_csc_p); cudaFree(d_csc_x);
  cudaFree(d_discordant); cudaFree(d_result); cudaFree(d_scratch);
}


extern "C" void matrix_Kendall_sparse_per_cell_pair_a_distance_different_blocks(
    int* a_csc_i_in, int* a_csc_p_in, double* a_csc_x_in,
    int* b_csc_i_in, int* b_csc_p_in, double* b_csc_x_in, double* result,
    int* num_rows, int* num_columns, int* num_columns_b, int* num_elements_a, int* num_elements_b)
{
  int n_genes = *num_rows, n_cells_a = *num_columns, n_cells_b = *num_columns_b;
  int nnz_a = *num_elements_a, nnz_b = *num_elements_b;
  std::vector<float> ax(nnz_a), bx(nnz_b);
  for (int k = 0; k < nnz_a; ++k) ax[k] = (float)a_csc_x_in[k];
  for (int k = 0; k < nnz_b; ++k) bx[k] = (float)b_csc_x_in[k];
  int cap = pcp_max_col_nnz(a_csc_p_in, n_cells_a) + pcp_max_col_nnz(b_csc_p_in, n_cells_b);
  if (cap < 1) cap = 1;

  int *d_ai, *d_ap, *d_bi, *d_bp, *d_disc; float *d_ax, *d_bx, *d_scratch;
  cudaMalloc(&d_ai, nnz_a * sizeof(int)); cudaMalloc(&d_ap, (n_cells_a + 1) * sizeof(int)); cudaMalloc(&d_ax, nnz_a * sizeof(float));
  cudaMalloc(&d_bi, nnz_b * sizeof(int)); cudaMalloc(&d_bp, (n_cells_b + 1) * sizeof(int)); cudaMalloc(&d_bx, nnz_b * sizeof(float));
  cudaMalloc(&d_disc, (size_t)n_cells_a * n_cells_b * sizeof(int));
  cudaMemcpy(d_ai, a_csc_i_in, nnz_a * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ap, a_csc_p_in, (n_cells_a + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ax, ax.data(), nnz_a * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bi, b_csc_i_in, nnz_b * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bp, b_csc_p_in, (n_cells_b + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bx, bx.data(), nnz_b * sizeof(float), cudaMemcpyHostToDevice);

  int nblocks, nthreads_total;
  pcp_grid_for_scratch((long long)n_cells_a * n_cells_b, cap, &nblocks, &nthreads_total);
  cudaMalloc(&d_scratch, (size_t)nthreads_total * 4 * cap * sizeof(float));

  RkendallPCP_A_different_blocks<<<nblocks, 128>>>(d_ap, d_ai, d_ax, d_bp, d_bi, d_bx,
      n_genes, n_cells_a, n_cells_b, cap, d_scratch, d_disc);
  gpuErrchk(cudaPeekAtLastError());

  std::vector<int> h_disc((size_t)n_cells_a * n_cells_b);
  cudaMemcpy(h_disc.data(), d_disc, (size_t)n_cells_a * n_cells_b * sizeof(int), cudaMemcpyDeviceToHost);
  double norm = (double)n_genes * (n_genes - 1);
  for (size_t i = 0; i < (size_t)n_cells_a * n_cells_b; ++i) result[i] = (double)h_disc[i] * 2.0 / norm;

  cudaFree(d_ai); cudaFree(d_ap); cudaFree(d_ax);
  cudaFree(d_bi); cudaFree(d_bp); cudaFree(d_bx);
  cudaFree(d_disc); cudaFree(d_scratch);
}


// ---------- Hybrid drivers ----------
extern "C" void matrix_Kendall_sparse_per_cell_pair_hybrid_distance_same_block(
    int* csc_i_in, int* csc_p_in, double* csc_x_in,
    int*, int*, double*, double* result,
    int* num_rows, int* num_columns, int*, int* num_elements, int*)
{
  int n_genes = *num_rows, n_cells = *num_columns, nnz = *num_elements;
  std::vector<float> csc_x_f(nnz);
  for (int k = 0; k < nnz; ++k) csc_x_f[k] = (float)csc_x_in[k];
  int cap = 2 * pcp_max_col_nnz(csc_p_in, n_cells); if (cap < 1) cap = 1;

  int *d_csc_i, *d_csc_p, *d_discordant; float *d_csc_x, *d_scratch; double* d_result;
  cudaMalloc(&d_csc_i, nnz * sizeof(int));
  cudaMalloc(&d_csc_p, (n_cells + 1) * sizeof(int));
  cudaMalloc(&d_csc_x, nnz * sizeof(float));
  cudaMalloc(&d_discordant, (size_t)n_cells * n_cells * sizeof(int));
  cudaMalloc(&d_result, (size_t)n_cells * n_cells * sizeof(double));
  cudaMemcpy(d_csc_i, csc_i_in, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csc_p, csc_p_in, (n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csc_x, csc_x_f.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_discordant, 0, (size_t)n_cells * n_cells * sizeof(int));

  int nblocks, nthreads_total;
  pcp_grid_for_scratch((long long)n_cells * n_cells, cap, &nblocks, &nthreads_total);
  cudaMalloc(&d_scratch, (size_t)nthreads_total * 4 * cap * sizeof(float));

  RkendallPCP_Hybrid_same_block<<<nblocks, 128>>>(d_csc_p, d_csc_i, d_csc_x, n_genes, n_cells,
      cap, PCP_HYBRID_THRESHOLD, d_scratch, d_discordant);
  gpuErrchk(cudaPeekAtLastError());

  dim3 ft(16, 16), fb((n_cells + 15) / 16, (n_cells + 15) / 16);
  FinalizeKendallPerCellPair<<<fb, ft>>>(n_cells, n_genes, d_discordant, d_result);
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(result, d_result, (size_t)n_cells * n_cells * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_csc_i); cudaFree(d_csc_p); cudaFree(d_csc_x);
  cudaFree(d_discordant); cudaFree(d_result); cudaFree(d_scratch);
}


extern "C" void matrix_Kendall_sparse_per_cell_pair_hybrid_distance_different_blocks(
    int* a_csc_i_in, int* a_csc_p_in, double* a_csc_x_in,
    int* b_csc_i_in, int* b_csc_p_in, double* b_csc_x_in, double* result,
    int* num_rows, int* num_columns, int* num_columns_b, int* num_elements_a, int* num_elements_b)
{
  int n_genes = *num_rows, n_cells_a = *num_columns, n_cells_b = *num_columns_b;
  int nnz_a = *num_elements_a, nnz_b = *num_elements_b;
  std::vector<float> ax(nnz_a), bx(nnz_b);
  for (int k = 0; k < nnz_a; ++k) ax[k] = (float)a_csc_x_in[k];
  for (int k = 0; k < nnz_b; ++k) bx[k] = (float)b_csc_x_in[k];
  int cap = pcp_max_col_nnz(a_csc_p_in, n_cells_a) + pcp_max_col_nnz(b_csc_p_in, n_cells_b);
  if (cap < 1) cap = 1;

  int *d_ai, *d_ap, *d_bi, *d_bp, *d_disc; float *d_ax, *d_bx, *d_scratch;
  cudaMalloc(&d_ai, nnz_a * sizeof(int)); cudaMalloc(&d_ap, (n_cells_a + 1) * sizeof(int)); cudaMalloc(&d_ax, nnz_a * sizeof(float));
  cudaMalloc(&d_bi, nnz_b * sizeof(int)); cudaMalloc(&d_bp, (n_cells_b + 1) * sizeof(int)); cudaMalloc(&d_bx, nnz_b * sizeof(float));
  cudaMalloc(&d_disc, (size_t)n_cells_a * n_cells_b * sizeof(int));
  cudaMemcpy(d_ai, a_csc_i_in, nnz_a * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ap, a_csc_p_in, (n_cells_a + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ax, ax.data(), nnz_a * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bi, b_csc_i_in, nnz_b * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bp, b_csc_p_in, (n_cells_b + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bx, bx.data(), nnz_b * sizeof(float), cudaMemcpyHostToDevice);

  int nblocks, nthreads_total;
  pcp_grid_for_scratch((long long)n_cells_a * n_cells_b, cap, &nblocks, &nthreads_total);
  cudaMalloc(&d_scratch, (size_t)nthreads_total * 4 * cap * sizeof(float));

  RkendallPCP_Hybrid_different_blocks<<<nblocks, 128>>>(d_ap, d_ai, d_ax, d_bp, d_bi, d_bx,
      n_genes, n_cells_a, n_cells_b, cap, PCP_HYBRID_THRESHOLD, d_scratch, d_disc);
  gpuErrchk(cudaPeekAtLastError());

  std::vector<int> h_disc((size_t)n_cells_a * n_cells_b);
  cudaMemcpy(h_disc.data(), d_disc, (size_t)n_cells_a * n_cells_b * sizeof(int), cudaMemcpyDeviceToHost);
  double norm = (double)n_genes * (n_genes - 1);
  for (size_t i = 0; i < (size_t)n_cells_a * n_cells_b; ++i) result[i] = (double)h_disc[i] * 2.0 / norm;

  cudaFree(d_ai); cudaFree(d_ap); cudaFree(d_ax);
  cudaFree(d_bi); cudaFree(d_bp); cudaFree(d_bx);
  cudaFree(d_disc); cudaFree(d_scratch);
}


// ---------- Variant B drivers ----------
// pcp_warps_per_block moved to pcp_dispatch.


static void pcp_B_same_block_impl(
    int* csc_i_in, int* csc_p_in, double* csc_x_in, double* result,
    int* num_rows, int* num_columns, int* num_elements, int small_k)
{
  int n_genes = *num_rows, n_cells = *num_columns, nnz = *num_elements;
  std::vector<float> csc_x_f(nnz);
  for (int k = 0; k < nnz; ++k) csc_x_f[k] = (float)csc_x_in[k];
  int cap = 2 * pcp_max_col_nnz(csc_p_in, n_cells); if (cap < 1) cap = 1;

  int *d_csc_i, *d_csc_p, *d_discordant; float *d_csc_x; double* d_result;
  cudaMalloc(&d_csc_i, nnz * sizeof(int));
  cudaMalloc(&d_csc_p, (n_cells + 1) * sizeof(int));
  cudaMalloc(&d_csc_x, nnz * sizeof(float));
  cudaMalloc(&d_discordant, (size_t)n_cells * n_cells * sizeof(int));
  cudaMalloc(&d_result, (size_t)n_cells * n_cells * sizeof(double));
  cudaMemcpy(d_csc_i, csc_i_in, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csc_p, csc_p_in, (n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csc_x, csc_x_f.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_discordant, 0, (size_t)n_cells * n_cells * sizeof(int));

  int W = pcp_warps_per_block(cap);
  size_t smem = (size_t)W * 4 * cap * sizeof(float);
  cudaFuncSetAttribute(RkendallPCP_B_same_block, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  long long P = (long long)n_cells * n_cells;
  int blocks = (int)((P + W - 1) / W); if (blocks > 65535) blocks = 65535; if (blocks < 1) blocks = 1;
  dim3 threads(32, W);

  RkendallPCP_B_same_block<<<blocks, threads, smem>>>(d_csc_p, d_csc_i, d_csc_x, n_genes, n_cells, cap, small_k, d_discordant);
  gpuErrchk(cudaPeekAtLastError());

  dim3 ft(16, 16), fb((n_cells + 15) / 16, (n_cells + 15) / 16);
  FinalizeKendallPerCellPair<<<fb, ft>>>(n_cells, n_genes, d_discordant, d_result);
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(result, d_result, (size_t)n_cells * n_cells * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_csc_i); cudaFree(d_csc_p); cudaFree(d_csc_x);
  cudaFree(d_discordant); cudaFree(d_result);
}


extern "C" void matrix_Kendall_sparse_per_cell_pair_b_distance_same_block(
    int* csc_i_in, int* csc_p_in, double* csc_x_in,
    int*, int*, double*, double* result,
    int* num_rows, int* num_columns, int*, int* num_elements, int*)
{ pcp_B_same_block_impl(csc_i_in, csc_p_in, csc_x_in, result, num_rows, num_columns, num_elements, PCP_B_SMALL); }


extern "C" void matrix_Kendall_sparse_per_cell_pair_b0_distance_same_block(
    int* csc_i_in, int* csc_p_in, double* csc_x_in,
    int*, int*, double*, double* result,
    int* num_rows, int* num_columns, int*, int* num_elements, int*)
{ pcp_B_same_block_impl(csc_i_in, csc_p_in, csc_x_in, result, num_rows, num_columns, num_elements, 0); }


static void pcp_B_diff_impl(
    int* a_csc_i_in, int* a_csc_p_in, double* a_csc_x_in,
    int* b_csc_i_in, int* b_csc_p_in, double* b_csc_x_in, double* result,
    int* num_rows, int* num_columns, int* num_columns_b, int* num_elements_a, int* num_elements_b, int small_k)
{
  int n_genes = *num_rows, n_cells_a = *num_columns, n_cells_b = *num_columns_b;
  int nnz_a = *num_elements_a, nnz_b = *num_elements_b;
  std::vector<float> ax(nnz_a), bx(nnz_b);
  for (int k = 0; k < nnz_a; ++k) ax[k] = (float)a_csc_x_in[k];
  for (int k = 0; k < nnz_b; ++k) bx[k] = (float)b_csc_x_in[k];
  int cap = pcp_max_col_nnz(a_csc_p_in, n_cells_a) + pcp_max_col_nnz(b_csc_p_in, n_cells_b);
  if (cap < 1) cap = 1;

  int *d_ai, *d_ap, *d_bi, *d_bp, *d_disc; float *d_ax, *d_bx;
  cudaMalloc(&d_ai, nnz_a * sizeof(int)); cudaMalloc(&d_ap, (n_cells_a + 1) * sizeof(int)); cudaMalloc(&d_ax, nnz_a * sizeof(float));
  cudaMalloc(&d_bi, nnz_b * sizeof(int)); cudaMalloc(&d_bp, (n_cells_b + 1) * sizeof(int)); cudaMalloc(&d_bx, nnz_b * sizeof(float));
  cudaMalloc(&d_disc, (size_t)n_cells_a * n_cells_b * sizeof(int));
  cudaMemcpy(d_ai, a_csc_i_in, nnz_a * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ap, a_csc_p_in, (n_cells_a + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ax, ax.data(), nnz_a * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bi, b_csc_i_in, nnz_b * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bp, b_csc_p_in, (n_cells_b + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bx, bx.data(), nnz_b * sizeof(float), cudaMemcpyHostToDevice);

  int W = pcp_warps_per_block(cap);
  size_t smem = (size_t)W * 4 * cap * sizeof(float);
  cudaFuncSetAttribute(RkendallPCP_B_different_blocks, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  long long P = (long long)n_cells_a * n_cells_b;
  int blocks = (int)((P + W - 1) / W); if (blocks > 65535) blocks = 65535; if (blocks < 1) blocks = 1;
  dim3 threads(32, W);

  RkendallPCP_B_different_blocks<<<blocks, threads, smem>>>(d_ap, d_ai, d_ax, d_bp, d_bi, d_bx,
      n_genes, n_cells_a, n_cells_b, cap, small_k, d_disc);
  gpuErrchk(cudaPeekAtLastError());

  std::vector<int> h_disc((size_t)n_cells_a * n_cells_b);
  cudaMemcpy(h_disc.data(), d_disc, (size_t)n_cells_a * n_cells_b * sizeof(int), cudaMemcpyDeviceToHost);
  double norm = (double)n_genes * (n_genes - 1);
  for (size_t i = 0; i < (size_t)n_cells_a * n_cells_b; ++i) result[i] = (double)h_disc[i] * 2.0 / norm;

  cudaFree(d_ai); cudaFree(d_ap); cudaFree(d_ax);
  cudaFree(d_bi); cudaFree(d_bp); cudaFree(d_bx);
  cudaFree(d_disc);
}


extern "C" void matrix_Kendall_sparse_per_cell_pair_b_distance_different_blocks(
    int* a_csc_i_in, int* a_csc_p_in, double* a_csc_x_in,
    int* b_csc_i_in, int* b_csc_p_in, double* b_csc_x_in, double* result,
    int* num_rows, int* num_columns, int* num_columns_b, int* num_elements_a, int* num_elements_b)
{ pcp_B_diff_impl(a_csc_i_in, a_csc_p_in, a_csc_x_in, b_csc_i_in, b_csc_p_in, b_csc_x_in, result,
                  num_rows, num_columns, num_columns_b, num_elements_a, num_elements_b, PCP_B_SMALL); }


extern "C" void matrix_Kendall_sparse_per_cell_pair_b0_distance_different_blocks(
    int* a_csc_i_in, int* a_csc_p_in, double* a_csc_x_in,
    int* b_csc_i_in, int* b_csc_p_in, double* b_csc_x_in, double* result,
    int* num_rows, int* num_columns, int* num_columns_b, int* num_elements_a, int* num_elements_b)
{ pcp_B_diff_impl(a_csc_i_in, a_csc_p_in, a_csc_x_in, b_csc_i_in, b_csc_p_in, b_csc_x_in, result,
                  num_rows, num_columns, num_columns_b, num_elements_a, num_elements_b, 0); }


// ---------- Per-pair dispatch drivers ----------
// Device-queried opt-in shared budget (cached). NOT hardcoded 96 KiB: query
// cudaDevAttrMaxSharedMemoryPerBlockOptin so the kernel uses the real per-block
// shared ceiling of whatever GPU it runs on (Turing 64 KiB ... B200 228 KiB).
// pcp_device_smem_budget moved to pcp_dispatch.


// PcpDispatchCfg + pcp_dispatch_cfg (dynamic-occupancy launch config) moved to
// pcp_dispatch.cu / pcp_dispatch.cuh. The dispatch KERNELS stay here.


extern "C" void matrix_Kendall_sparse_per_cell_pair_dispatch_distance_same_block(
    int* csc_i_in, int* csc_p_in, double* csc_x_in,
    int*, int*, double*, double* result,
    int* num_rows, int* num_columns, int*, int* num_elements, int*)
{
  int n_genes = *num_rows, n_cells = *num_columns, nnz = *num_elements;
  std::vector<float> csc_x_f(nnz);
  for (int k = 0; k < nnz; ++k) csc_x_f[k] = (float)csc_x_in[k];
  int cap_global = 2 * pcp_max_col_nnz(csc_p_in, n_cells); if (cap_global < 1) cap_global = 1;
  PcpDispatchCfg cfg = pcp_dispatch_cfg(cap_global);

  int *d_csc_i, *d_csc_p, *d_discordant; float *d_csc_x; double* d_result;
  cudaMalloc(&d_csc_i, nnz * sizeof(int));
  cudaMalloc(&d_csc_p, (n_cells + 1) * sizeof(int));
  cudaMalloc(&d_csc_x, nnz * sizeof(float));
  cudaMalloc(&d_discordant, (size_t)n_cells * n_cells * sizeof(int));
  cudaMalloc(&d_result, (size_t)n_cells * n_cells * sizeof(double));
  cudaMemcpy(d_csc_i, csc_i_in, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csc_p, csc_p_in, (n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csc_x, csc_x_f.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_discordant, 0, (size_t)n_cells * n_cells * sizeof(int));

  long long P = (long long)n_cells * n_cells;
  long long cover_blocks = (P + cfg.W - 1) / cfg.W; if (cover_blocks < 1) cover_blocks = 1;
  float* d_scratch = nullptr;
  long long blocks;
  if (cfg.needs_fallback) {
    // bound grid warps by a global-scratch budget; each grid-warp owns a slab
    // big enough for the heaviest pair (cap_global). Heavy pairs are rare, so a
    // small grid grid-strides over them fine.
    const size_t budget = (size_t)2 * 1024 * 1024 * 1024;
    size_t per_warp = (size_t)4 * cap_global * sizeof(float);
    long long want_warps = (long long)(budget / (per_warp ? per_warp : 1));
    if (want_warps < cfg.W) want_warps = cfg.W;
    long long want_blocks = (want_warps + cfg.W - 1) / cfg.W;
    blocks = want_blocks < cover_blocks ? want_blocks : cover_blocks;
    if (blocks > 65535) blocks = 65535; if (blocks < 1) blocks = 1;
    cudaMalloc(&d_scratch, (size_t)blocks * cfg.W * 4 * cap_global * sizeof(float));
  } else {
    blocks = cover_blocks; if (blocks > 65535) blocks = 65535;
  }

  bool do_log = getenv("HOBO_PCP_LOG") != nullptr;
  unsigned long long* d_counters = nullptr;
  if (do_log) { cudaMalloc(&d_counters, 2 * sizeof(unsigned long long));
                cudaMemset(d_counters, 0, 2 * sizeof(unsigned long long)); }

  cudaFuncSetAttribute(RkendallPCP_Dispatch_same_block,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cfg.smem_bytes);
  dim3 threads(32, cfg.W);
  RkendallPCP_Dispatch_same_block<<<(int)blocks, threads, cfg.smem_bytes>>>(
      d_csc_p, d_csc_i, d_csc_x, n_genes, n_cells, cfg.cap_smem, PCP_B_SMALL,
      d_scratch, cap_global, d_discordant, d_counters);
  gpuErrchk(cudaPeekAtLastError());

  if (do_log) {
    unsigned long long h_c[2] = {0, 0};
    cudaMemcpy(h_c, d_counters, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_counters);
    unsigned long long tot = h_c[0] + h_c[1];
    printf("  [pcp SAME %dx%d cap_smem=%d cap_global=%d W=%d fb=%d] shared(B)=%llu fallback(A)=%llu (%.2f%% A)\n",
           n_cells, n_cells, cfg.cap_smem, cap_global, cfg.W, (int)cfg.needs_fallback,
           h_c[0], h_c[1], tot ? 100.0 * h_c[1] / tot : 0.0);
    fflush(stdout);
  }

  dim3 ft(16, 16), fb((n_cells + 15) / 16, (n_cells + 15) / 16);
  FinalizeKendallPerCellPair<<<fb, ft>>>(n_cells, n_genes, d_discordant, d_result);
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(result, d_result, (size_t)n_cells * n_cells * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_csc_i); cudaFree(d_csc_p); cudaFree(d_csc_x);
  cudaFree(d_discordant); cudaFree(d_result);
  if (d_scratch) cudaFree(d_scratch);
}


extern "C" void matrix_Kendall_sparse_per_cell_pair_dispatch_distance_different_blocks(
    int* a_csc_i_in, int* a_csc_p_in, double* a_csc_x_in,
    int* b_csc_i_in, int* b_csc_p_in, double* b_csc_x_in, double* result,
    int* num_rows, int* num_columns, int* num_columns_b, int* num_elements_a, int* num_elements_b)
{
  int n_genes = *num_rows, n_cells_a = *num_columns, n_cells_b = *num_columns_b;
  int nnz_a = *num_elements_a, nnz_b = *num_elements_b;
  std::vector<float> ax(nnz_a), bx(nnz_b);
  for (int k = 0; k < nnz_a; ++k) ax[k] = (float)a_csc_x_in[k];
  for (int k = 0; k < nnz_b; ++k) bx[k] = (float)b_csc_x_in[k];
  int cap_global = pcp_max_col_nnz(a_csc_p_in, n_cells_a) + pcp_max_col_nnz(b_csc_p_in, n_cells_b);
  if (cap_global < 1) cap_global = 1;
  PcpDispatchCfg cfg = pcp_dispatch_cfg(cap_global);

  int *d_ai, *d_ap, *d_bi, *d_bp, *d_disc; float *d_ax, *d_bx;
  cudaMalloc(&d_ai, nnz_a * sizeof(int)); cudaMalloc(&d_ap, (n_cells_a + 1) * sizeof(int)); cudaMalloc(&d_ax, nnz_a * sizeof(float));
  cudaMalloc(&d_bi, nnz_b * sizeof(int)); cudaMalloc(&d_bp, (n_cells_b + 1) * sizeof(int)); cudaMalloc(&d_bx, nnz_b * sizeof(float));
  cudaMalloc(&d_disc, (size_t)n_cells_a * n_cells_b * sizeof(int));
  cudaMemcpy(d_ai, a_csc_i_in, nnz_a * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ap, a_csc_p_in, (n_cells_a + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ax, ax.data(), nnz_a * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bi, b_csc_i_in, nnz_b * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bp, b_csc_p_in, (n_cells_b + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bx, bx.data(), nnz_b * sizeof(float), cudaMemcpyHostToDevice);

  long long P = (long long)n_cells_a * n_cells_b;
  long long cover_blocks = (P + cfg.W - 1) / cfg.W; if (cover_blocks < 1) cover_blocks = 1;
  float* d_scratch = nullptr;
  long long blocks;
  if (cfg.needs_fallback) {
    const size_t budget = (size_t)2 * 1024 * 1024 * 1024;
    size_t per_warp = (size_t)4 * cap_global * sizeof(float);
    long long want_warps = (long long)(budget / (per_warp ? per_warp : 1));
    if (want_warps < cfg.W) want_warps = cfg.W;
    long long want_blocks = (want_warps + cfg.W - 1) / cfg.W;
    blocks = want_blocks < cover_blocks ? want_blocks : cover_blocks;
    if (blocks > 65535) blocks = 65535; if (blocks < 1) blocks = 1;
    cudaMalloc(&d_scratch, (size_t)blocks * cfg.W * 4 * cap_global * sizeof(float));
  } else {
    blocks = cover_blocks; if (blocks > 65535) blocks = 65535;
  }

  bool do_log = getenv("HOBO_PCP_LOG") != nullptr;
  unsigned long long* d_counters = nullptr;
  if (do_log) { cudaMalloc(&d_counters, 2 * sizeof(unsigned long long));
                cudaMemset(d_counters, 0, 2 * sizeof(unsigned long long)); }

  cudaFuncSetAttribute(RkendallPCP_Dispatch_different_blocks,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cfg.smem_bytes);
  dim3 threads(32, cfg.W);
  RkendallPCP_Dispatch_different_blocks<<<(int)blocks, threads, cfg.smem_bytes>>>(
      d_ap, d_ai, d_ax, d_bp, d_bi, d_bx, n_genes, n_cells_a, n_cells_b,
      cfg.cap_smem, PCP_B_SMALL, d_scratch, cap_global, d_disc, d_counters);
  gpuErrchk(cudaPeekAtLastError());

  if (do_log) {
    unsigned long long h_c[2] = {0, 0};
    cudaMemcpy(h_c, d_counters, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_counters);
    unsigned long long tot = h_c[0] + h_c[1];
    printf("  [pcp DIFF %dx%d cap_smem=%d cap_global=%d W=%d fb=%d] shared(B)=%llu fallback(A)=%llu (%.2f%% A)\n",
           n_cells_a, n_cells_b, cfg.cap_smem, cap_global, cfg.W, (int)cfg.needs_fallback,
           h_c[0], h_c[1], tot ? 100.0 * h_c[1] / tot : 0.0);
    fflush(stdout);
  }

  std::vector<int> h_disc((size_t)n_cells_a * n_cells_b);
  cudaMemcpy(h_disc.data(), d_disc, (size_t)n_cells_a * n_cells_b * sizeof(int), cudaMemcpyDeviceToHost);
  double norm = (double)n_genes * (n_genes - 1);
  for (size_t i = 0; i < (size_t)n_cells_a * n_cells_b; ++i) result[i] = (double)h_disc[i] * 2.0 / norm;

  cudaFree(d_ai); cudaFree(d_ap); cudaFree(d_ax);
  cudaFree(d_bi); cudaFree(d_bp); cudaFree(d_bx);
  cudaFree(d_disc);
  if (d_scratch) cudaFree(d_scratch);
}


// Kendall: warp-per-pair (variant B, shared mem) or thread-per-pair (variant A,
// global scratch) — identical dispatch to matrix_Kendall_distance_same_block.
// Writes discordant counts (upper triangle) into d_disc.
void pc_kendall_same_block_device(const float* d_A, int N, int M,
                                         int* d_disc) {
    PcKernelTimer _kt;
    cudaMemset(d_disc, 0, (size_t)M * M * sizeof(int));
    dk_run_same(d_A, N, M, d_disc);   // variant B (shared) or G (warp-global); device-budget, W<=16
}
