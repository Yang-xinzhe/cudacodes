#ifndef TILED_COARSE
#define TILED_COARSE

#define BM 64
#define BK 8
#define BN 64
#define COARSE_FACTOR 8

__global__ void tiled_coarse_xgemm_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K);

float run_tiled_coarse_xgemm_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K);

#endif