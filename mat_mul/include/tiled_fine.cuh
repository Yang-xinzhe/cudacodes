#ifndef TILED_FINE
#define TILED_FINE

#define TILE_SIZE 32

__global__ void tile_xgemm_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K);

float run_tile_xgemm_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K);

#endif