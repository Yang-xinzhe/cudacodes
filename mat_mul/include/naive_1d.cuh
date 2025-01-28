#ifndef NAIVE_1D
#define NAIVE_1D

__global__ void naive_xgemm_1d_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K);

float run_naive_xgemm_1d_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K);

#endif