#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "naive_2d.cuh"

__global__ void naive_xgemm_2d_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < M && col < N) {
        float sum = 0.0f;
        for(int k = 0 ; k < K ; ++k) {
            sum += A_gpu[row * K + k] * B_gpu[k * N + col];
        }
        C_gpu[row * N + col] = sum;
    }
}

float run_naive_xgemm_2d_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K){
    const int BLOCK_DIM = 16;
    dim3 block_size(BLOCK_DIM, BLOCK_DIM); // 16x16=256 threads
    dim3 grid_size(((N + block_size.x - 1) / block_size.x), (M + block_size.y - 1) / block_size.y);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float ms = 0.0f;

    cudaFuncSetCacheConfig(naive_xgemm_2d_kernel, cudaFuncCachePreferL1);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    naive_xgemm_2d_kernel<<<grid_size, block_size>>>(A_gpu, B_gpu, C_gpu, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, start, end);
    printf(">> naive_xgemm_2d_kernel execute time: %.3f ms\n", ms);
    return ms;
}