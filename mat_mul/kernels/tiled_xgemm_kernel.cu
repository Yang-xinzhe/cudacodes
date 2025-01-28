#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "tiled_fine.cuh"

__global__ void tile_xgemm_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float a_smem[TILE_SIZE][TILE_SIZE];
    __shared__ float b_smem[TILE_SIZE][TILE_SIZE];

    float sum = 0.f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for(int tile = 0 ; tile < num_tiles ; ++tile) { 
        int offset = tile * TILE_SIZE;

        // Loading Matrix A tiles A = M x K
        if(row < M && (offset + threadIdx.x) < K) {
            a_smem[threadIdx.y][threadIdx.x] = A_gpu[row * K + (offset + threadIdx.x)];
        } else {
            a_smem[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Loading Matrix B tiles B = K x N
        if((offset + threadIdx.y) < K && col < N) {
            b_smem[threadIdx.y][threadIdx.x] = B_gpu[(offset + threadIdx.y) * N + col];
        } else {
            b_smem[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for(int i = 0 ; i < TILE_SIZE ; i++) {
            sum += a_smem[threadIdx.y][i] * b_smem[i][threadIdx.x];
        }

        __syncthreads();
    }
    if(row < M && col < N) {
        C_gpu[row * N + col] = sum;
    }
}

float run_tile_xgemm_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K) {
    dim3 block_size(TILE_SIZE / 2, TILE_SIZE / 2); // 256 threads twice faster than 512 threads
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float ms = 0.0f;

    int shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

    cudaFuncSetCacheConfig(tile_xgemm_kernel, cudaFuncCachePreferShared);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    tile_xgemm_kernel<<<grid_size, block_size, shared_mem_size>>>(A_gpu, B_gpu, C_gpu, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, start, end);
    printf(">> tile_xgemm_kernel execute time: %.3f ms\n", ms);
    return ms;
}