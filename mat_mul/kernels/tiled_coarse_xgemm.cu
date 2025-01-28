#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "tiled_coarse.cuh"

__global__ void tiled_coarse_xgemm_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K) {
    int bidy = blockIdx.y;
    int bidx = blockIdx.x;

    // for loading B's tiles
    int b_tidy = threadIdx.x / BN;
    int b_tidx = threadIdx.x % BN;

    // for loading A's tiles
    int a_tidy = threadIdx.x / BK;
    int a_tidx = threadIdx.x % BK;

    // working on C[row, col]
    int row = bidy * BM + (b_tidy * COARSE_FACTOR);
    int col = bidx * BN + b_tidx;

    __shared__ float a_smem[BM * BK];
    __shared__ float b_smem[BK * BN];

    float sum[COARSE_FACTOR] = {0.f};

    for(int tile = 0 ; tile < K ; tile += BK) {
        if((bidy * BM + a_tidy) < M && (tile + a_tidx) < K) {
            a_smem[a_tidy * BK + a_tidx] = A_gpu[(bidy * BM + a_tidy) * K + (tile + a_tidx)];
        } else {
            a_smem[a_tidy * BK + a_tidx] = 0.f;
        }

        if((tile + b_tidy) < K && (bidx * BN + b_tidx) < N) {
            b_smem[b_tidy * BN + b_tidx] = B_gpu[(tile + b_tidy) * N + (bidx * BN + b_tidx)];
        } else {
            b_smem[b_tidy * BN + b_tidx] = 0.f;
        }

        __syncthreads();

        for(int k = 0 ; k < BK ; ++k) {
            float b_reg = b_smem[k * BN + bidx];
            for(int c = 0 ; c < COARSE_FACTOR ; ++c) {
                sum[c] += a_smem[((b_tidy * COARSE_FACTOR + c) * BK + k)] * b_reg;
            }
        }
        __syncthreads();
    }
    for(int c = 0 ; c < COARSE_FACTOR ; ++c) {
        if((row + c) < M && col < N) {
            C_gpu[(row + c) * N + col] = sum[c];
        }
    }
}   

float run_tiled_coarse_xgemm_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K) {
    dim3 block_size(BM * BN / COARSE_FACTOR / 2); // TODO: 256 threads faster than 512 threads
    dim3 grid_size((N + BN -1) / BN, (M + BM - 1) / BM);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float ms = 0.0f;
    
    int shared_mem_size = (BM * BK + BK * BN) * sizeof(float);

    cudaDeviceSynchronize();
    cudaFuncSetCacheConfig(tiled_coarse_xgemm_kernel, cudaFuncCachePreferShared);
    cudaEventRecord(start);
    tiled_coarse_xgemm_kernel<<<grid_size, block_size, shared_mem_size>>>(A_gpu, B_gpu, C_gpu, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, start, end);
    printf(">> tiled_coarse_xgemm_kernel execute time: %.3f ms\n", ms);
    return ms;
}