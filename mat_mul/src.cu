#include <assert.h>
#include <stdbool.h>
#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "naive_1d.cuh"
#include "naive_2d.cuh"
#include "tiled_fine.cuh"
#include "tiled_coarse.cuh"

// Make sure it's normally distributed (正态分布)
float random_normal_clamp(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min) {
        return min;
    } 
    if (num > max) {
        return max;
    }
    return num;
}

/*Benchmark a kernel against torch.matmul for different size*/
void benchmark_kernel_for_size(int minDim, int maxDim) {
    const int NUM_RUNS = 10;
    
    // Results
    FILE *naive_1d_file = fopen("benchmarks/naive_1d.csv", "w");
    FILE *naive_2d_file = fopen("benchmarks/naive_2d.csv", "w");
    FILE *tiled_fine_file = fopen("benchmarks/tiled_fine.csv", "w");
    FILE *tiled_coarse_file = fopen("benchmarks/tiled_coarse.csv", "w");

    fprintf(naive_1d_file, "matrix_size,time_ms\n");
    fprintf(naive_2d_file, "matrix_size,time_ms\n");
    fprintf(tiled_fine_file, "matrix_size,time_ms\n");
    fprintf(tiled_coarse_file, "matrix_size,time_ms\n");

    for(int dim = minDim ; dim <= maxDim ; dim <<= 1) {
        printf("\n>> Testing matrix size: %d x %d\n", dim, dim);
        int M = dim, N = dim, K = dim;
        int a_size = M * K;
        int b_size = K * N;
        int c_size = M * N;

        // 主机内存分配和初始化
        float *A = (float *)malloc(a_size * sizeof(float));
        float *B = (float *)malloc(b_size * sizeof(float));
        float *C = (float *)malloc(c_size * sizeof(float));
        float *C_ref = (float *)malloc(c_size * sizeof(float));
        
        // init matrix with random values
        for(int i = 0 ; i < a_size ; ++i) {
            A[i] = random_normal_clamp(-10, 10);
        }
        for(int i = 0 ; i < b_size ; ++i) {
            B[i] = random_normal_clamp(-10, 10);
        }
        for(int i = 0 ; i < c_size ; ++i) {
            C[i] = 0.f;
        }

        float *A_gpu, *B_gpu, *C_gpu;
        cudaMalloc(&A_gpu, sizeof(float) * a_size);
        cudaMalloc(&B_gpu, sizeof(float) * b_size);
        cudaMalloc(&C_gpu, sizeof(float) * c_size);

        cudaMemcpy(A_gpu, A, sizeof(float) * a_size, cudaMemcpyHostToDevice);
        cudaMemcpy(B_gpu, B, sizeof(float) * b_size, cudaMemcpyHostToDevice);

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        float ms = 0.0f;
        // Warm up
        cudaFuncSetCacheConfig(nullptr, cudaFuncCachePreferL1);
        cudaDeviceSynchronize();
        
        float total_time = 0.0f;
        for(int run = 0; run < NUM_RUNS; run++) {
            ms = run_naive_xgemm_1d_kernel(A_gpu, B_gpu, C_gpu, M, N, K);
            total_time += ms;
        }
        printf("Naive 1D GEMM average time: %.3f ms\n", total_time / NUM_RUNS);
        fprintf(naive_1d_file, "%d,%.3f\n", dim, total_time / NUM_RUNS);

        total_time = 0.0f;
        for(int run = 0; run < NUM_RUNS; run++) {
            ms = run_naive_xgemm_2d_kernel(A_gpu, B_gpu, C_gpu, M, N, K);
            total_time += ms;
        }
        printf("Naive 2D GEMM average time: %.3f ms\n", total_time / NUM_RUNS);
        fprintf(naive_2d_file, "%d,%.3f\n", dim, total_time / NUM_RUNS);

        total_time = 0.0f;
        for(int run = 0; run < NUM_RUNS; run++) {
            ms = run_tile_xgemm_kernel(A_gpu, B_gpu, C_gpu, M, N, K);
            total_time += ms;
        }
        printf("Tiled Fine GEMM average time: %.3f ms\n", total_time / NUM_RUNS);
        fprintf(tiled_fine_file, "%d,%.3f\n", dim, total_time / NUM_RUNS);

        total_time = 0.0f;
        for(int run = 0; run < NUM_RUNS; run++) {
            ms = run_tiled_coarse_xgemm_kernel(A_gpu, B_gpu, C_gpu, M, N, K);
            total_time += ms;
        }
        printf("Tiled Coarse GEMM average time: %.3f ms\n", total_time / NUM_RUNS);
        fprintf(tiled_coarse_file, "%d,%.3f\n", dim, total_time / NUM_RUNS);

        cudaEventDestroy(start);
        cudaEventDestroy(end);
        cudaFree(A_gpu);
        cudaFree(B_gpu);
        cudaFree(C_gpu);
        free(A);
        free(B);
        free(C);
        free(C_ref);
    }
        fclose(naive_1d_file);
        fclose(naive_2d_file);
        fclose(tiled_fine_file);
        fclose(tiled_coarse_file);
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("multiProcessorCount: %d \n", prop.multiProcessorCount);
    const int minDim = 128;
    const int maxDim = 16384;
    benchmark_kernel_for_size(minDim, maxDim);
}