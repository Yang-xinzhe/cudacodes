#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include "utils.cuh"


const int  M = 4096;
const int  N = 4096;

__global__ void naive_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < M){
        float sum = 0.0f;
        for(int col = 0 ; col < N ; ++col) {
            sum += mat_gpu[row * N + col] * vec_gpu[col];
        }
        res_gpu[row] = sum;
    }
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}

__global__ void coalesced_warp_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N) {
    assert(blockDim.x == warpSize);
    
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float partial_sums[32];

    float partial_sum = 0.0f;
    for(int col = tid ; col < N ; col += blockDim.x) {
        partial_sum += mat_gpu[bid * N + col] * vec_gpu[col];
    }

    partial_sums[tid] = partial_sum;
    __syncthreads();

    // warp level sum reduction
    // only first thread writes the output to global memory
    float sum = warpReduceSum(partial_sums[tid]);
    if(tid == 0){
        res_gpu[bid] = sum;
    }
}

__global__ void improved_coalesced_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N) {
    // 使用更大的block size
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threads_per_row = blockDim.x;  // 例如256
    const int row = bid;
    
    // 共享内存存储部分和
    __shared__ float sdata[256];  // 假设block size是256
    
    float partial_sum = 0.0f;
    // 每个线程处理更少的元素
    for(int col = tid; col < N; col += threads_per_row) {
        partial_sum += mat_gpu[row * N + col] * vec_gpu[col];
    }
    
    // 存入共享内存
    sdata[tid] = partial_sum;
    __syncthreads();
    
    // block内归约
    for(int s = threads_per_row/2; s > 32; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // warp级别最后归约
    if(tid < 32) {
        partial_sum = sdata[tid];
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, 16);
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, 8);
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, 4);
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, 2);
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, 1);
        
        if(tid == 0) {
            res_gpu[row] = partial_sum;
        }
    }
}

void cpu_sgemv(float* __restrict__ matrix, float* __restrict__ vector, float* __restrict__ result, int M, int N){
    for(int i = 0 ; i < M ; ++i){
        float sum = 0.0f;
        for(int j = 0 ; j < N ; ++j){
            sum += matrix[i * N + j] * vector[j];
        }
        result[i] = sum;
    }
}

bool check_result(float* result_gpu, float* result_cpu, int M) {
    for(int i = 0 ; i < M ; ++i) {
        if(fabs(result_gpu[i] - result_cpu[i]) > 0.001) {
            printf("Result Failed at res_gpu[%d] = %.5f, res_cpu[%d] = %.5f\n", i, result_gpu[i], i, result_cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // int cudaCores = prop.multiProcessorCount * 128;
    // float clockGHz = prop.clockRate / 1e6;

    // float theoretical_max_GFLOPS = cudaCores * clockGHz * 2;
    // float theoretical_max_memory_bandwidth = (2 * prop.memoryClockRate * prop.memoryBusWidth) / (8.0 * 1e6);

    // printf(">> Theoretical max %.3f GFLOPS \n", theoretical_max_GFLOPS);
    // printf(">> Theoretical max memory bandwidth %.3f GB/s\n", theoretical_max_memory_bandwidth);

    size_t mat_size = M * N;
    size_t vec_size = N;
    
    size_t mat_totalsize = mat_size * sizeof(float);
    size_t vec_totalsize = vec_size * sizeof(float);

    float* matrix = (float*)malloc(mat_totalsize);  
    float* vector = (float*)malloc(vec_totalsize);
    float* result = (float*)malloc(M * sizeof(float));
    float* result_cpu = (float*)malloc(M * sizeof(float));

    for(size_t i = 0 ; i < mat_size ; ++i){
        matrix[i] = random_normal_clamped(-10.f, 10.0f);
        if(i < vec_size) {
            vector[i] = random_normal_clamped(-10.0f, 10.0f);
        }
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float ms = 0.0f;

    float* mat_gpu, *vec_gpu, *res_gpu;
    // Allocate Device
    cudaEventRecord(start);
    cudaMalloc((void**)&mat_gpu, mat_totalsize);
    cudaMalloc((void**)&vec_gpu, vec_totalsize);
    cudaMalloc((void**)&res_gpu, M * sizeof(float));
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    printf(">> GPU Allocation time: %.3f ms\n", ms);

    // Copy data from Host to Device
    cudaEventRecord(start);
    cudaMemcpy(mat_gpu, matrix, mat_totalsize, cudaMemcpyHostToDevice);
    cudaMemcpy(vec_gpu, vector, vec_totalsize, cudaMemcpyHostToDevice);
    cudaMemcpy(res_gpu, result, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    printf(">> Host to Device Data Transfer Time: %.3f ms\n", ms);

    // // Run cuBLAS kernel
    cublasHandle_t handle;
    cublasCreate(&handle);

    // // Sgemv: y = (alpha * A * x) + (beta * y)
    float alpha = 1.0f, beta = 0.0f;
    cudaEventRecord(start);
    cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, mat_gpu, N, vec_gpu, 1, &beta, res_gpu, 1);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    printf("--------cuBLAS sgemv kernel-------\n");
    print_kernel_essential(M, N, ms);
    printf("--------------------------------\n");

    cudaMemcpy(result, res_gpu, M * sizeof(float), cudaMemcpyDeviceToHost);
    cpu_sgemv(matrix, vector, result_cpu, M, N);

    assert(check_result(result, result_cpu, M));
    
    dim3 block_size(32);
    dim3 grid_size(M);
    int shared_memory_size = 32 * sizeof(float);

    cudaEventRecord(start);
    coalesced_warp_sgemv_kernel<<<grid_size, block_size, shared_memory_size>>>(mat_gpu, vec_gpu, res_gpu, M, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    printf("-------coalesced_warp_sgemv_kernel--------\n");
    print_kernel_essential(M, N, ms);

    cudaMemcpy(result, res_gpu, M * sizeof(float), cudaMemcpyDeviceToHost);
    cpu_sgemv(matrix, vector, result_cpu, M, N);
    assert(check_result(result, result_cpu, M));

    printf("Result Correct!\n");
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return 0;
}