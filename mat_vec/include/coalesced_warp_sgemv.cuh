# pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include "utils.cuh"

__device__ __forceinline__ float warpReduce(float val);
__global__ void coalesced_warp_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N);
__global__ void multi_warp_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N);
void run_coalesced_warp_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N);
void run_multi_warp_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N);

