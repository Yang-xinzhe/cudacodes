# pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

__device__ __forceinline__ void blockReduceSum(float val, float *smem, int tid, int blockDimX);
__global__ void coalesced_warpblock_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N);
float run_kernel_coalesced_warpblock_sgemv(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N);
