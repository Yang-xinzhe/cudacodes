# pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void naive_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N);
void run_naive_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N);