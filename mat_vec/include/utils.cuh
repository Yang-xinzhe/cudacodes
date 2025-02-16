#ifndef __UTILS_H_
#define __UTILS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

typedef struct benchmark{
    float peak_gflops;
    float peak_memory_bandwidth;
}benchmark;


float random_normal_clamped(float min, float max);
float compute_gflops(int M, int N, float ms);
float compute_peak_gflops(float gflops);
float compute_peak_memory_bandwidth(int M, int N, float ms);
void print_kernel_essential(int M, int N, float ms);

#endif