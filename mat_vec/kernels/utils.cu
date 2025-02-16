#include "utils.cuh"

/*
Helper function to generate a clamped random number sampled from a
normal distribution with mean 0 and std 1
*/
// Box-Muller
float random_normal_clamped(float min, float max){
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cos(2.0f * M_PI * u2);
    if(num < min){
        return min;
    }
    if(num > max){
        return num;
    }
    return num;
}

// GFLOPS = (2*M*N ops) / (ms * 1e6) : ops=multiply-add per element
float compute_gflops(int M, int N, float ms) {
    return (2 * M * N) / (ms * 1e6);
}

// Return achieved GFLOPS as percentage of GPU's theoretical peak GFLOPS
float compute_peak_gflops(float gflops){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Total 3072 CUDA Cores are distributed across 24 SMs
    int cudaCores = prop.multiProcessorCount * 128;
    float clockGHz = prop.clockRate / 1e6;
    float theoretical_max_gflops = cudaCores * clockGHz * 2;
    return (gflops / theoretical_max_gflops) * 100;
}


float compute_peak_memory_bandwidth(int M, int N, float ms) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Memory bandwidth (GB/s) = (2 * clock_rate * bus_width) / (8 * 1e6)
    // - clock_rate: DDR memory runs 2x per clock
    // - bus_width: bits per transfer
    // - /8: bits to bytes
    // - /1e6: MHz to GB/s
    float theoretical_max_memory_bandwidth = (2 * prop.memoryClockRate * prop.memoryBusWidth) / (8.0 * 1e6);
    
    // MatVec: Matrix A: M * N, vector x: N, result vector y: M
    size_t totalFloats = (size_t)(M * N + N + M);
    float totalBytes = (float)totalFloats * sizeof(float);

    float secs = ms / 1000.0f;
    float gbPerSec = (totalBytes / secs) / 1.0e9;

    return (gbPerSec / theoretical_max_memory_bandwidth) * 100;
}


void print_kernel_essential(int M, int N, float ms) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int cudaCores = prop.multiProcessorCount * 128;
    float clockGHz = prop.clockRate / 1e6;
    float theoretical_max_gflops = cudaCores * clockGHz * 2;
    float theoretical_max_memory_bandwidth = (2 * prop.memoryClockRate * prop.memoryBusWidth) / (8.0 * 1e6);
    float gflops = compute_gflops(M, N, ms);
    printf(">> Execution time: %.3f ms\n", ms);
    printf(">> Achieved (GFLOPS): %.3f \n", gflops);
    printf(">> Theoretical max (GFLOPS): %.3f\n", theoretical_max_gflops);
    printf(">> Theoretical max memory bandwidth: %.3f GB/s\n", theoretical_max_memory_bandwidth);
    printf(">> Achieves %.3f %% of peak GFLOPS\n", compute_peak_gflops(gflops));
    printf(">> Achieves %.3f %% of peak Memory Bandwidth\n", compute_peak_memory_bandwidth(M, N, ms));
}
