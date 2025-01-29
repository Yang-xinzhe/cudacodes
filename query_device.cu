#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int dev_count;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&dev_count);
    cudaGetDeviceProperties(&prop, 0);

    printf(">> CUDA enabled devices in the system: %d\n", dev_count);
    printf(">> Compute capability: %d.%d\n", prop.major, prop.minor);

    printf(">> Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf(">> Max block size: %d\n", prop.maxThreadsPerBlock);

    printf(">> Number of SMs: %d\n", prop.multiProcessorCount);
    printf(">> Clock Rate of the SMs (in kHz): %d\n", prop.clockRate);

    printf(">> Max threads dimension: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf(">> Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor); // 1536 / 256 = 6 blocks per SM

    printf(">> Registers available per block: %d\n", prop.regsPerBlock); // 65536
    printf(">> Registers available per SM: %d\n", prop.regsPerMultiprocessor); // 65536

    printf(">> Warp size (threads per warp): %d\n", prop.warpSize);
    printf(">> Shared memory size per block: %zd bytes\n", prop.sharedMemPerBlock); // 48 KB
    printf(">> Shared memory size per SM: %zd bytes\n", prop.sharedMemPerMultiprocessor); // 100 KB

    // CUDA Core → Registers → Shared Memory → L1 Cache → L2 Cache → Global Memory (VRAM)
    printf(">> L2 Cache size: %d bytes\n", prop.l2CacheSize); // 35 MB

    printf(">> Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf(">> Memory clock rate: %d kHz\n", prop.memoryClockRate);

    int cudaCores = prop.multiProcessorCount * 128;
    float clock_GHz = prop.clockRate / 1e6;
    float gflops = cudaCores * clock_GHz * 2; // Fused Multiply-Add， two float calculate

    printf(">> Theoretical Max GFLOPS: %.2f\n", gflops);

    float memoryBandwidth = (2 * prop.memoryClockRate * prop.memoryBusWidth) / (8.0 * 1e6);
    printf(">> Maximum Memory Bandwidth: %.2f GB/s\n", memoryBandwidth);
}