#include "coalesced_warp_sgemv.cuh"

__device__ __forceinline__ float warpReduce(float val){
    for(int offset = warpSize / 2 ; offset > 0 ; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// With a Warp in a Block -> block_size(32)
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
    float sum = warpReduce(partial_sums[tid]);
    if(tid == 0){
        res_gpu[bid] = sum;
    }

}

// With multiple Warps in a Block -> block_size(32 * 4)
__global__ void multi_warp_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid / warpSize;  // warp ID
    int lane = tid % warpSize; // 线程在warp内的位置
    
   // Assuming using 4 Warps, share memory 4 * 32 = 128
    __shared__ float partial_sums[128];
    
    
    float partial_sum = 0.0f;
    for(int col = tid; col < N; col += blockDim.x) {
        partial_sum += mat_gpu[bid * N + col] * vec_gpu[col];
    }
    
    partial_sum = warpReduce(partial_sum);
    
    if(lane == 0) {
        partial_sums[wid] = partial_sum;
    }
    __syncthreads();
    
    if(wid == 0) {
        float sum = (lane < blockDim.x/warpSize) ? partial_sums[lane] : 0.0f;
        sum = warpReduce(sum);
        if(lane == 0) {
            res_gpu[bid] = sum;
        }
    }
}

void run_coalesced_warp_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N){
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float ms = 0.0f;

    dim3 block_size(32); // single warp in a Block
    dim3 grid_size(M); // M row matrix, every block calculate a row
    int share_memory_size = warpSize * sizeof(float);

    cudaEventRecord(start);
    coalesced_warp_sgemv_kernel<<<grid_size, block_size, share_memory_size>>>(mat_gpu, vec_gpu, res_gpu, M, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    print_kernel_essential(M, N, ms);
}

void run_multi_warp_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float ms = 0.0f;

    dim3 block_size(128); // 4 Warps in a Block
    dim3 grid_size(M);
    int share_memory_size = 4 * warpSize * sizeof(float);

    cudaEventRecord(start);
    multi_warp_sgemv_kernel<<<grid_size, block_size, share_memory_size>>>(mat_gpu, vec_gpu, res_gpu, M, N);
    cudaEventRecord(end);
    cudaEventElapsedTime(&ms, start, end);
    print_kernel_essential(M, N, ms);
}