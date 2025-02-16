#include "coalesced_warp_sgemv.cuh"
#include "coalesced_warpblock_sgemv.cuh"

__device__ __forceinline__ void blockReduceSum(float val, float *smem, int tid, int blockDimX) {
    // 1. do warpReduce sum 
    val = warpReduce(val); // using shuffle instruction to do reduction within the warp

    // 2. do blockReduce sum
    if (blockDimX > warpSize) { // More than one warp in a block
        int lane = tid % warpSize; // Thread index within a Warp
        int wid = tid / warpSize; // Warp index within a Block
        
        // First Thread contain the reduction result from a warp
        if(lane == 0) {
            smem[wid] = val;
        }

        __syncthreads();

        // First Warp reduction
        if(tid < warpSize) {
            val = tid < ((blockDimX + warpSize - 1) / warpSize) ? smem[tid] : 0.0f;
            val = warpReduce(val);
            if (tid == 0) {
                smem[0] = val;
            }
        }
    } else {
        if(tid == 0) {
            smem[0] = val;
        }
    }
    // syncthreads();
    // sync not needed because only thread 0 reads from smem[0]
}


/*
Coalesced Warp Block Sgemv kernel

- Each block is assigned to a row of the matrix A
- Each block calculates one output element of y
- The columns are accessed in coalesced manner by threads
- Performs warp level + block level sum reduction
*/
__global__ void coalesced_warpblock_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N) {
    extern __shared__ float smem[];

    int bid = blockIdx.x;
    if(bid >= M) return;

    int tid = threadIdx.x;
    // each thread calculates its own partial output
    float partial_sum = 0.0f;
    for(int col = tid ; col < N ; col += blockDim.x) {
        partial_sum += mat_gpu[bid * N + col] * vec_gpu[col];
    }

    // block level sum reduction
    // only first thread reads the first location in shared memory
    // only first thread writes the output to global memory
    blockReduceSum(partial_sum, smem, tid, blockDim.x);
    if(tid == 0) {
        float sum = smem[0];
        res_gpu[bid] = sum;
    }
}

float run_kernel_coalesced_warpblock_sgemv(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N) {
    int NUM_THREADS = 64;
    int warp_size = 32;

    dim3 block_size(NUM_THREADS);
    dim3 grid_size(M);

    size_t shared_memory_size = ((block_size.x + warp_size - 1) / warp_size) * sizeof(float);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float ms = 0.0f;

    cudaEventRecord(start);
    coalesced_warp_sgemv_kernel<<<grid_size, block_size, shared_memory_size>>>(mat_gpu, vec_gpu, res_gpu, M, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
}