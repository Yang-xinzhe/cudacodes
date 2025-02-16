#include "naive_sgemv.cuh"

__global__ void naive_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < M) {
        float sum = 0.0f;
        for(int col = 0 ; col < N ; ++col){
            sum += mat_gpu[row * N + col] * vec_gpu[col];
        }
        res_gpu[row] = sum;
    }
}

void run_naive_sgemv_kernel(float* __restrict__ mat_gpu, float* __restrict__ vec_gpu, float* __restrict__ res_gpu, int M, int N){
    dim3 block_size(512);
    dim3 grid_size((M + block_size.x - 1) / block_size.x);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float ms = 0.0f;
    cudaEventRecord(start);
    naive_sgemv_kernel<<<grid_size, block_size>>>(mat_gpu, vec_gpu, res_gpu, M, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    printf("naive_sgemv_kernel execute time: %.3f ms", ms);
}
