// Matrix Multiplication (xGEMM) kernels

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <chrono>
#include <omp.h>

// 向上取整的除法
// 例如：CEIL_DIV(5, 2) = 3，因为 5/2 = 2.5，向上取整为 3
#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1)/ (y)) : ((x) / (y)))

#define BM 64
#define BK 8
#define BN 64
// Fine-grained (细粒度)：每个线程执行很少的工作，例如一个线程计算一个元素
// Coarse-grained (粗粒度): 每个线程执行更多的工作，例如一个线程计算八个元素
#define COURSE_FACTOR 8
#define TILE_SIZE 32

inline void CudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %s: %s at %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/*
Naive xGEMM kernel:

- 2D blocks, 2D threads
- Each thread calculates one element of the output matrix C
- No shared memory, only global memory access
*/
__global__ void naive_xgemm_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K) {
    int row = 0, col = 0;
#ifdef NAIVE_GEMM_1D
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    row = thread_id / N;
    col = thread_id % N;
#endif
    // for coalesced memory access
    // maps rows to y-direction, cols to x-directions
    // 列有限对访问内存更加友好
#ifdef NAIVE_GEMM_2D
    row = blockDim.y * blockIdx.y + threadIdx.y;
    col = blockDim.x * blockIdx.x + threadIdx.x;
#endif
    if(row < M && col < N) {
        float acc = 0.0f;
        for(int k = 0 ; k < K ; ++k){
            acc += A_gpu[row * K + k] * B_gpu[k * N + col];
        }
        C_gpu[row * N + col] = acc;
    }
}

// 将大矩阵分块，每次加载一对小块到共享内存，
// 计算小块点乘并累加，循环直到处理完所有小块。
__global__ void tiled_xgemm_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K) {
    // 1024 = 32 x 32 x 32 x 32 1024维的矩阵由32x32的块组成（总共有32x32个32x32维的块）

    // 计算全局矩阵(1024×1024)中的位置
    // 行位置计算:
    // blockIdx.y * TILE_SIZE: 块的起始行(每个块是32×32)，比如第2个块起始行就是 2*32=64
    // threadIdx.y: 在当前32×32块内的行偏移(0-31)
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 列位置计算:
    // blockIdx.x * TILE_SIZE: 块的起始列(每个块是32×32)，比如第3个块起始列就是 3*32=96
    // threadIdx.x: 在当前32×32块内的列偏移(0-31)
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // tile that will be loaded by THIS block
    __shared__ float a_smem[TILE_SIZE][TILE_SIZE];
    __shared__ float b_smem[TILE_SIZE][TILE_SIZE];

    float sum = 0.f;
    
    // THIS block will loop over the tiles in common dimension
    for(int tile_num = 0 ; tile_num < CEIL_DIV(K, TILE_SIZE) ; ++tile_num) {
        int offset = tile_num * TILE_SIZE;
        
        // out of bound check
        // Loading data into share_memory
        // same row, different column for A
        /*
            对于A矩阵的访问:
            行 = blockIdx.y * TILE_SIZE + threadIdx.y // 全局行号
            列 = offset + threadIdx.x // 小块内的列号
        */
        if(row < M  && (offset + threadIdx.x) < K) {
            a_smem[threadIdx.y][threadIdx.x] = A_gpu[row * K + offset + threadIdx.x];
        } else {
            a_smem[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // different row, same columns for B
        // 对于B矩阵的访问:
        // 行 = offset + threadIdx.y  // 当前处理的K维度位置
        // 列 = col                   // 全局列号(相对于1024维矩阵)
        if((offset + threadIdx.y) < K && col < N) {
            b_smem[threadIdx.y][threadIdx.x] = B_gpu[(offset + threadIdx.y) * N + col];
        } else {
            b_smem[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        /*
            a_smem[threadIdx.y][0:31]: A矩阵当前行的32个元素
            b_smem[0:31][threadIdx.x]: B矩阵当前列的32个元素
        */
        for(int i = 0 ; i < TILE_SIZE ; ++i) {
            sum += a_smem[threadIdx.y][i] *  b_smem[i][threadIdx.x];
        }
        __syncthreads();
    }
    // 小块计算的循环累加和，才是大矩阵中真正需要的计算结果
    if(row < M && col < N) {
        C_gpu[row * N + col] = sum;
    }
}

__global__ void tiled_xgemm_1d_coarse_kernel(float* __restrict__ A_gpu, float* __restrict__ B_gpu, float* __restrict__ C_gpu, int M, int N, int K) {
// 1. Grid（二维）: 16 × 16 个块
//    - blockIdx.y: 0 到 15  // 行方向的块索引
//    - blockIdx.x: 0 到 15  // 列方向的块索引

// 2. Block（一维）: 每个块包含一维排列的线程
//    - threadIdx.x: 一维线程索引
//    - 总线程数 = (BM * BN) / COURSE_FACTOR = 512   
    int by = blockIdx.y;
    int bx = blockIdx.x;
    
    // 同一个线程先加载B矩阵的值，再加载A矩阵的值
    // A(64x8) B(8x64)
    // for within each tile + for loading B's tile
    int ty = threadIdx.x / BN;
    int tx = threadIdx.x % BN;

    // for loading A's tile
    int aty = threadIdx.x / BK;
    int atx = threadIdx.x % BK;

    // working on C[row, col]
    // 行方向上一个线程处理8个元素，而列方向上只处理1个元素，不会导致不连续的内存访问
    int row = by * BM + (ty * COURSE_FACTOR);
    int col = bx * BN + tx;

    // shared memory for A and B for computing tiles
    __shared__ float a_smem[BM * BK];   // 64 * 8 = 512个元素的一维数组
    __shared__ float b_smem[BK * BN];   // 8 * 64 = 512个元素的一维数组

    /*
    1. a_smem的一维布局 (512个元素)：
    [0,1,2,3,4,5,6,7, | 8,9,10,11,12,13,14,15, | ... | 504,505,506,507,508,509,510,511]
    |----第0行8个元素----|----第1行8个元素-------|     |--------第63行8个元素---------|

    访问方式：
    a_smem[aty * BK + atx]
    例如：要访问"第2行第3列"：a_smem[2 * 8 + 3] = a_smem[19]

    2. b_smem的一维布局 (512个元素)：
    [0,1,2...63, | 64,65,66...127, | ... | 448,449...511]
    |--第0行64个--|---第1行64个----|     |--第7行64个--|

    访问方式：
    b_smem[ty * BN + tx]
    例如：要访问"第1行第3列"：b_smem[1 * 64 + 3] = b_smem[67]
    */

    float sum[COURSE_FACTOR] = {0.f};

    for(int tile = 0 ; tile < K ; tile += BK) { // 每次处理8列的数据, K是总矩阵的列数, 总共需要128次循环才能处理完整个K维度
        // load tile into shared memory for both A and B 
        if((by * BM + aty) < M && (tile + atx) < K) {
            a_smem[aty * BK + atx] = A_gpu[(by * BM + aty) * K + (tile + atx)];
        } else {
            a_smem[aty * BK + atx] = 0.f;
        }
        if((tile + ty) < K && (bx * BN + tx) < N) {
            b_smem[ty * BN + tx] = B_gpu[(tile + ty) * N + (bx * BN + tx)];
        } else {
            b_smem[ty * BN + tx] = 0.f;
        }

        __syncthreads();

        // inner loop
        // each thread compute 8 elements
        /*
        在每个小块内的计算过程：

        A矩阵的行向量(8个元素)：    B矩阵的列向量(1个元素)：
        k=0时：
        [a0 a1 a2 a3 a4 a5 a6 a7]  ×  [b0]  -> sum += a0*b0
                                    
        k=1时：
        [a0 a1 a2 a3 a4 a5 a6 a7]  ×  [b1]  -> sum += a1*b1

        k=2时：
        [a0 a1 a2 a3 a4 a5 a6 a7]  ×  [b2]  -> sum += a2*b2

        ...直到k=7

        而且因为是粗粒度计算，每个线程要计算8个这样的点积：

        对于c=0到7：
        A的第c行: [a0 a1 ... a7]    B的列: [b0]
                                        [b1]
                                        [b2]
                                        ...
                                        [b7]

        代码实现：
        */
        for(int k = 0 ; k < BK ; ++k) {
            float b_reg = b_smem[k * BN + tx];
            for(int c = 0 ; c < COURSE_FACTOR ; ++c) {
                sum[c] += a_smem[((ty * COURSE_FACTOR + c) * BK + k)] * b_reg;
            }
        }
        __syncthreads();
    }
    for(int c = 0 ; c < COURSE_FACTOR ; ++c) {
        if((row + c) < M && col < N) {
            C_gpu[(row + c) * N + col] = sum[c];
        }
    }
}   

// O(MNK)
void gemm_cpu_naive(float* A, float* B, float* C, int M , int N, int K) {
    // A x B = C 
    // A: MxK B: KxN C: MxN
    #pragma omp parallel for collapse(2)
    for(int i = 0 ; i < M ; ++i) {
        for(int j = 0 ; j < N ; ++j) {
            float sum = 0.f;
            for(int k = 0; k < K ; ++k){
                // A[i][k] * B[k][j] A的行与B的列的点积
                // 使用一维数组来表示二维数组: A[i][k] = A[i * K + k], B[k][j] = B[k * N + j]
                sum += (A[i * K + k] * B[k * N + j]);
            }
            // C[i][j] = C[i * N + j] 
            C[i * N + j] = sum;
        }
    }
}

void gemm_cpu_omp(float* A, float* B, float* C, int M, int N, int K) {
    double start_time = omp_get_wtime(); 
    #pragma omp parallel num_threads(20)
    {
        #pragma omp for collapse(2) schedule(dynamic, 32)
        for(int i = 0; i < M; ++i) {
            for(int j = 0; j < N; ++j) {
                double sum = 0.f;
                // 使用向量化
                #pragma omp simd reduction(+:sum)
                for(int k = 0; k < K; ++k) {
                    sum += (double)A[i * K + k] * (double)B[k * N + j];
                }
                C[i * N + j] = (float)sum;
            }
        }
    }
    
    double end_time = omp_get_wtime();
    printf("OpenMP矩阵乘法耗时: %.3f秒, %.3f ms\n", end_time - start_time, (end_time - start_time) * 1000);
}

// Make sure it's normally distributed (正态分布)
float random_normal_clamp(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min) {
        return min;
    } 
    if (num > max) {
        return max;
    }
    return num;
}

void benchmark_all_kernels(
    float* A_gpu, float* B_gpu, float* C_gpu, 
    int M, int N, int K, 
    int num_runs = 10)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // 1. Naive 1D GEMM
    dim3 block_size_1d(BM * BN / COURSE_FACTOR);
    dim3 grid_size_1d(CEIL_DIV(M * N, block_size_1d.x));
    
    printf("\n>> Testing naive_xgemm_1d_kernel:\n");
    float total_time_1d = 0.0f;
    for(int i = 0; i < num_runs; i++) {
        cudaEventRecord(start);
        naive_xgemm_kernel<<<grid_size_1d, block_size_1d>>>(A_gpu, B_gpu, C_gpu, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        total_time_1d += ms;
    }
    printf("Average time: %.3f ms\n", total_time_1d / num_runs);

    // 2. Naive 2D GEMM
    dim3 block_size_2d(32, 16);
    dim3 grid_size_2d(CEIL_DIV(N, block_size_2d.x), CEIL_DIV(M, block_size_2d.y));
    
    printf("\n>> Testing naive_xgemm_2d_kernel:\n");
    float total_time_2d = 0.0f;
    for(int i = 0; i < num_runs; i++) {
        cudaEventRecord(start);
        naive_xgemm_kernel<<<grid_size_2d, block_size_2d>>>(A_gpu, B_gpu, C_gpu, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        total_time_2d += ms;
    }
    printf("Average time: %.3f ms\n", total_time_2d / num_runs);

    // 3. Tiled GEMM
    dim3 block_size_tiled(TILE_SIZE, TILE_SIZE);
    dim3 grid_size_tiled(CEIL_DIV(N, TILE_SIZE), CEIL_DIV(M, TILE_SIZE));
    
    printf("\n>> Testing tiled_xgemm_kernel:\n");
    float total_time_tiled = 0.0f;
    for(int i = 0; i < num_runs; i++) {
        cudaEventRecord(start);
        tiled_xgemm_kernel<<<grid_size_tiled, block_size_tiled>>>(A_gpu, B_gpu, C_gpu, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        total_time_tiled += ms;
    }
    printf("Average time: %.3f ms\n", total_time_tiled / num_runs);

    // 4. Tiled Coarse GEMM
    dim3 block_size_coarse(BM * BN / COURSE_FACTOR);
    dim3 grid_size_coarse(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    
    printf("\n>> Testing tiled_xgemm_1d_coarse_kernel:\n");
    float total_time_coarse = 0.0f;
    for(int i = 0; i < num_runs; i++) {
        cudaEventRecord(start);
        tiled_xgemm_1d_coarse_kernel<<<grid_size_coarse, block_size_coarse>>>(A_gpu, B_gpu, C_gpu, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        total_time_coarse += ms;
    }
    printf("Average time: %.3f ms\n", total_time_coarse / num_runs);

    // 打印性能对比总结
    printf("\n>> Performance Summary (average over %d runs):\n", num_runs);
    printf("Naive 1D GEMM:     %.3f ms\n", total_time_1d / num_runs);
    printf("Naive 2D GEMM:     %.3f ms\n", total_time_2d / num_runs);
    printf("Tiled GEMM:        %.3f ms\n", total_time_tiled / num_runs);
    printf("Tiled Coarse GEMM: %.3f ms\n", total_time_coarse / num_runs);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int M = 8192;
    int N = 8192; 
    int K = 8192;

    int a_size = M * K;
    int b_size = K * N;
    int c_size = M * N;

    printf("A matrix size: (%d)\n", a_size);
    printf("B matrix size: (%d)\n", b_size);
    printf("C matrix size: (%d)\n", c_size);

    float *A = (float *)calloc(a_size, sizeof(float));
    float *B = (float *)calloc(b_size, sizeof(float));
    float *C = (float *)calloc(c_size, sizeof(float));
    float *C_cpu = (float *)calloc(c_size, sizeof(float));

    // init matrix with random values
    for(int i = 0 ; i < a_size ; ++i) {
        A[i] = random_normal_clamp(-10, 10);
    }
    for(int i = 0 ; i < b_size ; ++i) {
        B[i] = random_normal_clamp(-10, 10);
    }
    for(int i = 0 ; i < c_size ; ++i) {
        C[i] = 0.f;
    }

    float *A_gpu, *B_gpu, *C_gpu;

    dim3 block_size, grid_size;
#ifdef NAIVE_GEMM_1D
    block_size = dim3(BM * BN / COURSE_FACTOR); // (64×64)/8 = 512 个线程
    // 这是一维的线程配置，所有线程都在x维度上
    // threadIdx.x 的范围是 [0, 511]
    grid_size = dim3(CEIL_DIV(M * N, block_size.x)); // 2048个Block
    // 这也是一维的block网格
    // blockIdx.x 的范围是 [0, 2047]
#endif
#ifdef NAIVE_GEMM_2D
    block_size = dim3(32, 16);
    grid_size = dim3(CEIL_DIV(N, block_size.x), CEIL_DIV(M, block_size.y));
#endif
#ifdef TILE_GEMM
    block_size = dim3(TILE_SIZE, TILE_SIZE);
    grid_size = dim3(CEIL_DIV(N, TILE_SIZE), CEIL_DIV(M, TILE_SIZE));
#endif
#ifdef TILE_COARSE_GEMM
    block_size = dim3(BM * BN / COURSE_FACTOR);
    grid_size = dim3(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
#endif


    cudaEvent_t start, end;
    CudaAssert(cudaEventCreate(&start), __FILE__, __LINE__);
    CudaAssert(cudaEventCreate(&end), __FILE__, __LINE__);
    float ms = 0.0f;

    cudaEventRecord(start);
    cudaMalloc(&A_gpu, sizeof(float) * a_size);
    cudaMalloc(&B_gpu, sizeof(float) * b_size);
    cudaMalloc(&C_gpu, sizeof(float) * c_size);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    printf(">> GPU Allocation Elapsed time: %.3f ms\n", ms);

    cudaEventRecord(start);
    cudaMemcpy(A_gpu, A, sizeof(float) * a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(float) * b_size, cudaMemcpyHostToDevice);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    printf(">> Host to Device data transfer time: %.3f ms\n", ms);

    cudaEventRecord(start);
    tiled_xgemm_1d_coarse_kernel<<<grid_size, block_size>>>(A_gpu, B_gpu, C_gpu, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    printf(">> tiled_xgemm_1d_coarse_kernel execute time: %.3f ms\n", ms);

    cudaEventRecord(start);
    cudaMemcpy(C, C_gpu, c_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    printf(">> Device to Host data transfer time: %.3f ms\n", ms);

    printf(">> Running GEMM on CPU...\n");
    // clock_t ts = clock();
    auto ts = std::chrono::high_resolution_clock::now();
    gemm_cpu_naive(A, B, C_cpu, M, N, K);
    // gemm_cpu_omp(A, B, C_cpu, M, N, K);
    auto te = std::chrono::high_resolution_clock::now();
    // clock_t te = clock();
    printf("CPU execute done!\n");
    auto elapse_time = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
    // float elapsed_time = (te - ts) * 1000 / CLOCKS_PER_SEC;
    printf("Elasped time: %.3f ms\n", static_cast<float>(elapse_time));

    bool match = true;
    float eps = 0.0001;
    for(int i = 0 ; i < c_size ; ++i) {
        if(fabs(C_cpu[i] - C[i]) > eps) {
            match = false;
            printf("Mismatch at index %d: CPU=%.6f, GPU=%.6f, diff=%.6f\n", i, C_cpu[i], C[i], fabs(C_cpu[i]-C[i]));
            break;
        }
    }
    printf("\n>> Result match for CPU and GPU ");
    printf("%s\n", match ? "true" : "false");

    // benchmark_all_kernels(A_gpu, B_gpu, C_gpu, M, N, K, 20);

    free(A);
    free(B);
    free(C);
    free(C_cpu);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);

    cudaDeviceReset();

    return 0;
}