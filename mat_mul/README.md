# Learning CUDA Programming Through Matrix Multiplication (GEMM) Optimization

Matrix multiplication is a fundamental operation in linear algebra and plays a crucial role in many computational tasks. For two matrices A (M×K) and B (K×N), their product C (M×N) is computed as:
![alt text](media/mat_mul.png)

$$A_{M \times K} \times B_{K \times N} = C_{M \times N}$$

$$c_{mn} = \sum_{k=0}^{K-1} a_{mk} \cdot b_{kn}$$

## Naive_xgemm_kernel
``naive_xgemm_1d_kernel``, performs general matrix multiplication (GEMM) in a naive, one-dimensional (1D) parallelized manner.
```
int tid = blockDim.x * blockIdx.x + threadIdx.x;
```
``tid`` represents the global thread index across all thread blocks.

For each valid thread corresponding to an output element $C[row,col]$, the kernel performs the dot product:
```
float sum = 0.0f;
for (int k = 0; k < K; k++) {
    sum += A_gpu[row * K + k] * B_gpu[k * N + col];
}
C_gpu[row * N + col] = sum;
```
This naive 1D GEMM kernel **assigns one thread per output element** in $C$,iterates over the shared dimension $K$,and performs the matrix multiplication directly in global memory. While functional, it can be significantly optimized for better performance.

## Tiled_xgemm_kernel
![alt text](media/tiled_matrix.png)
``tiled_xgemm_kernel``,this kernel uses **shared memory tiling**, significantly reducing **global memory accesses** and improving **memory coalescing**. 

$$
\begin{aligned}
\begin{bmatrix} 
A_{00} & A_{01} & \cdots & A_{0p} \\
A_{10} & A_{11} & \cdots & A_{1p} \\
\vdots & \vdots & \ddots & \vdots \\
A_{m0} & A_{m1} & \cdots & A_{mp} 
\end{bmatrix} 
\quad \times 
\quad 
\begin{bmatrix} 
B_{00} & B_{01} & \cdots & B_{0n} \\
B_{10} & B_{11} & \cdots & B_{1n} \\
\vdots & \vdots & \ddots & \vdots \\
B_{p0} & B_{p1} & \cdots & B_{pn} 
\end{bmatrix}
\quad = 
\quad
\begin{bmatrix} 
C_{00} & C_{01} & \cdots & C_{0n} \\
C_{10} & C_{11} & \cdots & C_{1n} \\
\vdots & \vdots & \ddots & \vdots \\
C_{m0} & C_{m1} & \cdots & C_{mn} 
\end{bmatrix}
\end{aligned}
$$

$$
C_{ij} = \sum_{k=1}^{K} A_{ik} \cdot B_{kj}
$$
```tile_xgemm_kernel```Each thread is responsible for calculating one element of the result matrix uses **shared memory** to load and store submatrices (tiles) of $A$ and $B$,thereby optimizing memory access and improving performance on GPUs.

Each thread is responsible for calculating one element of the result matrix $C$,corresponding to a particular $row$ and $col$ 
```
__shared__ float a_smem[TILE_SIZE][TILE_SIZE];
__shared__ float b_smem[TILE_SIZE][TILE_SIZE];
```
**Tiles** are of size $TILE\_SIZE \times TILE\_SIZE$,allowing multiple threads to work on smaller sections of the matrices at a time, improving the efficiency of memory access.

- A 1024x1024 matrix is divided into 32x32 smaller sub-matrices, each of size 32x32, for computation.

```
for(int i = 0 ; i < TILE_SIZE ; i++) {
    sum += a_smem[threadIdx.y][i] * b_smem[i][threadIdx.x];
}
```

Each thread computes a part of the dot product between a row of $A$ ans a column of $B$ by iterating over the tile size and performing element-wise multiplication.
## Benchmark GEMM
![alt text](media/performance_comparison.png)

![alt text](media/relative_performance.png)