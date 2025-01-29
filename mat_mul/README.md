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
\mathbf{A} &=
\begin{bmatrix}
A_{00} & A_{01} & \cdots & A_{0p} \\
A_{10} & A_{11} & \cdots & A_{1p} \\
\vdots & \vdots & \ddots & \vdots \\
A_{m0} & A_{m1} & \cdots & A_{mp}
\end{bmatrix}
\quad
\mathbf{B} &=
\begin{bmatrix}
B_{00} & B_{01} & \cdots & B_{0n} \\
B_{01} & B_{11} & \cdots & B_{1n} \\
\vdots & \vdots & \ddots & \vdots \\
B_{p0} & B_{p1} & \cdots & B_{pn}
\end{bmatrix}
\end{aligned}
$$





![alt text](media/performance_comparison.png)

![alt text](media/relative_performance.png)