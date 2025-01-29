# Learning CUDA Programming Through Matrix Multiplication (GEMM) Optimization

Matrix multiplication is a fundamental operation in linear algebra and plays a crucial role in many computational tasks. For two matrices A (M×K) and B (K×N), their product C (M×N) is computed as:
![alt text](media/mat_mul.png)
$$
c_{mn} = \sum_{k=0}^{K-1} a_{mk} \cdot b_{kn}
$$

![alt text](media/performance_comparison.png)

![alt text](media/relative_performance.png)