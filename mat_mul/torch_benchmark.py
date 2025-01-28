import torch
import time
import os

sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

with open('benchmarks/pytorch.csv', 'w') as f:
    f.write('matrix_size,time_ms\n')
    
    for matrix_size in sizes:
        print(f">> Testing matrix size: {matrix_size}x{matrix_size}")
        
        # 初始化随机矩阵
        A = torch.rand((matrix_size, matrix_size)).clamp(min=-10, max=10)
        B = torch.rand((matrix_size, matrix_size)).clamp(min=-10, max=10)
        
        A, B = A.cuda(), B.cuda()
        
        for _ in range(5):
            _ = torch.matmul(A, B)
            torch.cuda.synchronize()
        
        total_time = 0
        runs = 10
        for _ in range(runs):
            torch.cuda.synchronize()
            start_time = time.time()
            C = torch.matmul(A, B)
            torch.cuda.synchronize()
            end_time = time.time()
            total_time += (end_time - start_time) * 1000 
        
        avg_time = total_time / runs
        print(f">> PyTorch average time: {avg_time:.3f} ms")

        f.write(f"{matrix_size},{avg_time:.3f}\n")
        f.flush() 
        
        del A, B, C
        torch.cuda.empty_cache()

print("Benchmark completed. Results saved to benchmarks/pytorch.csv")