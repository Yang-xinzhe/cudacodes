import torch
import time

# matrix_size = 4096 # (1024 x 1024)
matrix_size = 1024
max_val = 10
min_val = -10

# Initialize random tensors with a normal distribution and clamp values
A = torch.rand((matrix_size, matrix_size)).clamp(min=min_val, max=max_val)
B = torch.rand((matrix_size, matrix_size)).clamp(min=min_val, max=max_val)

A, B = A.cuda(), B.cuda()

print(f">> Benchmark torch.matmul for {matrix_size} x {matrix_size} matrices...")

start_time = time.time()
C = torch.matmul(A, B)
end_time = time.time()

elasped_time = end_time - start_time 

print(f">> Matrix multiplication completed in {elasped_time:.3f} seconds, {elasped_time*1000:.3f} ms")
