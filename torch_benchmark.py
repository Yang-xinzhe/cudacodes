import torch
import torch.nn.functional as F
import time

vector = torch.randn(5, dtype=float)
print("Input Vector: ", vector)

# softmax along the last dimension
output = F.softmax(vector, dim=-1)
print("Output Vector: ", output)

# Initialize the matrix on device 
matrix = torch.randn(1024, 32768, device='cuda', dtype=torch.float64)

# Warm up
_ = torch.nn.functional.softmax(matrix, dim=-1)

# Ensure all CUDA Operation are finished
torch.cuda.synchronize()

total_time = 0
n_iters = 5

for i in range(n_iters):
    # Measure time
    torch.cuda.synchronize() # Ensure all CUDA Operation are finished
    start = time.time()
    _ = torch.nn.functional.softmax(matrix, dim=-1)
    torch.cuda.synchronize()
    end = time.time()

    total_time += (end - start) * 1000
    print(total_time)

print(f"Softmax computation time (average): {total_time/n_iters:.3f} ms")