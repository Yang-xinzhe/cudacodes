import torch
import time
import matplotlib.pyplot as plt
import numpy as np

def benchmark_matmul(size):
    # 初始化矩阵
    A = torch.rand((size, size)).clamp(min=-10, max=10)
    B = torch.rand((size, size)).clamp(min=-10, max=10)
    
    A, B = A.cuda(), B.cuda()
    
    # 预热GPU
    for _ in range(5):
        _ = torch.matmul(A, B)
    
    # 计时
    start_time = time.time()
    C = torch.matmul(A, B)
    torch.cuda.synchronize()  # 确保GPU运算完成
    end_time = time.time()
    
    return (end_time - start_time) * 1000  # 转换为毫秒

# 设置不同的矩阵大小
sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
times = []  # PyTorch的时间
cuda_times = [0.013, 0.022, 0.095, 0.629, 5.130, 39.659, 317.064]  # CUDA的时间

# 运行基准测试
print("开始基准测试...")
for size in sizes:
    print(f"测试矩阵大小: {size}x{size}")
    avg_time = 0
    runs = 10  # 每个大小运行10次取平均
    for _ in range(runs):
        avg_time += benchmark_matmul(size)
    avg_time /= runs
    times.append(avg_time)
    print(f"PyTorch平均耗时: {avg_time:.3f} ms")

# 绘制图表
plt.figure(figsize=(10, 6))
x_positions = np.arange(len(sizes))

# 绘制PyTorch结果
plt.plot(x_positions, times, 'bo-', linewidth=2, markersize=8, label='PyTorch')
# 绘制CUDA结果
plt.plot(x_positions, cuda_times, 'ro-', linewidth=2, markersize=8, label='CUDA')

plt.grid(True)
plt.xlabel('Matrix Size')
plt.ylabel('Execute time (ms)')
plt.title('MatMul benchmark')
plt.legend()

# 添加数据标签
for i, (torch_time, cuda_time) in enumerate(zip(times, cuda_times)):
    plt.annotate(f'{torch_time:.2f}ms', 
                (x_positions[i], torch_time), 
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center',
                color='blue')
    plt.annotate(f'{cuda_time:.2f}ms', 
                (x_positions[i], cuda_time), 
                textcoords="offset points", 
                xytext=(0,-20),  # 增加与x轴的距离
                ha='center',
                color='red')

plt.xticks(x_positions, sizes)


ymin = min(min(times), min(cuda_times)) * 0.1  # 从0.5改为0.1，使小值部分跨度更大
ymax = max(max(times), max(cuda_times)) * 1.2
plt.ylim(ymin, ymax)

plt.savefig('matmul_benchmark.png', dpi=300, bbox_inches='tight')
plt.show() 