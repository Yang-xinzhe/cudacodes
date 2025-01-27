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
sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
times = []

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
    print(f"平均耗时: {avg_time:.3f} ms")

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(sizes, times, 'bo-', linewidth=2, markersize=8)
plt.grid(True)
plt.xlabel('Matrix Size')
plt.ylabel('Execute time (ms)')
plt.title('MatMul benchmark')

# 添加数据标签
for i, (size, time) in enumerate(zip(sizes, times)):
    plt.annotate(f'{time:.2f}ms', 
                (size, time), 
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center')

# 设置x轴刻度
plt.xticks(sizes)

# 保存图表
plt.savefig('matmul_benchmark.png', dpi=300, bbox_inches='tight')
plt.show() 