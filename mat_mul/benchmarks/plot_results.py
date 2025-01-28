import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import LogFormatter

class CustomLogFormatter(LogFormatter):
    def __call__(self, x, pos=None):
        if x < 1:
            return f"{x:.3f}"
        elif x < 10:
            return f"{x:.2f}"
        elif x < 100:
            return f"{x:.1f}"
        else:
            return f"{x:.0f}"

# 设置更易区分的颜色
implementations = {
    'pytorch': {'color': '#E41A1C', 'marker': 'o', 'label': 'PyTorch'},      # 红色
    'naive_1d': {'color': '#377EB8', 'marker': 's', 'label': 'Naive 1D'},    # 蓝色
    'naive_2d': {'color': '#4DAF4A', 'marker': '^', 'label': 'Naive 2D'},    # 绿色
    'tiled_fine': {'color': '#984EA3', 'marker': 'D', 'label': 'Tiled Fine'},    # 紫色
    'tiled_coarse': {'color': '#FF7F00', 'marker': 'v', 'label': 'Tiled Coarse'} # 橙色
}

# 读取所有CSV文件
results = {}
for impl in implementations.keys():
    file_path = f'{impl}.csv'
    if os.path.exists(file_path):
        results[impl] = pd.read_csv(file_path)

# 创建图表
plt.figure(figsize=(12, 8))

# 绘制每个实现的性能曲线
for name, config in implementations.items():
    if name in results:
        data = results[name]
        plt.plot(data['matrix_size'], data['time_ms'],
                color=config['color'],
                marker=config['marker'],
                linewidth=2,
                markersize=8,
                label=config['label'])

# 设置对数刻度
plt.xscale('log', base=2)
plt.yscale('log')

# 设置网格
plt.grid(True, which="both", ls="-", alpha=0.2)

# 自定义y轴格式
ax = plt.gca()
formatter = CustomLogFormatter()
ax.yaxis.set_major_formatter(formatter)
# 不设置minor formatter以减少标签数量
ax.yaxis.set_minor_formatter(plt.NullFormatter())

# 调整y轴刻度密度
ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))

# 添加标签和标题
plt.xlabel('Matrix Size', fontsize=12, fontweight='bold')
plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
plt.title('Matrix Multiplication Performance Comparison', fontsize=14, pad=20, fontweight='bold')

# 自定义x轴刻度
matrix_sizes = results['pytorch']['matrix_size']
plt.xticks(matrix_sizes, matrix_sizes, rotation=45)

# 添加图例
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')

# 计算和打印加速比
print("\nSpeedup relative to PyTorch implementation:")
pytorch_data = results['pytorch']
for size in pytorch_data['matrix_size']:
    print(f"\nMatrix size {size}x{size}:")
    pytorch_time = pytorch_data[pytorch_data['matrix_size'] == size]['time_ms'].values[0]
    
    for name, data in results.items():
        if name != 'pytorch':
            impl_time = data[data['matrix_size'] == size]['time_ms'].values[0]
            speedup = pytorch_time / impl_time
            print(f"{implementations[name]['label']}: {speedup:.2f}x")

plt.close()

# 创建相对性能图表
plt.figure(figsize=(12, 8))

# 计算每个实现相对于PyTorch的性能比
for name, config in implementations.items():
    if name != 'pytorch' and name in results:
        data = results[name]
        relative_perf = []
        for size in data['matrix_size']:
            pytorch_time = pytorch_data[pytorch_data['matrix_size'] == size]['time_ms'].values[0]
            impl_time = data[data['matrix_size'] == size]['time_ms'].values[0]
            relative_perf.append(pytorch_time / impl_time)
        
        plt.plot(data['matrix_size'], relative_perf,
                color=config['color'],
                marker=config['marker'],
                linewidth=2,
                markersize=8,
                label=config['label'])

plt.xscale('log', base=2)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.axhline(y=1, color='#666666', linestyle='--', alpha=0.5)

# 为相对性能图表设置合适的y轴格式
ax = plt.gca()
formatter = CustomLogFormatter()
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))

plt.xlabel('Matrix Size', fontsize=12, fontweight='bold')
plt.ylabel('Speedup relative to PyTorch', fontsize=12, fontweight='bold')
plt.title('Performance Relative to PyTorch Implementation', fontsize=14, pad=20, fontweight='bold')

plt.xticks(matrix_sizes, matrix_sizes, rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('relative_performance.png', dpi=300, bbox_inches='tight')
plt.close() 