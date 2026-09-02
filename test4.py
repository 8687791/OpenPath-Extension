import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
# 提取你表格中的核心对比指标 (%)
old_final = [82.0, 26.0, 40.0, 80.0, 56.0, 68.0, 76.0, 34.0, 74.0]
new_density = [100.0, 93.88, 91.67, 100.0, 94.0, 76.0, 97.44, 46.0, 82.0]
new_graph = [100.0, 93.88, 91.67, 98.0, 96.0, 80.0, 97.44, 56.0, 84.0]

# 2. 全局字体和样式设置 (适配 IEEE 论文)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 3. 创建画布 (设置为适合跨双栏的宽尺寸，例如 8 x 3 英寸)
fig, ax = plt.subplots(figsize=(8, 3.2))

# 设置柱子的宽度和位置
x = np.arange(len(classes))
width = 0.25  

# 4. 绘制分组柱状图 (加入不同颜色和底纹以适配黑白打印)
rects1 = ax.bar(x - width, old_final, width, label='Old Method Final', 
                color='#cccccc', edgecolor='black')
rects2 = ax.bar(x, new_density, width, label='Ours (Density-Core FPS)', 
                color='#4c72b0', edgecolor='black', hatch='//')
rects3 = ax.bar(x + width, new_graph, width, label='Ours (Graph Centrality NMS)', 
                color='#55a868', edgecolor='black', hatch='\\\\')

# 5. 添加文本、标签和网格
ax.set_ylabel('Validation Accuracy (%)', weight='bold')
ax.set_xlabel('Pathology Categories', weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 115) # 顶部留出空间放图例

# 优化网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# 6. 图例设置 (放在顶部，不遮挡数据)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, frameon=False)

# 7. 导出为高精度 PDF (直接用于 LaTeX)
plt.tight_layout()
plt.savefig('class_accuracy_comparison.pdf', format='pdf', bbox_inches='tight')
plt.show()