import pandas as pd
import os

# 读取你的账本
df = pd.read_csv('al_file/train.csv')

print("📊 --- 数据集全景扫描 ---")
print(f"总图片数: {len(df)}")

# 查看各个数字标签的占比
counts = df.iloc[:, 1].value_counts().sort_index()
print("\n🔍 各类别标签的数量与占比：")
for label, count in counts.items():
    percent = (count / len(df)) * 100
    print(f"标签 {label}: {count} 张 ({percent:.2f}%)")

# 打印前 5 行，看看长什么样
print("\n👀 账本前 5 行抽样：")
print(df.head())