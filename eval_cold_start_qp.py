import os
import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="计算冷启动阶段第一轮的种子纯度 (QP)")
    parser.add_argument("--query_csv", type=str, default="al_file/query_round_5.csv", help="VLM生成的冷启动文件")
    parser.add_argument("--truth_csv", type=str, default="al_file/train.csv", help="包含全量真实标签的CSV文件")
    # 严格对齐 OpenPath 论文中 CRC100K 的 ID 类别：LYM, NORM, TUM 对应的标签
    parser.add_argument("--id_cls", nargs="+", type=int, default=[3, 6, 8], help="目标已知类的Label整数值")
    args = parser.parse_args()

    # 1. 确保文件存在
    if not os.path.exists(args.query_csv):
        print(f"❌ 错误：找不到冷启动筛选结果文件 {args.query_csv}，请先运行 vlm_generative_ood.py")
        return
    if not os.path.exists(args.truth_csv):
        print(f"❌ 错误：找不到真实标签文件 {args.truth_csv}")
        return

    # 2. 读取数据
    query_df = pd.read_csv(args.query_csv)
    truth_df = pd.read_csv(args.truth_csv)

    # 3. 建立 真实“图片路径 -> 标签” 的映射字典（防止绝对路径/相对路径不一致，统一提取文件名比对）
    truth_df.columns = ['img', 'label'] # 确保列名统一
    truth_dict = {os.path.basename(row['img']): row['label'] for _, row in truth_df.iterrows()}

    # 4. 统计冷启动挑出的样本的真实标签
    total_selected = len(query_df)
    id_count = 0
    ood_count = 0
    missing_count = 0

    print(f"📊 开始评估冷启动种子集，总计挑选样本数: {total_selected}")
    print("-" * 50)

    for _, row in query_df.iterrows():
        img_name = os.path.basename(row['img'])
        if img_name in truth_dict:
            real_label = int(truth_dict[img_name])
            if real_label in args.id_cls:
                id_count += 1
            else:
                ood_count += 1
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"⚠️ 警告：有 {missing_count} 张图片在原始 train.csv 中未匹配到真实标签，请检查路径！")

    # 5. 计算并打印核心指标 (QP)
    effective_total = total_selected - missing_count
    if effective_total == 0:
        print("❌ 无法计算 QP，有效匹配样本数为 0。")
        return

    qp = (id_count / effective_total) * 100

    print(f"✨ 【评估结果】")
    print(f" └─ 真正目标类 (ID Samples) 数量: {id_count}")
    print(f" └─ 误选未知类 (OOD Samples) 数量: {ood_count}")
    print(f" └─ 💡 第一轮查询精准度 Query Precision (QP): {qp:.2f}%")
    print("-" * 50)
    
    # 6. 提供与论文对比的直观参考
    print("📈 【论文对齐参考基线 - CRC100K】")
    print(" └─ 传统随机冷启动 (Random Selection) QP: 33.60%")
    print(" └─ 原论文 OpenPath (BioMedCLIP)    QP: 78.00%")
    if qp > 78.00:
        print(f" 🎉 恭喜！你的大模型生成式冷启动超越了 OpenPath 原论文基线 {qp - 78.00:.2f} 个百分点！")
    else:
        print(f" 💡 当前结果未超越原论文，可能需要优化 vlm_generative_ood.py 中的关键词过滤策略（Prompt）。")

if __name__ == "__main__":
    main()
    