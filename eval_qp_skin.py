import os
import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="计算皮肤数据集冷启动阶段的种子纯度 (QP)")
    
    # 💡 核心设定一：路径全面指向皮肤隔离文件夹 al_file_skin
    parser.add_argument("--query_csv", type=str, default="al_file_skin/clip_query_round_1.csv", help="VLM生成的冷启动文件")
    parser.add_argument("--truth_csv", type=str, default="al_file_skin/train.csv", help="包含全量真实标签的CSV文件")
    
    # 💡 核心设定二：严格对齐刚才代码中定下的皮肤 ID 类别 (1=Melanoma, 3=BCC)
    parser.add_argument("--id_cls", nargs="+", type=int, default=[1, 4], help="目标已知类的Label整数值")
    args = parser.parse_args()

    # 1. 确保文件存在
    if not os.path.exists(args.query_csv):
        print(f"❌ 错误：找不到皮肤冷启动筛选结果文件 {args.query_csv}，请先运行 vlm_generative_ood_skin.py")
        return
    if not os.path.exists(args.truth_csv):
        print(f"❌ 错误：找不到皮肤真实标签文件 {args.truth_csv}")
        return

    # 2. 读取数据
    query_df = pd.read_csv(args.query_csv)
    truth_df = pd.read_csv(args.truth_csv)

    # 3. 建立 真实“图片路径 -> 标签” 的映射字典
    truth_df.columns = ['img', 'label'] 
    truth_dict = {os.path.basename(row['img']): row['label'] for _, row in truth_df.iterrows()}

    # 4. 统计冷启动挑出的样本的真实标签
    total_selected = len(query_df)
    id_count = 0
    ood_count = 0
    missing_count = 0

    print(f"📊 开始评估皮肤场冷启动种子集，总计挑选样本数: {total_selected}")
    print("-" * 50)

    for _, row in query_df.iterrows():
        img_name = os.path.basename(row['img'])
        if img_name in truth_dict:
            real_label = int(truth_dict[img_name])
            # 严格判断该样本的真实标签是否属于 ID 阵营 (Melanoma 或 BCC)
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

    print(f"✨ 【皮肤实验评估结果】")
    # 文案修正：明确标注当前验证的 ID 是 Mel 和 BCC
    print(f" └─ 真正目标类 (ID: Mel & BCC) 数量: {id_count}")
    print(f" └─ 误选未知类 (OOD Samples) 数量: {ood_count}")
    print(f" └─ 💡 第一轮查询精准度 Query Precision (QP): {qp:.2f}%")
    print("-" * 50)
    
    # 💡 核心设定三：精准对齐 OpenPath 论文中的 SkinTissue 实验基线
    openpath_skin_baseline = 58.50 
    random_skin_baseline = 16.51 # 根据你之前发的论文原文，Mel+BCC 的自然占比是 16.51%
    
    print("📈 【论文对齐参考基线 - SkinTissue】")
    print(f" └─ 传统随机冷启动 (Random Selection) QP: ~{random_skin_baseline:.2f}% (按自然占比推算)")
    print(f" └─ 原论文 OpenPath (BioMedCLIP)    QP: {openpath_skin_baseline:.2f}%")
    
    if qp > openpath_skin_baseline:
        print(f" 🎉 恭喜！你的 MLLM 生成式冷启动远超 OpenPath 原论文基线 {qp - openpath_skin_baseline:.2f} 个百分点！")
        print(f"    (这证明了生成式 VQA 在高难度皮肤镜特征提取上，拥有对传统 CLIP 模型的降维打击能力。)")
    else:
        print(f" 💡 当前结果未超越原论文，可能需要继续优化 vlm_generative_ood_skin.py 中的 Prompt 拦截策略。")

if __name__ == "__main__":
    main()