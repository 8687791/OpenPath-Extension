import pandas as pd

# 1. 读取大模型刚挑出的 56 个种子
df_query = pd.read_csv('al_file/query_round_10.csv')

# 2. 读取你的皮肤大账本真值（假设是 al_file_skin/train.csv）
df_gt = pd.read_csv('al_file/train.csv')

# 3. 建立真值字典
gt_dict = dict(zip(df_gt.iloc[:, 0], df_gt.iloc[:, 1]))

# 4. 统计在这 56 张图里，真正属于已知类（比如类索引 1 和 4）的比例
id_cls = [1, 4]  # 你的已知类标签
true_positives = 0

for img_path in df_query['img']:
    if gt_dict.get(img_path) in id_cls:
        true_positives += 1

qa_precision = true_positives / len(df_query)
print(f"📊 [QA 质量报告] 初始种子集绝对纯度 (Query Precision): {qa_precision:.2%}")
print(f"   └─ 56张样本中，真正的 ID 数量: {true_positives} 张，误入的 OOD 噪声: {len(df_query) - true_positives} 张。")