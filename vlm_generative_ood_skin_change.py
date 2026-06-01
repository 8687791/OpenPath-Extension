import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import random
import argparse
import pickle  # ⚡️ 引入序列化工具，用于物理保存全量候选池
from tqdm import tqdm

# ==========================================
# 1. 环境与网络配置 (针对云容器环境优化)
# ==========================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# 导入原本的数据集加载器
from dataset.alb_dataset2 import Tumor_dataset_val, get_loader


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tensor_to_pil(tensor):
    inv_normalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    if tensor.dim() == 4:
        tensor = tensor[0]
    t = tensor.clone()
    t = inv_normalize(t)
    t = t.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(t)


def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def load_internvl(model_path="pretrained/InternVL2_5-2B"):
    print(f"🚀 正在加载轻量级 InternVL2.5-2B 基础大模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).eval().cuda()
    print("✅ 模型加载成功！")
    return model, tokenizer


def generative_ood_discovery(args, dataloader, model, tokenizer):
    """
    基于全新 10 分类皮肤疾病提示词的生成式阅片与高级过滤逻辑 (包含实时计数展示补丁)
    """
    id_candidates_names = []
    id_features = []
    ood_descriptions = []
    
    # 💡 10 大皮肤病单选题
    vqa_prompt = (
        "You are an expert dermatopathologist. Classify this skin lesion image into EXACTLY ONE of the following 10 distinct categories:\n"
        "1. Eczema (湿疹)\n"
        "2. Melanoma (黑色素瘤 - 恶性肿瘤)\n"
        "3. Atopic Dermatitis (特应性皮炎)\n"
        "4. Basal Cell Carcinoma (基底细胞癌)\n"
        "5. Melanocytic Nevi (色素痣)\n"
        "6. Benign Keratosis-like Lesions (良性角化病样皮损)\n"
        "7. Psoriasis / Lichen Planus (银屑病/扁平苔藓)\n"
        "8. Seborrheic Keratoses (脂溢性角化病)\n"
        "9. Fungal Infections / Tinea (真菌感染/癣)\n"
        "10. Viral Infections / Warts (病毒感染/疣)\n\n"
        "Respond strictly in this format:\n"
        "Category: [Input the exact category name or keywords from the list above, e.g., Melanoma or Melanocytic Nevi]\n"
        "Description: [A brief one-sentence reasoning focusing on lesion appearance]"
    )
    
    # 💡 核心修改点 2：锁定第 2 类 (Melanoma) 和第 5 类 (Nevi) 作为大模型拦截目标
    gold_id_tags = ['melanoma', 'melanocytic nevi', 'nevi']
    
    transform = build_transform(input_size=448)
    print("\n🔍 开始基于 10-Way 皮肤单选的生成式阅片与 OOD 精准拦截...")
    
    # ⚡️ 注入动态实时状态窗口
    pbar = tqdm(dataloader, desc="皮肤阅片进度")
    
    for idx, sample in enumerate(pbar):
        img_tensor = sample['img']
        img_name = sample['img_name'][0]
        
        pil_img = tensor_to_pil(img_tensor)
        pixel_values = transform(pil_img).unsqueeze(0).to(torch.bfloat16).cuda()
        
        with torch.no_grad():
            generation_config = dict(max_new_tokens=64, do_sample=False)
            generated_report = model.chat(tokenizer, pixel_values, vqa_prompt, generation_config)
            
            report_lines = generated_report.lower().split('\n')
            
            chosen_category = ""
            for line in report_lines:
                if "category:" in line:
                    chosen_category = line.replace("category:", "").strip()
                    break
            
            is_id = any(tag in chosen_category for tag in gold_id_tags)
            
            if is_id:
                vision_embeds = model.extract_feature(pixel_values).mean(dim=1).cpu().float().numpy()
                id_candidates_names.append(img_name)
                id_features.append(vision_embeds[0])
            else:
                ood_descriptions.append({
                    "img_name": img_name,
                    "report": generated_report
                })
        
        # ⚡️ 实时刷新看板
        pbar.set_postfix({
            "ID_Found": len(id_candidates_names),
            "OOD_Blocked": len(ood_descriptions)
        })

    print(f"\n🌟 皮肤扫描完成！找到 {len(id_candidates_names)} 个已知类 (ID) 样本候选，拦截 {len(ood_descriptions)} 个潜在的皮肤未知类 (OOD) 杂病样本。")
    
    if len(ood_descriptions) > 0:
        print("🧬 正在对拦截的皮肤 OOD 杂病进行文本动态聚类以发现隐性新疾病...")
        corpus = [item["report"] for item in ood_descriptions]
        vectorizer = TfidfVectorizer(max_features=128, stop_words='english')
        X = vectorizer.fit_transform(corpus)
        num_clusters = max(2, min(5, len(corpus) // 3)) 
        
        if len(corpus) >= num_clusters:
            kmeans_ood = KMeans(n_clusters=num_clusters, random_state=42).fit(X)
            for i, item in enumerate(ood_descriptions):
                item["discovered_cluster"] = int(kmeans_ood.labels_[i])
                
        os.makedirs(args.output_dir, exist_ok=True)
        out_json = os.path.join(args.output_dir, "discovered_ood_clusters_skin.json")
        with open(out_json, "w", encoding='utf-8') as f:
            json.dump(ood_descriptions, f, indent=4, ensure_ascii=False)
        print(f"✅ OOD 聚类结果已保存至 {out_json}")

    if len(id_features) == 0:
        return np.array([]), np.array([])

    return np.array(id_candidates_names), np.vstack(id_features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--init_num", default=50, type=int)
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    
    parser.add_argument("--train_csv", default="al_file_skin/train.csv", type=str)
    parser.add_argument("--output_dir", default="al_file_skin", type=str)
    args = parser.parse_args()
    
    seed_torch()

    print(f"📂 正在解析 {args.train_csv} 并加载皮肤未标注数据池...")
    if not os.path.exists(args.train_csv):
        print(f"❌ 错误：找不到 {args.train_csv}")
        return

    train_df = pd.read_csv(args.train_csv)
    
    # 临时建立快速查表字典，用于最后的 QP 线上核验
    truth_dict = {os.path.basename(row.iloc[0]): int(row.iloc[1]) for _, row in train_df.iterrows()}
    
    # 随机抽取 500 张图送入大模型精选池
    if len(train_df) > 500:
        print(f"⚠️ 检测到皮肤原始数据量庞大 ({len(train_df)}张)。正在随机采样 500 张作为冷启动精选池...")
        train_df = train_df.sample(n=500, random_state=42).reset_index(drop=True)
    train_files = train_df.to_dict('records')
    
    train_dataset = Tumor_dataset_val(args, train_files)
    train_loader = get_loader(args, train_dataset, shuffle=False, drop=False, batch_size=1)
    
    model, tokenizer = load_internvl(model_path="pretrained/InternVL2_5-2B")
    
    names_id_vlm, embeds_id = generative_ood_discovery(args, train_loader, model, tokenizer)
    
    if len(names_id_vlm) == 0:
        print("❌ 错误：大模型未在探索池中发现任何匹配的 ID 样本！")
        return

    gold_id_classes = [1, 4] 

    # ⚡️ 核心物理保存：将大模型初筛出的【全量候选池】的路径名字和视觉特征 Embedding 固化存盘
    os.makedirs(args.output_dir, exist_ok=True)
    save_cache_path = os.path.join(args.output_dir, "vlm_id_candidates.pkl")
    with open(save_cache_path, "wb") as f:
        pickle.dump({"names": names_id_vlm, "embeds": embeds_id}, f)
    print(f"💾 [科研存盘成功]：已将全量初筛池数据固化到本地: {save_cache_path}")

    # 📊 --- 阶段 1: 大模型初筛全量候选池纯度 (QA/QP) 评测 ---
    print("\n" + "="*50)
    print(f"📊 --- 阶段 1: 大模型初筛全量候选池纯度 (QA/QP) 评测 ---")
    vlm_raw_id_count = 0
    vlm_raw_ood_count = 0
    
    for name in names_id_vlm:
        img_basename = os.path.basename(name)
        if img_basename in truth_dict:
            real_label = truth_dict[img_basename]
            if real_label in gold_id_classes:
                vlm_raw_id_count += 1
            else:
                vlm_raw_ood_count += 1
                
    raw_qp = (vlm_raw_id_count / len(names_id_vlm)) * 100 if len(names_id_vlm) > 0 else 0
    print(f" ├─ 大模型初筛总候选数: {len(names_id_vlm)}")
    print(f" ├─ 真正目标类 (ID: 1=Mel, 4=Nevi) 数量: {vlm_raw_id_count}")
    print(f" ├─ 混入错误未知类 (OOD 噪声) 数量: {vlm_raw_ood_count}")
    print(f" └─ 💡 原始初筛池精准度 (Raw Query Precision): {raw_qp:.2f}%")
    print("="*50 + "\n")

    # 阶段 2 ———— 原始 K-Means++ 聚类控量多样性挑选
    if len(names_id_vlm) < args.init_num:
        print(f"⚠️ 警告：找到的 ID 候选样本({len(names_id_vlm)})少于当前 {args.init_num} 张的预算，将全数选中！")
        selected_names = names_id_vlm.tolist()
    else:
        print("🎯 正在对找到的皮肤黄金 ID 样本进行多样性 K-Means++ 采样...")
        kmeans = KMeans(n_clusters=args.init_num, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(embeds_id)
        
        selected_names = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(embeds_id - center, axis=1)
            selected_names.append(names_id_vlm[np.argmin(distances)])
            
    # 结果落盘
    df_selected = pd.DataFrame({'img': selected_names, 'label': ['Unknown'] * len(selected_names)})
    out_csv = os.path.join(args.output_dir, "query_round_5.csv")
    df_selected.to_csv(out_csv, index=False)
    print(f"✅ 皮肤种子集结果已保存在 {out_csv}")

    # 📊 --- 阶段 2: 最终 50 张精选冷启动种子进行核验
    print("\n" + "="*50)
    print(f"📊 --- 阶段 2: 最终 50 张精选冷启动种子精准度 (QP) 战报 ---")
    
    final_id_count = 0
    final_ood_count = 0
    
    for name in selected_names:
        img_basename = os.path.basename(name)
        if img_basename in truth_dict:
            real_label = truth_dict[img_basename]
            if real_label in gold_id_classes:
                final_id_count += 1
            else:
                final_ood_count += 1
                
    final_qp = (final_id_count / len(selected_names)) * 100
    print(f" ├─ 采样总目标数: {len(selected_names)}")
    print(f" ├─ 真正目标类 (ID: 1=Mel, 4=Nevi) 数量: {final_id_count}")
    print(f" ├─ 误选未知类 (OOD 噪声) 数量: {final_ood_count}")
    print(f" └─ 💡 聚类精选后查询精准度 Query Precision (QP): {final_qp:.2f}%")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()