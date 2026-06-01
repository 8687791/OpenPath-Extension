import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import random
import argparse
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
from sklearn.metrics.pairwise import cosine_similarity  # ⚡️ 算法 C 所需依赖

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
    基于标准 9 分类提示词的生成式阅片与高级过滤逻辑 (已加入实时数量监测)
    """
    id_candidates_names = []
    id_features = []
    ood_descriptions = []
    
    vqa_prompt = (
        "You are an expert pathologist. Classify this H&E pathology image into EXACTLY ONE of the following 9 distinct categories:\n"
        "1. LYM (Lymphocytes)\n"
        "2. NORM (Normal colon mucosa)\n"
        "3. TUM (Adenocarcinoma epithelium / Tumor)\n"
        "4. ADI (Adipose tissue)\n"
        "5. BACK (Background)\n"
        "6. DEB (Debris)\n"
        "7. MUC (Mucus)\n"
        "8. MUS (Smooth muscle)\n"
        "9. STR (Cancer-associated stroma)\n\n"
        "Respond strictly in this format:\n"
        "Category: [Input the exact category abbreviation from the list above, e.g., TUM or MUS]\n"
        "Description: [A brief one-sentence reason]"
    )
    
    gold_id_tags = ['tum', 'str', 'lym']
    transform = build_transform(input_size=448)
    print("\n🔍 开始基于 9-Way 临床单选的生成式阅片与 OOD 精准拦截...")
    
    pbar = tqdm(dataloader, desc="阅片进度")
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
        
        pbar.set_postfix({
            "ID_Found(TUM/STR/LYM)": len(id_candidates_names),
            "OOD_Blocked": len(ood_descriptions)
        })

    print(f"\n🌟 扫描完成！找到 {len(id_candidates_names)} 个已知类 (ID) 样本候选，发现 {len(ood_descriptions)} 个潜在的未知类 (OOD) 样本。")
    
    if len(ood_descriptions) > 0:
        print("🧬 正在对未知类 (OOD) 进行文本动态聚类以发现新病种...")
        corpus = [item["report"] for item in ood_descriptions]
        vectorizer = TfidfVectorizer(max_features=128, stop_words='english')
        X = vectorizer.fit_transform(corpus)
        num_clusters = max(2, min(5, len(corpus) // 3)) 
        
        if len(corpus) >= num_clusters:
            kmeans_ood = KMeans(n_clusters=num_clusters, random_state=42).fit(X)
            for i, item in enumerate(ood_descriptions):
                item["discovered_cluster"] = int(kmeans_ood.labels_[i])
                
        os.makedirs("al_file", exist_ok=True)
        with open("al_file/discovered_ood_clusters.json", "w", encoding='utf-8') as f:
            json.dump(ood_descriptions, f, indent=4, ensure_ascii=False)
        print("✅ OOD 聚类结果已保存至 al_file/discovered_ood_clusters.json")

    return np.array(id_candidates_names), np.vstack(id_features)


# ==================================================
# 🧪 算法 B: 密度自适应核心采样 (Density + FPS)
# ==================================================
def run_density_core_sampling(names, embeds, init_num, k_neighbors=5, noise_percentile=15):
    n_samples = embeds.shape[0]
    dot_product = np.dot(embeds, embeds.T)
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    dist_matrix = np.sqrt(np.maximum(norms**2 + norms.T**2 - 2 * dot_product, 0.0))
    
    knn_dists = np.sort(dist_matrix, axis=1)[:, 1:k_neighbors + 1].mean(axis=1)
    threshold = np.percentile(knn_dists, 100 - noise_percentile)
    valid_indices = np.where(knn_dists <= threshold)[0]
    
    if len(valid_indices) < init_num: 
        valid_indices = np.arange(n_samples)
    
    selected_indices = [valid_indices[np.argmin(knn_dists[valid_indices])]]
    min_dists = dist_matrix[selected_indices[0], :]
    
    while len(selected_indices) < init_num:
        candidate_indices = [i for i in valid_indices if i not in selected_indices]
        if not candidate_indices: break
        best_next = candidate_indices[np.argmax(min_dists[candidate_indices])]
        selected_indices.append(best_next)
        min_dists = np.minimum(min_dists, dist_matrix[best_next, :])
        
    return names[selected_indices].tolist()


# ==================================================
# 🧪 算法 C: 图论中心度与 NMS 抑制采样 (Graph Centrality)
# ==================================================
def run_graph_centrality_nms(names, embeds, init_num, sim_thresh=0.75, penalty=0.5):
    sim_matrix = cosine_similarity(embeds)
    sim_matrix[sim_matrix < sim_thresh] = 0 
    centrality = sim_matrix.sum(axis=1) 
    
    selected_indices = []
    valid_mask = np.ones(len(names), dtype=bool)
    
    for _ in range(init_num):
        masked_centrality = centrality * valid_mask
        if masked_centrality.max() <= 0:
            candidates = np.where(valid_mask)[0]
            best_idx = np.random.choice(candidates) if len(candidates) > 0 else 0
        else:
            best_idx = np.argmax(masked_centrality)
            
        selected_indices.append(best_idx)
        valid_mask[best_idx] = False
        
        centrality -= sim_matrix[best_idx] * penalty
        centrality = np.maximum(centrality, 0)
        
    return names[selected_indices].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--init_num", default=50, type=int)
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    # 可在此通过命令行参数指定核心采样策略：A (K-Means++), B (Density Core), C (Graph Centrality)
    parser.add_argument("--algo_mode", default="A", choices=["A", "B", "C"], type=str)
    args = parser.parse_args()
    
    seed_torch()

    print("📂 正在解析 al_file/train.csv 并加载数据池...")
    if not os.path.exists('al_file/train.csv'):
        print("❌ 错误：找不到 al_file/train.csv")
        return

    train_df = pd.read_csv('al_file/train.csv')
    
    if len(train_df) > 500:
        print(f"⚠️ 检测到原始数据量巨大 ({len(train_df)}张)。正在随机采样 500 张作为冷启动精选池...")
        train_df = train_df.sample(n=500, random_state=42).reset_index(drop=True)
    train_files = train_df.to_dict('records')
    
    train_dataset = Tumor_dataset_val(args, train_files)
    train_loader = get_loader(args, train_dataset, shuffle=False, drop=False, batch_size=1)
    
    model, tokenizer = load_internvl(model_path="pretrained/InternVL2_5-2B")
    
    names_id_vlm, embeds_id = generative_ood_discovery(args, train_loader, model, tokenizer)
    
    if len(names_id_vlm) == 0:
        print("❌ 错误：未发现任何匹配的 ID 样本！")
        return

    # 检测并截断处理
    if len(names_id_vlm) < args.init_num:
        print(f"⚠️ 警告：找到的 ID 候选样本({len(names_id_vlm)})少于预算({args.init_num})，全数选中！")
        selected_names = names_id_vlm.tolist()
    else:
        # 🧪 主动学习核心策略分流
        if args.algo_mode == "A":
            print("🎯 [策略 A] 正在对找到的 ID 样本进行多样性 K-Means++ 采样...")
            kmeans = KMeans(n_clusters=args.init_num, init='k-means++', n_init=10, random_state=42)
            kmeans.fit(embeds_id)
            selected_names = []
            for center in kmeans.cluster_centers_:
                distances = np.linalg.norm(embeds_id - center, axis=1)
                selected_names.append(names_id_vlm[np.argmin(distances)])
                
        elif args.algo_mode == "B":
            print("🎯 [策略 B] 正在执行密度自适应核心采样 (Density + FPS)...")
            selected_names = run_density_core_sampling(names_id_vlm, embeds_id, args.init_num)
            
        elif args.algo_mode == "C":
            print("🎯 [策略 C] 正在执行图论中心度与 NMS 抑制采样 (Graph Centrality)...")
            selected_names = run_graph_centrality_nms(names_id_vlm, embeds_id, args.init_num)
            
    # 将保存的文件名和打印路径更改为 query_round_4.csv
    os.makedirs("al_file", exist_ok=True)
    df_selected = pd.DataFrame({'img': selected_names, 'label': ['Unknown'] * len(selected_names)})
    df_selected.to_csv('al_file/query_round_10.csv', index=False)
    print(f"✅ 第四轮生成式冷启动挑选完成！[策略 {args.algo_mode}] 结果已保存至 al_file/query_round_10.csv")


if __name__ == '__main__':
    main()