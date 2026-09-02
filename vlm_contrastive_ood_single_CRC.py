import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import random
import argparse
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer, AutoModel

# 导入原始数据集加载器
from dataset.alb_dataset2 import Tumor_dataset_val, get_loader

# ==========================================
# 1. 环境与网络配置
# ==========================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

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

def load_internvl(model_path="pretrained/InternVL2_5-8B"):
    print(f"🚀 正在加载 InternVL2.5-8B 大模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    print("✅ 模型加载成功！")
    return model, tokenizer

# ==================================================
# 判定单个样本是否属于目标类（用于实时与最终计算）
# ==================================================
def is_sample_ground_truth(name, true_labels_dict, target_class):
    target_upper = target_class.upper()
    for csv_path in true_labels_dict.keys():
        if str(name) in str(csv_path):
            path_upper = str(csv_path).upper()
            if f"/{target_upper}/" in path_upper or f"{target_upper}-" in path_upper:
                return True
            break
    return False

# ==================================================
# 2. 核心大模型发现逻辑 (批量化 Batched 推理 + 实时 QP)
# ==================================================
def generative_ood_discovery(args, dataloader, model, tokenizer, target_class, true_labels_dict):
    id_candidates_names = []
    id_features = []
    ood_descriptions = []
    true_id_hits = 0  # 记录实时命中真阳性样本数
    
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
    
    target_keywords = {
        'lym': ['lym', 'lymphocyte'],
        'norm': ['norm', 'mucosa'],
        'tum': ['tum', 'adenocarcinoma', 'tumor'],
        'adi': ['adi', 'adipose'],
        'back': ['back', 'background'],
        'deb': ['deb', 'debris'],
        'muc': ['muc', 'mucus'],
        'mus': ['mus', 'smooth muscle'],
        'str': ['str', 'stroma']
    }

    transform = build_transform(input_size=448)
    print(f"\n🔍 开始基于 9-Way 临床单选的生成式阅片 (当前寻找目标类: {target_class.upper()})...")
    
    total_images_processed = 0
    target_device = model.device if hasattr(model, 'device') else torch.device("cuda:0")
    
    pbar = tqdm(dataloader, desc=f"阅片进度 [{target_class.upper()}]")
    
    for batch_idx, sample in enumerate(pbar):
        img_tensors = sample['img']      # [Batch, C, H, W]
        img_names = sample['img_name']   # List of Batch strings
        
        batch_size_current = len(img_names)
        
        # 预处理当前 Batch 的所有图片
        pixel_values_list = []
        for i in range(batch_size_current):
            single_img_tensor = img_tensors[i] if isinstance(img_tensors, list) else img_tensors[i]
            pil_img = tensor_to_pil(single_img_tensor)
            pv = transform(pil_img)
            pixel_values_list.append(pv)
            
        # 拼接并送入显存
        pixel_values = torch.stack(pixel_values_list).to(torch.bfloat16).to(target_device)
        
        prompts = [vqa_prompt] * batch_size_current
        num_patches_list = [1] * batch_size_current  
        generation_config = dict(max_new_tokens=64, do_sample=False)
        
        with torch.no_grad():
            try:
                # 优先调用官方批量推理 batch_chat
                generated_reports = model.batch_chat(
                    tokenizer, 
                    pixel_values=pixel_values, 
                    num_patches_list=num_patches_list,
                    questions=prompts, 
                    generation_config=generation_config
                )
            except (AttributeError, TypeError):
                generated_reports = []
                for i in range(batch_size_current):
                    rep = model.chat(tokenizer, pixel_values[i:i+1], vqa_prompt, generation_config)
                    generated_reports.append(rep)
                    
            vision_embeds_batch = model.extract_feature(pixel_values).mean(dim=1).cpu().float().numpy()
            
        # 遍历 Batch 结果
        for i in range(batch_size_current):
            img_name = img_names[i]
            generated_report = generated_reports[i]
            report_lower = generated_report.lower()
            vision_embeds = vision_embeds_batch[i]
            
            is_id = False
            for line in report_lower.split('\n'):
                if "category" in line:
                    for keyword in target_keywords[target_class.lower()]:
                        if keyword in line:
                            is_id = True
                            break
                if is_id: break
            
            if not is_id:
                for keyword in target_keywords[target_class.lower()]:
                    if keyword in report_lower.split('\n')[0]:
                        is_id = True
                        break

            if total_images_processed < 2:
                print(f"\n[诊断图 {total_images_processed+1}] 目标: {target_class.upper()}")
                print(f"模型输出:\n{generated_report}")
                print(f"匹配结果: {'✅ 命中' if is_id else '❌ 未命中'}\n")
            
            if is_id:
                id_candidates_names.append(img_name)
                id_features.append(vision_embeds)
                # 检验真阳性
                if is_sample_ground_truth(img_name, true_labels_dict, target_class):
                    true_id_hits += 1
            else:
                ood_descriptions.append({
                    "img_name": img_name,
                    "report": generated_report
                })
            
            total_images_processed += 1

        # 实时刷新进度条中的候选量与初筛 QP
        found_num = len(id_candidates_names)
        realtime_qp = (true_id_hits / found_num * 100) if found_num > 0 else 0.0
        pbar.set_postfix({
            "ID_Found": found_num,
            "Real_Hits": true_id_hits,
            "Realtime_QP": f"{realtime_qp:.2f}%",
            "OOD_Blocked": len(ood_descriptions)
        })

    raw_final_qp = (true_id_hits / len(id_candidates_names) * 100) if len(id_candidates_names) > 0 else 0.0
    print(f"🌟 扫描完成！类别 {target_class.upper()} 初筛命中 {len(id_candidates_names)} 个，原始准确率 (Raw QP): {raw_final_qp:.2f}%")
    
    # 动态 OOD 聚类保存
    if len(ood_descriptions) > 0:
        corpus = [item["report"] for item in ood_descriptions]
        vectorizer = TfidfVectorizer(max_features=128, stop_words='english')
        X = vectorizer.fit_transform(corpus)
        num_clusters = max(2, min(5, len(corpus) // 3)) 
        
        if len(corpus) >= num_clusters:
            kmeans_ood = KMeans(n_clusters=num_clusters, random_state=42).fit(X)
            for i, item in enumerate(ood_descriptions):
                item["discovered_cluster"] = int(kmeans_ood.labels_[i])
                
        os.makedirs("al_file", exist_ok=True)
        with open(f"al_file/discovered_ood_clusters_{target_class}.json", "w", encoding='utf-8') as f:
            json.dump(ood_descriptions, f, indent=4, ensure_ascii=False)

    features_array = np.vstack(id_features) if len(id_features) > 0 else np.array([])
    return np.array(id_candidates_names), features_array, raw_final_qp

# ==================================================
# 🧪 算法 A: K-Means++ 多样性采样
# ==================================================
def run_kmeans_sampling(names, embeds, init_num):
    if len(names) <= init_num:
        return names.tolist()
    kmeans = KMeans(n_clusters=init_num, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(embeds)
    
    selected_indices = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(embeds - center, axis=1)
        selected_indices.append(np.argmin(distances))
    return names[selected_indices].tolist()

# ==================================================
# 🧪 算法 B: 密度自适应核心采样 (Density + FPS)
# ==================================================
def run_density_core_sampling(names, embeds, init_num, k_neighbors=5, noise_percentile=15):
    if len(names) <= init_num:
        return names.tolist()
        
    n_samples = embeds.shape[0]
    dot_product = np.dot(embeds, embeds.T)
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    dist_matrix = np.sqrt(np.maximum(norms**2 + norms.T**2 - 2 * dot_product, 0.0))
    
    knn_dists = np.sort(dist_matrix, axis=1)[:, 1:min(k_neighbors + 1, n_samples)].mean(axis=1)
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
    if len(names) <= init_num:
        return names.tolist()
        
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

# ==================================================
# 📊 评估工具：计算 QA Accuracy
# ==================================================
def calculate_accuracy(selected_names, true_labels_dict, target_class):
    if not selected_names:
        return 0.0, 0
    
    correct = 0
    for name in selected_names:
        if is_sample_ground_truth(name, true_labels_dict, target_class):
            correct += 1
                
    accuracy = (correct / len(selected_names)) * 100
    return accuracy, correct

# ==================================================
# 🚀 主控制流
# ==================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int, help="8B 模型显存优化批次大小")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--init_num", default=50, type=int)
    parser.add_argument("--sample_size", default=1000, type=int, help="单类采样池规模（设为 -1 扫描全量）")
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--model_path", default="pretrained/InternVL2_5-8B", type=str)
    args = parser.parse_args()
    
    seed_torch()
    
    print("📂 正在解析 al_file/train.csv 并加载全量数据池...")
    if not os.path.exists('al_file/train.csv'):
        print("❌ 错误：找不到 al_file/train.csv")
        return
        
    full_train_df = pd.read_csv('al_file/train.csv', header=None, names=['img', 'label'])
    true_labels_dict = dict(zip(full_train_df['img'], full_train_df['label']))

    # 加载 8B 模型
    model, tokenizer = load_internvl(model_path=args.model_path)
    
    categories = ['lym', 'norm', 'tum', 'adi', 'back', 'deb', 'muc', 'mus', 'str']
    
    os.makedirs("al_file", exist_ok=True)
    report_path = "al_file/experiment_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("      🎯 多类独立主动学习筛选报告 ( QA 评估 - InternVL2.5-8B )      \n")
        f.write("="*60 + "\n\n")

    summary_list = []

    for target_class in categories:
        print(f"\n" + "="*50)
        print(f" ▶ 开始处理类别: {target_class.upper()}")
        print(f"="*50)
        
        if args.sample_size > 0 and len(full_train_df) > args.sample_size:
            subset_df = full_train_df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        else:
            subset_df = full_train_df
            
        subset_files = subset_df.to_dict('records')
        
        subset_dataset = Tumor_dataset_val(args, subset_files)
        subset_loader = get_loader(args, subset_dataset, shuffle=False, drop=False, batch_size=args.batch_size)
        
        names_id_vlm, embeds_id, raw_qp = generative_ood_discovery(
            args, subset_loader, model, tokenizer, target_class, true_labels_dict
        )
        
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"【 类别: {target_class.upper()} 】\n")
            f.write(f" - 大模型初筛认定目标类数量 (Pool): {len(names_id_vlm)} | 原始初筛纯度 (Raw QP): {raw_qp:.2f}%\n")
            
            if len(names_id_vlm) == 0:
                f.write(" - ⚠️ 未能找到任何属于该类的样本。\n\n")
                summary_list.append({"class": target_class.upper(), "pool": 0, "raw_qp": 0.0, "acc_a": 0.0, "acc_b": 0.0, "acc_c": 0.0})
                continue

            res_kmeans = run_kmeans_sampling(names_id_vlm, embeds_id, args.init_num)
            acc_a, correct_a = calculate_accuracy(res_kmeans, true_labels_dict, target_class)
            
            res_density = run_density_core_sampling(names_id_vlm, embeds_id, args.init_num)
            acc_b, correct_b = calculate_accuracy(res_density, true_labels_dict, target_class)
            
            res_graph = run_graph_centrality_nms(names_id_vlm, embeds_id, args.init_num)
            acc_c, correct_c = calculate_accuracy(res_graph, true_labels_dict, target_class)
            
            f.write(f"   [算法 A: K-Means++] 抽取 {len(res_kmeans)} 张 | 命中 {correct_a} 张 | QA 准确率: {acc_a:.2f}%\n")
            f.write(f"   [算法 B: DensityCore] 抽取 {len(res_density)} 张 | 命中 {correct_b} 张 | QA 准确率: {acc_b:.2f}%\n")
            f.write(f"   [算法 C: Graph + NMS] 抽取 {len(res_graph)} 张 | 命中 {correct_c} 张 | QA 准确率: {acc_c:.2f}%\n")
            f.write("-" * 60 + "\n\n")
            
            summary_list.append({
                "class": target_class.upper(), 
                "pool": len(names_id_vlm), 
                "raw_qp": raw_qp, 
                "acc_a": acc_a, 
                "acc_b": acc_b, 
                "acc_c": acc_c
            })
            
        print(f"✅ {target_class.upper()} 评估完成: Raw QP={raw_qp:.2f}% | A={acc_a:.2f}% | B={acc_b:.2f}% | C={acc_c:.2f}%")

    # 控制台打印最终汇总表格
    print("\n" + "="*70)
    print("📊 最终多类别全量抽取评测汇总表")
    print("="*70)
    print(f"{'Class':<8}{'Pool':<8}{'Raw QP':<12}{'K-Means++':<14}{'DensityCore':<14}{'Graph+NMS':<14}")
    for item in summary_list:
        print(f"{item['class']:<8}{item['pool']:<8}{item['raw_qp']:<12.2f}{item['acc_a']:<14.2f}{item['acc_b']:<14.2f}{item['acc_c']:<14.2f}")
    print("="*70)
    print(f"🎉 全部分类测试结束！详细 QA 测试指标已保存至: {report_path}")

if __name__ == '__main__':
    main()