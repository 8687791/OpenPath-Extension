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
    print(f"🚀 正在加载 InternVL2.5-8B 旗舰大模型...")
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

def is_sample_ground_truth(name, true_labels_dict, target_class="str"):
    target_upper = target_class.upper()
    for csv_path in true_labels_dict.keys():
        if str(name) in str(csv_path):
            path_upper = str(csv_path).upper()
            if f"/{target_upper}/" in path_upper or f"{target_upper}-" in path_upper:
                return True
            break
    return False

# ==================================================
# 2. STR 专精生成式阅片与特征提取
# ==================================================
def generative_ood_discovery_str(args, dataloader, model, tokenizer, true_labels_dict):
    id_candidates_names = []
    id_features = []
    ood_descriptions = []
    raw_hits = 0

    # 🔬 专门强化 STR 与其它混淆类（MUS 平滑肌、DEB 坏死碎屑）的区分
    vqa_prompt = (
        "You are an expert surgical pathologist specializing in colorectal cancer histology.\n"
        "Classify this H&E pathology image into EXACTLY ONE of the following 9 categories:\n"
        "1. STR (Cancer-associated stroma: desmoplastic, collagenous fibrous tissue with elongated spindle fibroblasts around tumor)\n"
        "2. MUS (Smooth muscle: dense parallel bundles of smooth muscle fibers with blunt-ended cigar-shaped nuclei, NOT desmoplastic stroma)\n"
        "3. DEB (Debris / Necrosis: amorphous pink necrotic breakdown or nuclear dust)\n"
        "4. TUM (Adenocarcinoma epithelium / Malignant glands)\n"
        "5. ADI (Adipose tissue / Fat vacuoles)\n"
        "6. BACK (Background / Clear glass slide)\n"
        "7. LYM (Lymphocytes / Dense lymphoid aggregates)\n"
        "8. MUC (Mucus / Pale mucin pools)\n"
        "9. NORM (Normal mucosa crypts)\n\n"
        "Respond STRICTLY in this two-line format:\n"
        "Category: <3-LETTER-ABBREVIATION>\n"
        "Description: <A brief one-sentence histological reason>"
    )
    
    # 严格限定 STR 的识别词，剔除容易泛化的词（如普通 connective）
    str_keywords = ['str', 'stroma', 'cancer-associated stroma', 'desmoplasia', 'desmoplastic']

    transform = build_transform(input_size=448)
    print(f"\n🔍 开始针对【STR (Cancer-associated stroma)】进行专精生成式扫描...")
    
    target_device = model.device if hasattr(model, 'device') else torch.device("cuda:0")
    total_images_processed = 0
    pbar = tqdm(dataloader, desc="STR 扫描进度")
    
    for batch_idx, sample in enumerate(pbar):
        img_tensors = sample['img']      
        img_names = sample['img_name']   
        batch_size_current = len(img_names)
        
        pixel_values_list = []
        for i in range(batch_size_current):
            single_img_tensor = img_tensors[i] if isinstance(img_tensors, list) else img_tensors[i]
            pil_img = tensor_to_pil(single_img_tensor)
            pv = transform(pil_img)
            pixel_values_list.append(pv)
            
        pixel_values = torch.stack(pixel_values_list).to(torch.bfloat16).to(target_device)
        prompts = [vqa_prompt] * batch_size_current
        num_patches_list = [1] * batch_size_current  
        generation_config = dict(max_new_tokens=64, do_sample=False)
        
        with torch.no_grad():
            try:
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
            
        for i in range(batch_size_current):
            img_name = img_names[i]
            generated_report = generated_reports[i]
            report_lower = generated_report.lower()
            vision_embeds = vision_embeds_batch[i]
            
            is_str = False
            cat_line = ""
            for line in report_lower.split('\n'):
                if "category" in line:
                    cat_line = line
                    break
            
            if not cat_line and len(report_lower.split('\n')) > 0:
                cat_line = report_lower.split('\n')[0]
                
            clean_cat = cat_line.replace(':', ' ').replace('(', ' ').replace(')', ' ').replace('-', ' ')
            words = clean_cat.split()
            
            for keyword in str_keywords:
                if keyword in words or keyword == cat_line.replace("category:", "").strip():
                    is_str = True
                    break

            if total_images_processed < 2:
                print(f"\n[诊断图 {total_images_processed+1}] 目标: STR")
                print(f"模型输出:\n{generated_report}")
                print(f"匹配结果: {'✅ 命中 STR' if is_str else '❌ 拦截/其他类别'}\n")
            
            if is_str:
                id_candidates_names.append(img_name)
                id_features.append(vision_embeds)
                if is_sample_ground_truth(img_name, true_labels_dict, target_class="str"):
                    raw_hits += 1
            else:
                ood_descriptions.append({
                    "img_name": img_name,
                    "report": generated_report
                })
            
            total_images_processed += 1

        found_num = len(id_candidates_names)
        realtime_raw_qp = (raw_hits / found_num * 100) if found_num > 0 else 0.0
        pbar.set_postfix({
            "STR_Pool": found_num,
            "Real_Hits": raw_hits,
            "Realtime_Raw_QP": f"{realtime_raw_qp:.2f}%",
            "OOD_Blocked": len(ood_descriptions)
        })

    final_raw_qp = (raw_hits / len(id_candidates_names) * 100) if len(id_candidates_names) > 0 else 0.0
    print(f"\n🌟 STR 扫描完成！初筛池大小: {len(id_candidates_names)} 张 | 真实命中: {raw_hits} 张 | 初筛纯度 (Raw QP): {final_raw_qp:.2f}%")
    
    if len(ood_descriptions) > 0:
        os.makedirs("al_file", exist_ok=True)
        with open("al_file/discovered_ood_clusters_str.json", "w", encoding='utf-8') as f:
            json.dump(ood_descriptions, f, indent=4, ensure_ascii=False)

    features_array = np.vstack(id_features) if len(id_features) > 0 else np.array([])
    return np.array(id_candidates_names), features_array, final_raw_qp, raw_hits

# ==================================================
# 🧪 三轨主动学习采样算法
# ==================================================
def run_kmeans_sampling(names, embeds, init_num):
    if len(names) <= init_num: return names.tolist()
    kmeans = KMeans(n_clusters=init_num, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(embeds)
    selected_indices = [np.argmin(np.linalg.norm(embeds - center, axis=1)) for center in kmeans.cluster_centers_]
    return names[selected_indices].tolist()

def run_density_core_sampling(names, embeds, init_num, k_neighbors=5, noise_percentile=15):
    if len(names) <= init_num: return names.tolist()
    n_samples = embeds.shape[0]
    dot_product = np.dot(embeds, embeds.T)
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    dist_matrix = np.sqrt(np.maximum(norms**2 + norms.T**2 - 2 * dot_product, 0.0))
    
    knn_dists = np.sort(dist_matrix, axis=1)[:, 1:min(k_neighbors + 1, n_samples)].mean(axis=1)
    threshold = np.percentile(knn_dists, 100 - noise_percentile)
    valid_indices = np.where(knn_dists <= threshold)[0]
    
    if len(valid_indices) < init_num: valid_indices = np.arange(n_samples)
    
    selected_indices = [valid_indices[np.argmin(knn_dists[valid_indices])]]
    min_dists = dist_matrix[selected_indices[0], :]
    
    while len(selected_indices) < init_num:
        candidate_indices = [i for i in valid_indices if i not in selected_indices]
        if not candidate_indices: break
        best_next = candidate_indices[np.argmax(min_dists[candidate_indices])]
        selected_indices.append(best_next)
        min_dists = np.minimum(min_dists, dist_matrix[best_next, :])
        
    return names[selected_indices].tolist()

def run_graph_centrality_nms(names, embeds, init_num, sim_thresh=0.75, penalty=0.5):
    if len(names) <= init_num: return names.tolist()
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

def calculate_accuracy(selected_names, true_labels_dict, target_class="str"):
    if not selected_names: return 0.0, 0
    correct = sum([1 for name in selected_names if is_sample_ground_truth(name, true_labels_dict, target_class)])
    return (correct / len(selected_names)) * 100, correct

# ==================================================
# 🚀 主函数
# ==================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int) 
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--init_num", default=50, type=int)
    parser.add_argument("--sample_size", default=800, type=int, help="抽取 STR 时采样的盲盒池大小")
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--model_path", default="pretrained/InternVL2_5-8B", type=str)
    args = parser.parse_args()
    
    print("📂 正在解析 al_file/train.csv 并加载数据池...")
    if not os.path.exists('al_file/train.csv'):
        print("❌ 错误：找不到 al_file/train.csv")
        return
        
    full_train_df = pd.read_csv('al_file/train.csv', header=None, names=['img', 'label'])
    true_labels_dict = dict(zip(full_train_df['img'], full_train_df['label']))

    # 加载 8B 模型
    model, tokenizer = load_internvl(model_path=args.model_path)
    
    os.makedirs("al_file", exist_ok=True)
    report_path = "al_file/str_experiment_report.txt"

    current_sample_size = min(args.sample_size, len(full_train_df))
    subset_df = full_train_df.sample(n=current_sample_size).reset_index(drop=True)
    subset_files = subset_df.to_dict('records')
    
    subset_dataset = Tumor_dataset_val(args, subset_files)
    subset_loader = get_loader(args, subset_dataset, shuffle=False, drop=False, batch_size=args.batch_size)
    
    names_id_vlm, embeds_id, raw_qp, raw_hits = generative_ood_discovery_str(
        args, subset_loader, model, tokenizer, true_labels_dict
    )
    
    if len(names_id_vlm) == 0:
        print("❌ 警告：未发现任何被判定为 STR 的样本！")
        return

    # 运行三轨算法抽取
    res_kmeans = run_kmeans_sampling(names_id_vlm, embeds_id, args.init_num)
    acc_a, correct_a = calculate_accuracy(res_kmeans, true_labels_dict, "str")
    
    res_density = run_density_core_sampling(names_id_vlm, embeds_id, args.init_num)
    acc_b, correct_b = calculate_accuracy(res_density, true_labels_dict, "str")
    
    res_graph = run_graph_centrality_nms(names_id_vlm, embeds_id, args.init_num)
    acc_c, correct_c = calculate_accuracy(res_graph, true_labels_dict, "str")
    
    # 落盘保存选中的样本列表供下游训练使用
    pd.DataFrame({'img': res_kmeans, 'label': ['STR'] * len(res_kmeans)}).to_csv("al_file/str_selected_kmeans.csv", index=False)
    pd.DataFrame({'img': res_density, 'label': ['STR'] * len(res_density)}).to_csv("al_file/str_selected_density.csv", index=False)
    pd.DataFrame({'img': res_graph, 'label': ['STR'] * len(res_graph)}).to_csv("al_file/str_selected_graph.csv", index=False)

    report_text = f"""
=================================================================
       🎯 STR (Cancer-Associated Stroma) 独立抽取评测战报
=================================================================
📦 随机盲盒测试池规模: {current_sample_size} 张
🔍 VLM (InternVL2.5-8B) 初筛候选池: {len(names_id_vlm)} 张
💡 原始初筛精准度 (Raw QP): {raw_qp:.2f}% (真实命中: {raw_hits}/{len(names_id_vlm)})
-----------------------------------------------------------------
🏆 三轨冷启动采样算法最终战报 (目标抽取数: {args.init_num} 张):
 🔬 [算法 A: K-Means++]     抽取: {len(res_kmeans)} | 命准: {correct_a} | 最终 QA: {acc_a:.2f}%
 🔬 [算法 B: DensityCore]   抽取: {len(res_density)} | 命准: {correct_b} | 最终 QA: {acc_b:.2f}%
 🔬 [算法 C: Graph + NMS]   抽取: {len(res_graph)} | 命准: {correct_c} | 最终 QA: {acc_c:.2f}%
=================================================================
"""
    print(report_text)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"✅ STR 独立抽取完成！详细战报已保存至: {report_path}")
    print(f"📁 选取的冷启动种子集已分别落盘至 al_file/str_selected_*.csv")

if __name__ == '__main__':
    main()