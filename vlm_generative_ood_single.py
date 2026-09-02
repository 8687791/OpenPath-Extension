import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import random
import argparse
import ctypes
import datetime
from tqdm import tqdm

# ==========================================
# 0. 核心底层死锁绝杀补丁
# ==========================================
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

for path in ['/usr/lib/x86_64-linux-gnu/libcuda.so', '/usr/lib/x86_64-linux-gnu/libcuda.so.1', '/usr/local/cuda/compat/lib/libcuda.so.1']:
    if os.path.exists(path):
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            break
        except:
            pass

os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"

# ==========================================
# 1. 环境与网络配置
# ==========================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dataset.alb_dataset2 import Tumor_dataset_val, get_loader

def seed_torch(seed=3407):
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
    print(f"🚀 正在加载旗舰级 InternVL2.5-8B 大模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    # 8B 模型适配：利用 device_map='auto' 支持单卡或多卡自动显存分配
    model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    
    print("✅ InternVL2.5-8B 模型加载成功！")
    return model, tokenizer

def generative_ood_discovery(args, dataloader, model, tokenizer, task_name, gold_id_tags):
    id_candidates_names = []
    id_features = []
    ood_descriptions = []
    
    # ✨ 强制要求不确定时输出 Other/Unknown，防止瞎猜降低纯度
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
        "If the image is completely ambiguous or clearly does not match any of these 10 categories, you MUST output 'Category: Other/Unknown'.\n\n"
        "Respond strictly in this format:\n"
        "Category: [Input the exact category name or keywords from the list above, e.g., Melanoma or Melanocytic Nevi]\n"
        "Description: [A brief one-sentence reasoning focusing on lesion appearance]"
    )
    
    transform = build_transform(input_size=448)
    print(f"\n🔍 [任务: {task_name}] 开始基于生成式阅片与 OOD 拦截 (触发词: {gold_id_tags})...")
    
    pbar = tqdm(dataloader, desc=f"扫描进度")
    
    # 获取模型实际主显卡设备（兼容 device_map）
    target_device = model.device if hasattr(model, 'device') else torch.device("cuda:0")
    
    for idx, sample in enumerate(pbar):
        img_tensor = sample['img']
        img_name = sample['img_name'][0]
        
        pil_img = tensor_to_pil(img_tensor)
        pixel_values = transform(pil_img).unsqueeze(0).to(torch.bfloat16).to(target_device)
        
        with torch.no_grad():
            generation_config = dict(max_new_tokens=64, do_sample=False)
            generated_report = model.chat(tokenizer, pixel_values, vqa_prompt, generation_config)
            
            report_lines = generated_report.lower().split('\n')
            
            chosen_category = ""
            for line in report_lines:
                if "category:" in line:
                    chosen_category = line.replace("category:", "").strip()
                    break
            
            # ✨ 根据扩充后的词表进行宽泛捕获
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
            "ID_Found": len(id_candidates_names),
            "OOD_Blocked": len(ood_descriptions)
        })
    
    if len(ood_descriptions) > 0:
        corpus = [item["report"] for item in ood_descriptions]
        vectorizer = TfidfVectorizer(max_features=128, stop_words='english')
        X = vectorizer.fit_transform(corpus)
        from sklearn.cluster import KMeans
        num_clusters = max(2, min(5, len(corpus) // 3)) 
        
        if len(corpus) >= num_clusters:
            kmeans_ood = KMeans(n_clusters=num_clusters, random_state=3407).fit(X)
            for i, item in enumerate(ood_descriptions):
                item["discovered_cluster"] = int(kmeans_ood.labels_[i])
                
        os.makedirs(args.output_dir, exist_ok=True)
        out_json = os.path.join(args.output_dir, f"discovered_ood_{task_name}.json")
        with open(out_json, "w", encoding='utf-8') as f:
            json.dump(ood_descriptions, f, indent=4, ensure_ascii=False)

    if len(id_features) == 0:
        return np.array([]), np.array([]), len(ood_descriptions)

    return np.array(id_candidates_names), np.vstack(id_features), len(ood_descriptions)

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

# ✨ 文本追加记录器
def append_to_log(file_path, text):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--init_num", default=50, type=int)
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--train_csv", default="al_file_skin/train.csv", type=str)
    parser.add_argument("--output_dir", default="al_file_skin", type=str)
    parser.add_argument("--model_path", default="pretrained/InternVL2_5-8B", type=str, help="InternVL2.5-8B 模型权重路径")
    
    args = parser.parse_args()
    seed_torch(3407)

    log_file = "single_test.txt"
    run_header = f"\n\n{'='*70}\n🕒 新一轮测试启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (模型: InternVL2.5-8B)\n{'='*70}"
    append_to_log(log_file, run_header)

    print(f"📂 正在解析 {args.train_csv} 并加载皮肤数据池...")
    if not os.path.exists(args.train_csv):
        print(f"❌ 错误：找不到 {args.train_csv}")
        return

    train_df = pd.read_csv(args.train_csv)
    truth_dict = {os.path.basename(row.iloc[0]): int(row.iloc[1]) for _, row in train_df.iterrows()}
    
    if len(train_df) > 300:
        print(f"⚠️ 正在使用 Seed: 2026 随机采样 300 张作为冷启动精选池进行快速测试...")
        train_df = train_df.sample(n=300, random_state=2026).reset_index(drop=True)
    train_files = train_df.to_dict('records')
    
    train_dataset = Tumor_dataset_val(args, train_files)
    train_loader = get_loader(args, train_dataset, shuffle=False, drop=False, batch_size=1)
    
    # 加载 8B 模型
    model, tokenizer = load_internvl(model_path=args.model_path)

    # 严格按 0 到 9 的真实标签 ID 排序
    TASKS_CONFIG = {
        "Eczema": {
            "gold_id_classes": [0], 
            "gold_id_tags": ['eczema', 'dermatitis']
        },
        "Melanoma": {
            "gold_id_classes": [1], 
            "gold_id_tags": ['melanoma', 'mel', 'malignant melanoma']
        },
        "Atopic_Dermatitis": {
            "gold_id_classes": [2], 
            "gold_id_tags": ['atopic dermatitis', 'ad', 'atopic eczema']
        },
        "Basal_Cell_Carcinoma": {
            "gold_id_classes": [3], 
            "gold_id_tags": ['basal cell carcinoma', 'bcc', 'basal cell']
        },
        "Melanocytic_Nevi": {
            "gold_id_classes": [4], 
            "gold_id_tags": ['melanocytic nevi', 'nevi', 'nevus', 'nv', 'mole']
        },
        "Benign_Keratosis": {
            "gold_id_classes": [5], 
            "gold_id_tags": ['benign keratosis', 'bkl', 'solar lentigo', 'lichen planus-like keratosis', 'seborrheic keratosis']
        },
        "Psoriasis": {
            "gold_id_classes": [6], 
            "gold_id_tags": ['psoriasis', 'lichen planus', 'ps', 'plaque psoriasis']
        },
        "Seborrheic_Keratoses": {
            "gold_id_classes": [7], 
            "gold_id_tags": ['seborrheic keratoses', 'seborrheic keratosis', 'sk', 'senile wart']
        },
        "Fungal_Infections": {
            "gold_id_classes": [8], 
            "gold_id_tags": ['fungal', 'tinea', 'dermatophytosis', 'mycosis', 'ringworm']
        },
        "Viral_Infections": {
            "gold_id_classes": [9], 
            "gold_id_tags": ['viral', 'warts', 'wart', 'hpv', 'verruca', 'molluscum']
        }
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # 串行跑所有的独立任务
    for task_name, task_info in TASKS_CONFIG.items():
        current_classes = task_info['gold_id_classes']
        current_tags = task_info['gold_id_tags']

        names_id_vlm, embeds_id, ood_count = generative_ood_discovery(args, train_loader, model, tokenizer, task_name, current_tags)
        
        if len(names_id_vlm) == 0:
            err_msg = f"""
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
▶️ 开始独立任务: {task_name}
▶️ 目标真实标签 ID: {current_classes} | 拦截词: {current_tags}
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀

🌟 [{task_name}] 扫描完成！找到 0 个目标类 (ID)，拦截 {ood_count} 个 OOD 杂病样本。
❌ 警告：任务 [{task_name}] 大模型未发现任何匹配的样本，跳过该任务。
"""
            print(err_msg.strip())
            append_to_log(log_file, err_msg.strip() + "\n")
            continue

        # 评测初筛纯度
        vlm_raw_id_count = sum([1 for n in names_id_vlm if truth_dict.get(os.path.basename(n)) in current_classes])
        vlm_raw_ood_count = len(names_id_vlm) - vlm_raw_id_count
        raw_qp = (vlm_raw_id_count / len(names_id_vlm)) * 100 if len(names_id_vlm) > 0 else 0
        
        # 处理样本过少的警告
        warning_msg = ""
        if len(names_id_vlm) < args.init_num:
            warning_msg = f"⚠️ 候选样本过少 ({len(names_id_vlm)})，将全数选中。\n"
            selected_b = names_id_vlm.tolist()
            selected_c = names_id_vlm.tolist()
        else:
            selected_b = run_density_core_sampling(names_id_vlm, embeds_id, args.init_num)
            selected_c = run_graph_centrality_nms(names_id_vlm, embeds_id, args.init_num)
        
        # 落盘保存文件
        csv_b = os.path.join(args.output_dir, f"query_round_6_density_{task_name}.csv")
        csv_c = os.path.join(args.output_dir, f"query_round_6_graph_{task_name}.csv")
        pd.DataFrame({'img': selected_b, 'label': ['Unknown'] * len(selected_b)}).to_csv(csv_b, index=False)
        pd.DataFrame({'img': selected_c, 'label': ['Unknown'] * len(selected_c)}).to_csv(csv_c, index=False)

        # 计算最终战报
        def calc_qp_report(selected_list):
            id_count = sum([1 for n in selected_list if truth_dict.get(os.path.basename(n)) in current_classes])
            ood_count = len(selected_list) - id_count
            qp = (id_count / len(selected_list)) * 100 if len(selected_list) > 0 else 0
            return id_count, ood_count, qp

        b_id, b_ood, b_qp = calc_qp_report(selected_b)
        c_id, c_ood, c_qp = calc_qp_report(selected_c)

        final_report = f"""
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
▶️ 开始独立任务: {task_name}
▶️ 目标真实标签 ID: {current_classes} | 拦截词: {current_tags}
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀

🌟 [{task_name}] 扫描完成！找到 {len(names_id_vlm)} 个目标类 (ID)，拦截 {ood_count} 个 OOD 杂病样本。

📊 --- [{task_name}] 初筛纯度评测 ---
 ├─ 大模型初筛总数: {len(names_id_vlm)}
 ├─ 命中目标类 (ID {current_classes}): {vlm_raw_id_count}
 └─ 💡 原始初筛精准度 (Raw QP): {raw_qp:.2f}%
{warning_msg}
🏆 --- [{task_name}] 终极双轨算法战报 ---
🔬 【方案 B: Density-Core FPS】 命准: {b_id} | 误选杂病: {b_ood} | 终极QP: {b_qp:.2f}%
🔬 【方案 C: Graph Centrality NMS】 命准: {c_id} | 误选杂病: {c_ood} | 终极QP: {c_qp:.2f}%
============================================================
"""
        print(final_report.strip())
        append_to_log(log_file, final_report.strip() + "\n\n")

    print(f"\n✅ 所有任务配置已全部串行执行完毕！战报已保存至: {log_file}")

if __name__ == '__main__':
    main()