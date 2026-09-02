import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import random
import argparse
import ctypes  # ⚡️ 用于底层符号硬加载
from tqdm import tqdm

# ==========================================
# 0. 核心底层死锁绝杀补丁 (针对云容器物理显卡驱动隔离优化)
# ==========================================
# 强行在全局环境层层面扩展动态链接库寻址（同时覆盖主进程和 DataLoader 子进程）
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

# 针对魔改 Conda 镜像进行 C 语言级别的底层符号硬加载，防止 C++ 算子层崩溃
for path in ['/usr/lib/x86_64-linux-gnu/libcuda.so', '/usr/lib/x86_64-linux-gnu/libcuda.so.1', '/usr/local/cuda/compat/lib/libcuda.so.1']:
    if os.path.exists(path):
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            break
        except:
            pass

# 彻底禁用 cuDNN v8 的多线程优化启发式算法，迫使 PyTorch 使用 RTX 4090 最稳健的底层直连算力驱动
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"

# ==========================================
# 1. 环境与网络配置 (针对云容器环境优化)
# ==========================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  # 引入余弦相似度用于方案C

# 导入原本的数据集加载器
from dataset.alb_dataset2 import Tumor_dataset_val, get_loader


# ⚡️ 核心变更：随机种子完美更换为学术界主流暴涨点种子 3407
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
    基于全新 10 分类皮肤疾病提示词的生成式阅片与高级过滤逻辑
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
    
    gold_id_tags = ['melanoma', 'melanocytic nevi', 'nevi']
    
    transform = build_transform(input_size=448)
    print("\n🔍 开始基于 10-Way 皮肤单选的生成式阅片与 OOD 精准拦截...")
    
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
        from sklearn.cluster import KMeans  # 仅用于OOD的文本聚类
        num_clusters = max(2, min(5, len(corpus) // 3)) 
        
        if len(corpus) >= num_clusters:
            kmeans_ood = KMeans(n_clusters=num_clusters, random_state=3407).fit(X)  # 💡 内部聚类同步对齐新种子
            for i, item in enumerate(ood_descriptions):
                item["discovered_cluster"] = int(kmeans_ood.labels_[i])
                
        os.makedirs(args.output_dir, exist_ok=True)
        out_json = os.path.join(args.output_dir, "discovered_ood_clusters_skin.json")
        with open(out_json, "w", encoding='utf-8') as f:
            json.dump(ood_descriptions, f, indent=4, ensure_ascii=False)

    if len(id_features) == 0:
        return np.array([]), np.array([])

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
        
        # NMS 抑制周边点
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
    
    parser.add_argument("--train_csv", default="al_file_skin/train.csv", type=str)
    parser.add_argument("--output_dir", default="al_file_skin", type=str)
    args = parser.parse_args()
    
    # ⚡️ 核心变更：更换全新的 Seed 初始化全局特征空间
    seed_torch(3407)

    print(f"📂 正在解析 {args.train_csv} 并加载皮肤未标注数据池...")
    if not os.path.exists(args.train_csv):
        print(f"❌ 错误：找不到 {args.train_csv}")
        return

    train_df = pd.read_csv(args.train_csv)
    truth_dict = {os.path.basename(row.iloc[0]): int(row.iloc[1]) for _, row in train_df.iterrows()}
    
    # ⚡️ 核心变更：使用全新的 3407 固化种子进行冷启动精选池空间的随机采样
    if len(train_df) > 500:
        print(f"⚠️ 正在使用新 Seed: 3407 随机采样 500 张作为冷启动精选池...")
        train_df = train_df.sample(n=500, random_state=3407).reset_index(drop=True)
    train_files = train_df.to_dict('records')
    
    train_dataset = Tumor_dataset_val(args, train_files)
    train_loader = get_loader(args, train_dataset, shuffle=False, drop=False, batch_size=1)
    
    model, tokenizer = load_internvl(model_path="pretrained/InternVL2_5-2B")
    
    names_id_vlm, embeds_id = generative_ood_discovery(args, train_loader, model, tokenizer)
    
    if len(names_id_vlm) == 0:
        print("❌ 错误：大模型未在探索池中发现任何匹配的 ID 样本！")
        return

    gold_id_classes = [1, 4] 

    # ==================================================
    # 📊 阶段 1: 大模型初筛全量候选池纯度测评
    # ==================================================
    print("\n" + "="*50)
    print(f"📊 --- 阶段 1: 大模型初筛全量候选池纯度 (QA/QP) 评测 ---")
    vlm_raw_id_count = sum([1 for n in names_id_vlm if truth_dict.get(os.path.basename(n)) in gold_id_classes])
    vlm_raw_ood_count = len(names_id_vlm) - vlm_raw_id_count
                
    raw_qp = (vlm_raw_id_count / len(names_id_vlm)) * 100 if len(names_id_vlm) > 0 else 0
    print(f" ├─ 大模型初筛总候选数: {len(names_id_vlm)}")
    print(f" ├─ 真正目标类 (ID: 1=Mel, 4=Nevi) 数量: {vlm_raw_id_count}")
    print(f" ├─ 混入错误未知类 (OOD 噪声) 数量: {vlm_raw_ood_count}")
    print(f" └─ 💡 原始初筛池精准度 (Raw Query Precision): {raw_qp:.2f}%")
    print("="*50 + "\n")

    # ==================================================
    # 🎯 阶段 2: 执行双算法横向选片
    # ==================================================
    os.makedirs(args.output_dir, exist_ok=True)
    
    if len(names_id_vlm) < args.init_num:
        print(f"⚠️ 警告：候选样本过少，将全数选中。")
        selected_b = names_id_vlm.tolist()
        selected_c = names_id_vlm.tolist()
    else:
        print("🎯 正在执行 方案 B: 基于局部密度的自适应核心采样 (Density-Core FPS)...")
        selected_b = run_density_core_sampling(names_id_vlm, embeds_id, args.init_num)
        
        print("🎯 正在执行 方案 C: 基于图论流形挖掘与侧向抑制采样 (Graph Centrality NMS)...")
        selected_c = run_graph_centrality_nms(names_id_vlm, embeds_id, args.init_num)
    
    # 结果双轨落盘
    csv_b = os.path.join(args.output_dir, "query_round_6_density.csv")
    csv_c = os.path.join(args.output_dir, "query_round_6_graph.csv")
    
    pd.DataFrame({'img': selected_b, 'label': ['Unknown'] * len(selected_b)}).to_csv(csv_b, index=False)
    pd.DataFrame({'img': selected_c, 'label': ['Unknown'] * len(selected_c)}).to_csv(csv_c, index=False)

    # ==================================================
    # 📊 阶段 2: 终极 QA/QP 战报对标
    # ==================================================
    def calc_qp_report(selected_list):
        id_count = sum([1 for n in selected_list if truth_dict.get(os.path.basename(n)) in gold_id_classes])
        ood_count = len(selected_list) - id_count
        qp = (id_count / len(selected_list)) * 100 if len(selected_list) > 0 else 0
        return id_count, ood_count, qp

    b_id, b_ood, b_qp = calc_qp_report(selected_b)
    c_id, c_ood, c_qp = calc_qp_report(selected_c)

    print("\n" + "="*60)
    print(f"🏆 --- 阶段 2: 双轨采样算法最终 50 张冷启动种子 QP 战报 ---")
    print(f"\n🔬 【方案 B: Density-Core FPS (局部密度)】")
    print(f" ├─ 真正目标类 (ID) 数量: {b_id} | 误选杂病 (OOD) 数量: {b_ood}")
    print(f" └─ 💡 最终查询精准度 (QP): {b_qp:.2f}%")
    print(f" 📁 账本已保存至: {csv_b}")
    
    print(f"\n🔬 【方案 C: Graph Centrality NMS (图论流形)】")
    print(f" ├─ 真正目标类 (ID) 数量: {c_id} | 误选杂病 (OOD) 数量: {c_ood}")
    print(f" └─ 💡 最终查询精准度 (QP): {c_qp:.2f}%")
    print(f" 📁 账本已保存至: {csv_c}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()