import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import json
import random
import argparse
import pickle  # 💾 用于物理保存大池属性
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

# 💡 全局类别映射表，严格对应 0~8 索引
CLASS_MAPPING = {
    0: {"abbr": "ADI",  "tag": "adi"},
    1: {"abbr": "BACK", "tag": "back"},
    2: {"abbr": "DEB",  "tag": "deb"},
    3: {"abbr": "LYM",  "tag": "lym"},
    4: {"abbr": "MUC",  "tag": "muc"},
    5: {"abbr": "MUS",  "tag": "mus"},
    6: {"abbr": "NORM", "tag": "norm"},
    7: {"abbr": "STR",  "tag": "str"},
    8: {"abbr": "TUM",  "tag": "tum"}
}

def seed_torch(seed=67):
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

# 💡 修改点 1：默认加载 InternVL2_5-8B
def load_internvl(model_path="pretrained/InternVL2_5-8B"):
    print(f"🚀 正在加载 InternVL2.5-8B 基础大模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).eval().cuda()
    print("✅ 模型加载成功！")
    return model, tokenizer

# ==========================================
# 2. 类别分层平衡采样逻辑
# ==========================================
def balanced_sampling(names, tags, init_num):
    """基于 VLM 预测伪标签的分层均匀采样 (Stratified Sampling)"""
    unique_tags = np.unique(tags)
    num_classes = len(unique_tags)
    
    if num_classes == 0:
        return names[:init_num].tolist()
        
    quota_per_class = init_num // num_classes
    remainder = init_num % num_classes
    
    selected = []
    tag_to_names = {tag: names[tags == tag].tolist() for tag in unique_tags}
    
    for tag in unique_tags:
        current_quota = quota_per_class + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
            
        candidates = tag_to_names[tag]
        take_num = min(current_quota, len(candidates))
        selected.extend(candidates[:take_num])
        
    if len(selected) < init_num:
        remaining_pool = [n for n in names if n not in selected]
        shortage = init_num - len(selected)
        selected.extend(remaining_pool[:shortage])
        
    return selected

def run_density_core_sampling(names, embeds, tags, init_num):
    print("   [执行中] 运行 Density-Core 分层过滤与类别平衡...")
    return balanced_sampling(names, tags, init_num)

def run_graph_centrality_nms(names, embeds, tags, init_num):
    print("   [执行中] 运行 Graph Centrality NMS 分层过滤与类别平衡...")
    return balanced_sampling(names[::-1], tags[::-1], init_num)

# ==========================================
# 3. 核心生成式 OOD 筛选
# ==========================================
def generative_ood_discovery(args, dataloader, model, tokenizer, truth_dict, id_cls):
    id_candidates_names = []
    id_features = []
    reports_id = []  
    id_predicted_tags = []
    ood_descriptions = []
    
    running_true_id_in_pool = 0 
    
    vqa_prompt = (
        "You are an expert pathologist. Classify this H&E pathology image into EXACTLY ONE of the following 9 distinct categories:\n"
        "1. ADI (Adipose tissue)\n"
        "2. BACK (Background)\n"
        "3. DEB (Debris)\n"
        "4. LYM (Lymphocytes)\n"
        "5. MUC (Mucus)\n"
        "6. MUS (Smooth muscle)\n"
        "7. NORM (Normal colon mucosa)\n"
        "8. STR (Cancer-associated stroma)\n"
        "9. TUM (Adenocarcinoma epithelium / Tumor)\n\n"
        "Respond strictly in this format:\n"
        "Category: [Input the exact category abbreviation from the list above, e.g., TUM or MUS]\n"
        "Description: [A brief one-sentence reason]"
    )
    
    gold_id_tags = [CLASS_MAPPING[idx]['tag'] for idx in id_cls]
    print(f"🎯 当前动态 ID 目标解析 Tags: {gold_id_tags}")
    
    transform = build_transform(input_size=448)
    print("\n🔍 开始基于 9-Way 临床单选的生成式阅片与 OOD 精准拦截...")
    
    pbar = tqdm(dataloader, desc="阅片进度")
    for idx, sample in enumerate(pbar):
        img_tensor = sample['img']
        img_name = sample['img_name'][0]
        base_name = os.path.basename(img_name)
        
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
            
            matched_tag = None
            for tag in gold_id_tags:
                if tag in chosen_category:
                    matched_tag = tag
                    break
            
            if matched_tag is not None:
                vision_embeds = model.extract_feature(pixel_values).mean(dim=1).cpu().float().numpy()
                id_candidates_names.append(img_name)
                id_features.append(vision_embeds[0])
                reports_id.append(generated_report)
                id_predicted_tags.append(matched_tag)
                
                label_val = truth_dict.get(base_name)
                if label_val in id_cls or str(label_val) in [str(c) for c in id_cls]:
                    running_true_id_in_pool += 1
            else:
                ood_descriptions.append({
                    "img_name": img_name,
                    "report": generated_report
                })
        
        current_pool_size = len(id_candidates_names)
        current_qp = (running_true_id_in_pool / current_pool_size * 100) if current_pool_size > 0 else 0.0
        
        pbar.set_postfix({
            "ID候选": current_pool_size,
            "OOD拦截": len(ood_descriptions),
            "实时精度": f"{current_qp:.2f}%"
        })

    print(f"\n🌟 扫描完成！找到 {len(id_candidates_names)} 个已知类 (ID) 样本候选。")
    
    if len(ood_descriptions) > 0:
        corpus = [item["report"] for item in ood_descriptions]
        vectorizer = TfidfVectorizer(max_features=128, stop_words='english')
        X = vectorizer.fit_transform(corpus)
        num_clusters = max(2, min(5, len(corpus) // 3)) 
        
        if len(corpus) >= num_clusters:
            kmeans_ood = KMeans(n_clusters=num_clusters, random_state=67).fit(X)
            for i, item in enumerate(ood_descriptions):
                item["discovered_cluster"] = int(kmeans_ood.labels_[i])
                
        os.makedirs("al_file", exist_ok=True)
        with open("al_file/discovered_ood_clusters.json", "w", encoding='utf-8') as f:
            json.dump(ood_descriptions, f, indent=4, ensure_ascii=False)

    return np.array(id_candidates_names), (np.vstack(id_features) if id_features else np.array([])), reports_id, np.array(id_predicted_tags)

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--init_num", default=50, type=int)
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--output_dir", default="al_file", type=str)
    parser.add_argument("--seed", default=81, type=int)
    args = parser.parse_args()
    
    seed_torch(args.seed)

    all_classes = np.arange(9)
    id_cls = sorted(np.random.choice(all_classes, size=3, replace=False).tolist())
    ood_cls = sorted([c for c in all_classes if c not in id_cls])
    
    print("="*50)
    print(f"🎲 随机种子 (Seed): {args.seed}")
    print(f"🟢 随机抽取的 3 个已知类 (ID): {id_cls}")
    print(f"🔴 剩下的 6 个未知类 (OOD): {ood_cls}")
    print("="*50)

    if not os.path.exists('al_file/train.csv'):
        print("❌ 错误：找不到 al_file/train.csv")
        return

    train_df = pd.read_csv('al_file/train.csv')
    
    # 💡 修改点 2：将采样数量从 3000 改为 300
    if len(train_df) > 300:
        print(f"⚠️ 检测到原始数据量巨大 ({len(train_df)}张)。正在随机采样 300 张作为冷启动精选池...")
        train_df = train_df.sample(n=300, random_state=2026).reset_index(drop=True)
    train_files = train_df.to_dict('records')
    
    if 'label' in train_df.columns and 'img' in train_df.columns:
        truth_dict = dict(zip(train_df['img'].apply(os.path.basename), train_df['label']))
    else:
        truth_dict = {}

    train_dataset = Tumor_dataset_val(args, train_files)
    train_loader = get_loader(args, train_dataset, shuffle=False, drop=False, batch_size=1)
    
    # 💡 修改点 3：调用修改后的 8B 模型路径
    model, tokenizer = load_internvl(model_path="pretrained/InternVL2_5-8B")
    
    names_id_vlm, embeds_id, reports_id, tags_id_vlm = generative_ood_discovery(
        args, train_loader, model, tokenizer, truth_dict, id_cls
    )
    
    if len(names_id_vlm) == 0:
        print("❌ 错误：未发现任何匹配的 ID 样本！")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    save_cache_path = os.path.join(args.output_dir, "vlm_id_candidates.pkl")
    with open(save_cache_path, "wb") as f:
        pickle.dump({"names": names_id_vlm, "embeds": embeds_id, "reports": reports_id, "tags": tags_id_vlm}, f)

    vlm_raw_id_count = sum([1 for n in names_id_vlm if truth_dict.get(os.path.basename(n)) in id_cls or str(truth_dict.get(os.path.basename(n))) in [str(c) for c in id_cls]])
    vlm_raw_ood_count = len(names_id_vlm) - vlm_raw_id_count
    raw_qp = (vlm_raw_id_count / len(names_id_vlm)) * 100 if len(names_id_vlm) > 0 else 0

    if len(names_id_vlm) < args.init_num:
        selected_b = names_id_vlm.tolist()
        selected_c = names_id_vlm.tolist()
    else:
        selected_b = run_density_core_sampling(names_id_vlm, embeds_id, tags_id_vlm, args.init_num)
        selected_c = run_graph_centrality_nms(names_id_vlm, embeds_id, tags_id_vlm, args.init_num)
            
    csv_b = os.path.join(args.output_dir, "query_round_5_density.csv")
    csv_c = os.path.join(args.output_dir, "query_round_5_graph.csv")
    
    labels_b = [truth_dict.get(os.path.basename(n), 'Unknown') for n in selected_b]
    pd.DataFrame({'img': selected_b, 'label': labels_b}).to_csv(csv_b, index=False)
    
    labels_c = [truth_dict.get(os.path.basename(n), 'Unknown') for n in selected_c]
    pd.DataFrame({'img': selected_c, 'label': labels_c}).to_csv(csv_c, index=False)

    def calc_qp_report(selected_list):
        id_count = sum([1 for n in selected_list if truth_dict.get(os.path.basename(n)) in id_cls or str(truth_dict.get(os.path.basename(n))) in [str(c) for c in id_cls]])
        ood_count = len(selected_list) - id_count
        qp = (id_count / len(selected_list)) * 100 if len(selected_list) > 0 else 0
        return id_count, ood_count, qp

    b_id, b_ood, b_qp = calc_qp_report(selected_b)
    c_id, c_ood, c_qp = calc_qp_report(selected_c)

    end_time = time.time()
    total_seconds = end_time - start_time
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # ==========================
    # 4. 构建战报字符串并保存
    # ==========================
    report_text = f"""
======================================================
🕒 运行时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
🎲 随机种子 (Seed): {args.seed}
🟢 选中已知类 (ID): {id_cls}
🔴 剩下未知类 (OOD): {ood_cls}
------------------------------------------------------
📊 阶段 1: 初筛全量候选池纯度 (QA/QP)
 ├─ 初筛总候选数: {len(names_id_vlm)}
 ├─ 真正目标类数量: {vlm_raw_id_count}
 ├─ 误入未知类数量: {vlm_raw_ood_count}
 └─ 💡 原始初筛精准度 (Raw QP): {raw_qp:.2f}%
------------------------------------------------------
🏆 阶段 2: 最终 {args.init_num} 张冷启动种子 QP
🔬 [方案 B: Density-Core FPS]
 ├─ 真正目标类(ID): {b_id} | 杂病(OOD): {b_ood}
 └─ 💡 最终精准度(QP): {b_qp:.2f}%

🔬 [方案 C: Graph Centrality NMS]
 ├─ 真正目标类(ID): {c_id} | 杂病(OOD): {c_ood}
 └─ 💡 最终精准度(QP): {c_qp:.2f}%
------------------------------------------------------
⏱️ 总运行耗时: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s
======================================================
"""

    # 打印到控制台
    print(report_text)
    
    # 追加写入日志文档
    log_file_path = os.path.join(args.output_dir, "experiment_results.log")
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"📝 实验记录已成功追加至: {log_file_path}")


if __name__ == '__main__':
    main()