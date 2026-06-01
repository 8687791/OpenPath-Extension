import os
import sys
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


# ==========================================
# 2. 双轨采样算法具体实现桩 (Stubs)
# ==========================================
def run_density_core_sampling(names, embeds, init_num):
    """ 方案 B: 基于局部密度的自适应核心采样 (Density-Core FPS) """
    print("   [执行中] 运行 Density-Core 过滤...")
    return names[:init_num].tolist()


def run_graph_centrality_nms(names, embeds, init_num):
    """ 方案 C: 基于图论流形挖掘与侧向抑制采样 (Graph Centrality NMS) """
    print("   [执行中] 运行 Graph Centrality NMS 过滤...")
    return names[-init_num:].tolist()


# ==========================================
# 3. 核心生成式 OOD 筛选 (含实时检测评估)
# ==========================================
def generative_ood_discovery(args, dataloader, model, tokenizer, truth_dict, gold_id_classes):
    """
    基于标准 9 分类提示词的生成式阅片与高级过滤逻辑 (含 tqdm 实时指标刷新)
    """
    id_candidates_names = []
    id_features = []
    reports_id = []  
    ood_descriptions = []
    
    # 实时统计控制台指标
    running_true_id_in_pool = 0  # 判定为 ID 且真值确实是目标数字分类的数量
    
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
    
    # 大模型文本匹配标签
    gold_id_tags = ['lym', 'norm', 'tum']
    transform = build_transform(input_size=448)
    
    print("\n🔍 开始基于 9-Way 临床单选的生成式阅片与 OOD 精准拦截...")
    
    # 使用 tqdm 包装数据加载器
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
            
            is_id = any(tag in chosen_category for tag in gold_id_tags)
            
            if is_id:
                vision_embeds = model.extract_feature(pixel_values).mean(dim=1).cpu().float().numpy()
                id_candidates_names.append(img_name)
                id_features.append(vision_embeds[0])
                reports_id.append(generated_report)
                
                # 💡 实时真值交叉检验：使用修正后的目标数字类别进行命中统计
                if truth_dict.get(base_name) in gold_id_classes:
                    running_true_id_in_pool += 1
            else:
                ood_descriptions.append({
                    "img_name": img_name,
                    "report": generated_report
                })
        
        # 💡 动态计算实时精准度并在 tqdm 进度条右侧显示
        current_pool_size = len(id_candidates_names)
        current_qp = (running_true_id_in_pool / current_pool_size * 100) if current_pool_size > 0 else 0.0
        
        pbar.set_postfix({
            "ID候选(Pool)": current_pool_size,
            "OOD拦截": len(ood_descriptions),
            "实时精度(Cur_QP)": f"{current_qp:.2f}%"
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

    return np.array(id_candidates_names), np.vstack(id_features), reports_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--init_num", default=50, type=int)
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--output_dir", default="al_file", type=str) 
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
    
    # ==================================================
    # 💡 提前建立真值字典与目标类别定义，供实时和最终评测使用
    # ==================================================
    if 'label' in train_df.columns and 'img' in train_df.columns:
        truth_dict = dict(zip(train_df['img'].apply(os.path.basename), train_df['label']))
    else:
        truth_dict = {}
        
    # 🎯 核心修改：定义真正目标数字类别 [3, 6, 8]（兼容整型与字符串型存储）
    gold_id_classes = [3, 6, 8, '3', '6', '8'] 

    train_dataset = Tumor_dataset_val(args, train_files)
    train_loader = get_loader(args, train_dataset, shuffle=False, drop=False, batch_size=1)
    
    model, tokenizer = load_internvl(model_path="pretrained/InternVL2_5-2B")
    
    # 💡 将真值映射传入 VLM 发现函数，启动实时看板监测
    names_id_vlm, embeds_id, reports_id = generative_ood_discovery(
        args, train_loader, model, tokenizer, truth_dict, gold_id_classes
    )
    
    if len(names_id_vlm) == 0:
        print("❌ 错误：未发现任何匹配 of ID 样本！")
        return

    # ==================================================
    # 💾 物理保存大池属性，支持后续独立消融实验
    # ==================================================
    os.makedirs(args.output_dir, exist_ok=True)
    save_cache_path = os.path.join(args.output_dir, "vlm_id_candidates.pkl")
    with open(save_cache_path, "wb") as f:
        pickle.dump({"names": names_id_vlm, "embeds": embeds_id, "reports": reports_id}, f)
    print(f"💾 [科研固化成功]：全量初筛池属性完整存盘: {save_cache_path}")

    # ==================================================
    # 📊 --- 阶段 1: 大模型初筛全量候选池纯度汇总评测 ---
    # ==================================================
    print("\n" + "="*50)
    print(f"📊 --- 阶段 1: 大模型初筛全量候选池纯度 (QA/QP) 评测 ---")
    vlm_raw_id_count = sum([1 for n in names_id_vlm if truth_dict.get(os.path.basename(n)) in gold_id_classes])
    vlm_raw_ood_count = len(names_id_vlm) - vlm_raw_id_count
                
    raw_qp = (vlm_raw_id_count / len(names_id_vlm)) * 100 if len(names_id_vlm) > 0 else 0
    print(f" ├─ 大模型初筛总候选数: {len(names_id_vlm)}")
    print(f" ├─ 真正目标类 (ID: 3, 6, 8) 数量: {vlm_raw_id_count}")
    print(f" ├─ 混入错误未知类 (OOD 噪声) 数量: {vlm_raw_ood_count}")
    print(f" └─ 💡 原始初筛池精准度 (Raw Query Precision): {raw_qp:.2f}%")
    print("="*50 + "\n")

    # ==================================================
    # 🎯 阶段 2 ———— 执行双算法横向选片与 B/C 消融同台对比
    # ==================================================
    if len(names_id_vlm) < args.init_num:
        print(f"⚠️ 警告：精选候选池样本少于预算({args.init_num})，将全数选中！")
        selected_b = names_id_vlm.tolist()
        selected_c = names_id_vlm.tolist()
    else:
        print("🎯 正在执行 方案 B: 基于局部密度的自适应核心采样 (Density-Core FPS)...")
        selected_b = run_density_core_sampling(names_id_vlm, embeds_id, args.init_num)
        
        print("🎯 正在执行 方案 C: 基于图论流形挖掘与侧向抑制采样 (Graph Centrality NMS)...")
        selected_c = run_graph_centrality_nms(names_id_vlm, embeds_id, args.init_num)
            
    # 结果双轨落盘
    csv_b = os.path.join(args.output_dir, "query_round_5_density.csv")
    csv_c = os.path.join(args.output_dir, "query_round_5_graph.csv")
    pd.DataFrame({'img': selected_b, 'label': ['Unknown'] * len(selected_b)}).to_csv(csv_b, index=False)
    pd.DataFrame({'img': selected_c, 'label': ['Unknown'] * len(selected_c)}).to_csv(csv_c, index=False)

    # 计算精选战报
    def calc_qp_report(selected_list):
        id_count = sum([1 for n in selected_list if truth_dict.get(os.path.basename(n)) in gold_id_classes])
        ood_count = len(selected_list) - id_count
        qp = (id_count / len(selected_list)) * 100 if len(selected_list) > 0 else 0
        return id_count, ood_count, qp

    b_id, b_ood, b_qp = calc_qp_report(selected_b)
    c_id, c_ood, c_qp = calc_qp_report(selected_c)

    print("\n" + "="*60)
    print(f"🏆 --- 阶段 2: 双轨采样算法最终 50 张冷启动种子 QP 战报 ---")
    print(f"\n🔬 【方案 B: Density-Core FPS (局部密度过滤)】")
    print(f" ├─ 真正目标类 (ID) 数量: {b_id} | 误选杂病 (OOD) 数量: {b_ood}")
    print(f" └─ 💡 最终查询精准度 (QP): {b_qp:.2f}%")
    print(f" 📁 结果已保存至: {csv_b}")
    
    print(f"\n🔬 【方案 C: Graph Centrality NMS (图论流形挖掘)】")
    print(f" ├─ 真正目标类 (ID) 数量: {c_id} | 误选杂病 (OOD) 数量: {c_ood}")
    print(f" └─ 💡 最终查询精准度 (QP): {c_qp:.2f}%")
    print(f" 📁 结果已保存至: {csv_c}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()