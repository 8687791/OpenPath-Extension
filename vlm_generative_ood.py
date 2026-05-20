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
    """
    将 DataLoader 输出的标准化 Tensor 还原为 PIL 图像，供大模型使用。
    """
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
    """InternVL 官方推荐的图像预处理 Transform"""
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
    """加载 InternVL2.5 模型与 Tokenizer"""
    print(f"🚀 正在加载轻量级 InternVL2.5-2B 基础大模型 (显存占用约 5~6GB)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    # 采用 bfloat16 精度加载以节省显存
    model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).eval().cuda()
    
    print("✅ 模型加载成功！")
    return model, tokenizer


def generative_ood_discovery(args, dataloader, id_cls_names, model, tokenizer):
    """
    基于 MLLM 的生成式阅片与 OOD 动态清洗逻辑
    """
    id_candidates_names = []
    id_features = []
    ood_descriptions = []
    
    # 💡 优化后的多模态提示词：强迫大模型先给出明确态度，再做细节描述，方便下游精准拦截
    vqa_prompt = (
        "Analyze this H&E pathology image. Does it contain any of the following target tissues: "
        "tumor/adenocarcinoma epithelium, lymphocytes, or normal colon mucosa? "
        "Answer with 'YES' or 'NO' first. Then provide a brief description of the pathological features."
    )
    
    transform = build_transform(input_size=448) # InternVL 默认输入尺寸
    
    print("\n🔍 开始基于 MLLM 的生成式阅片与 OOD 发现...")
    
    for idx, sample in enumerate(tqdm(dataloader, desc="阅片进度")):
        img_tensor = sample['img']
        img_name = sample['img_name'][0]
        
        # 1. 将原 DataLoader 的 Tensor 还原为纯净的 PIL 图像
        pil_img = tensor_to_pil(img_tensor)
        
        # 2. 转换为 InternVL 需要的 pixel_values
        pixel_values = transform(pil_img).unsqueeze(0).to(torch.bfloat16).cuda()
        
        with torch.no_grad():
            generation_config = dict(max_new_tokens=48, do_sample=False)
            generated_report = model.chat(tokenizer, pixel_values, vqa_prompt, generation_config)
            
            report_lower = generated_report.lower()
            
            # 💡 双保险拦截算法：检测目标关键词，并严格剔除包含否定词（如 no tumor, without lymphocytes）的样本
            has_id_keyword = any(cls in report_lower for cls in id_cls_names)
            is_negated = "no " in report_lower or "not " in report_lower or "without" in report_lower
            
            # 综合断定：只要模型明确说了 yes，或者包含关键词且没有被否定，才算作黄金 ID 候选
            is_id = report_lower.startswith("yes") or (has_id_keyword and not is_negated)
            
            if is_id:
                # 调用视觉塔 (Vision Model) 提取特征用于聚类
                vision_embeds = model.extract_feature(pixel_values).mean(dim=1).cpu().float().numpy()
                id_candidates_names.append(img_name)
                id_features.append(vision_embeds[0])
            else:
                ood_descriptions.append({
                    "img_name": img_name,
                    "report": generated_report
                })

    # ------------------------------------------
    # 创新点 2：利用生成的文本描述对 OOD 进行动态聚类
    # ------------------------------------------
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
        with open("al_file/discovered_ood_clusters2.json", "w", encoding='utf-8') as f:
            json.dump(ood_descriptions, f, indent=4, ensure_ascii=False)
        print("✅ OOD 聚类结果已保存至 al_file/discovered_ood_clusters2.json")

    return np.array(id_candidates_names), np.vstack(id_features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int, help="生成模型强制 bs=1 防止显存溢出")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--init_num", default=50, type=int, help="第一轮冷启动挑选的 ID 样本数")
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    args = parser.parse_args()
    
    seed_torch()

    # 💡 核心修复：将目标已知类别名称严格修改为仅匹配 OpenPath 中的 3, 6, 8 (LYM, NORM, TUM) 对应病理文本
    # 彻底剔除了之前混入的 'Adipose' 和 'Stroma' 噪声类
    id_cls_names = ['tumor', 'lymphocytes', 'normal', 'adenocarcinoma', 'epithelium']
    
    # 2. 加载数据
    print("📂 正在解析 al_file/train.csv 并加载数据池...")
    if not os.path.exists('al_file/train.csv'):
        print("❌ 错误：找不到 al_file/train.csv，请检查路径")
        return

    train_df = pd.read_csv('al_file/train.csv')
    
    # 核心优化：如果全量数据大于 500 张，冷启动阶段随机抽样 500 张让大模型看
    if len(train_df) > 500:
        print(f"⚠️ 检测到原始数据量巨大 ({len(train_df)}张)。")
        print("🚀 为了大幅度提升主动学习冷启动效率，正在随机采样 500 张作为大模型筛选池...")
        train_df = train_df.sample(n=500, random_state=42).reset_index(drop=True)
    train_files = train_df.to_dict('records')
    
    train_dataset = Tumor_dataset_val(args, train_files)
    train_loader = get_loader(args, train_dataset, shuffle=False, drop=False, batch_size=1)
    
    # 指定加载本地轻量大模型路径
    model, tokenizer = load_internvl(model_path="pretrained/InternVL2_5-2B")
    
    # 4. 执行生成式推理与 OOD 发现
    names_id_vlm, embeds_id = generative_ood_discovery(args, train_loader, id_cls_names, model, tokenizer)
    
    # 5. ID 样本的多样性挑选 (K-Means++)
    if len(names_id_vlm) == 0:
        print("❌ 错误：未发现任何匹配的 ID 样本，请检查数据集或分类词。")
        return

    if len(names_id_vlm) < args.init_num:
        print(f"⚠️ 警告：找到的 ID 候选样本({len(names_id_vlm)})少于预算({args.init_num})，将全数选中！")
        selected_names = names_id_vlm.tolist()
    else:
        print("🎯 正在对找到的 ID 样本进行多样性 K-Means++ 采样...")
        kmeans = KMeans(n_clusters=args.init_num, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(embeds_id)
        
        selected_names = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(embeds_id - center, axis=1)
            selected_names.append(names_id_vlm[np.argmin(distances)])
            
    # 6. 保存冷启动采样结果
    os.makedirs("al_file", exist_ok=True)
    df_selected = pd.DataFrame({'img': selected_names, 'label': ['Unknown'] * len(selected_names)})
    df_selected.to_csv('al_file/query_round_1.csv', index=False)
    print(f"✅ 第一轮生成式冷启动挑选完成！结果已保存至 al_file/query_round_2.csv")

if __name__ == '__main__':
    main()