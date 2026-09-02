import sys
import os
import time
import argparse
import torch
import numpy as np
import pandas as pd
import random
from open_clip import create_model_from_pretrained, get_tokenizer
from transformers import CLIPProcessor, CLIPModel
from sklearn import metrics
from sklearn.cluster import KMeans
import copy

from dataset.alb_dataset2 import Tumor_dataset_val_cls, get_loader

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_arguments():
    parser = argparse.ArgumentParser(description="Per-Class Active Learning Initialization (CRC100K)")
    parser.add_argument("--num_class", type=int, default=9, help="Train class num")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--crop_size", default=224)
    parser.add_argument("--gpu", nargs="+", type=int, default=[0])
    parser.add_argument("--batch_size", type=int, default=512, help="Train batch size")
    parser.add_argument("--num_workers", default=6)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model_type", type=str, default='combine')
    
    # 核心修改：默认设为 0-8 全类别，OOD 设为空，单类抽取数设为 50
    parser.add_argument("--id_cls", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8], help="ID classes to sample from")
    parser.add_argument("--ood_cls", nargs="+", type=int, default=[], help="OOD classes")
    parser.add_argument("--save_csv", type=str, default="al_file/query_round_5_per_class_50.csv")
    parser.add_argument("--init_num", type=int, default=50, help="Number of samples to select per ID class")
    
    return parser.parse_args()

def get_files(data_csv):
    data = pd.read_csv(data_csv)
    data_name = data.iloc[:, 0]
    data_label = data.iloc[:, 1]
    data_label = np.array(data_label).astype(np.uint8)
    data_name = data_name.to_list()
    new_file = [{"img": img, "label": label} for img, label in zip(data_name, data_label)]
    new_dict = {k: v for k, v in zip(data_name, data_label)}
    return new_file, new_dict

def kmean_cluster(embeds, n):
    cluster_learner = KMeans(n_clusters=n, init='k-means++', n_init='auto', random_state=42)
    cluster_learner.fit(embeds)
    cluster_idxs = cluster_learner.predict(embeds)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embeds - centers)**2
    dis = dis.sum(axis=1)
    q_idx = np.array([np.arange(embeds.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
    return q_idx

def load_vlm_models():
    print("🚀 正在加载 BiomedCLIP 与 PLIP 模型 (内存驻留)...")
    BMC_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    BMC_tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    BMC_model.cuda().eval()

    PLIP_model = CLIPModel.from_pretrained("vinid/plip")
    PLIP_processor = CLIPProcessor.from_pretrained("vinid/plip")
    PLIP_model.cuda().eval()
    print("✅ 双 VLM 加载完毕！")
    return BMC_model, BMC_tokenizer, PLIP_model, PLIP_processor

def zero_shot_inference_single_class(args, train_eval_loader, target_cls, global_id_cls, models):
    BMC_model, BMC_tokenizer, PLIP_model, PLIP_processor = models
    
    # 9类全局文本提示词映射
    all_text_prompts = [
        "An H&E image of Adipose",               # 0: ADI
        "An H&E image of background",            # 1: BACK
        "An H&E image of debris",                # 2: DEB
        "An H&E image of lymphocytes",           # 3: LYM
        "An H&E image of mucus",                 # 4: MUC
        "An H&E image of smooth muscle",         # 5: MUS
        "An H&E image of normal mucosa",         # 6: NORM
        "An H&E image of cancer-associated stroma", # 7: STR
        "An H&E image of adenocarcinoma epithelium" # 8: TUM
    ]
    
    # 构建当前目标类 vs 其他所有 ID 类的判别提示词列表 (index 0 为目标类)
    target_prompt = all_text_prompts[target_cls]
    other_id_classes = [c for c in global_id_cls if c != target_cls]
    distractor_prompts = [all_text_prompts[c] for c in other_id_classes]
    
    # 确保至少有一个混淆项
    text_prompt = [target_prompt] + distractor_prompts

    with torch.no_grad():
        pred_all, prob_all = torch.zeros((1, )), torch.zeros((1, len(text_prompt)))
        embed_dim = 1024 if 'combine' in args.model_type else 512
        embeddings = torch.zeros((1, embed_dim))
        names = []
        
        for counter, sample in enumerate(train_eval_loader):
            x_batch = sample['img'].cuda()
            batch_names = sample['img_name']

            if args.model_type == 'BMC':
                texts = BMC_tokenizer(text_prompt).cuda()
                image_features, text_features, logit_scale = BMC_model(x_batch, texts)
                probs = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
                embeddings = torch.cat([embeddings, image_features.cpu()], dim=0)
                
            elif args.model_type == 'plip':
                inputs = PLIP_processor(text=text_prompt, return_tensors="pt", padding=True)
                inputs['pixel_values'] = x_batch
                for key in inputs.keys(): inputs[key] = inputs[key].cuda()
                outputs = PLIP_model.forward(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                embeddings = torch.cat([embeddings, outputs.image_embeds.cpu()], dim=0)
                
            elif args.model_type == 'combine':
                inputs = PLIP_processor(text=text_prompt, return_tensors="pt", padding=True)
                inputs['pixel_values'] = x_batch
                for key in inputs.keys(): inputs[key] = inputs[key].cuda()
                outputs = PLIP_model.forward(**inputs)
                cur_probs1 = outputs.logits_per_image.softmax(dim=1)  
                
                texts = BMC_tokenizer(text_prompt).cuda()
                image_features, text_features, logit_scale = BMC_model(x_batch, texts)
                cur_probs2 = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
                
                probs = (cur_probs1 + cur_probs2) / 2
                combined_embeds = torch.cat((outputs.image_embeds.cpu(), image_features.cpu()), dim=1)
                embeddings = torch.cat((embeddings, combined_embeds), dim=0)

            logits_hard = torch.argmax(probs, dim=1)
            pred_all = torch.cat((pred_all, logits_hard.cpu()), dim=0)
            prob_all = torch.cat((prob_all, probs.cpu()), dim=0)
            names += batch_names
            
    pred_all, prob_all, embeddings = pred_all[1:], prob_all[1:], embeddings[1:]
    y_pred = pred_all.numpy().astype(np.uint8)
    
    return y_pred, prob_all, np.array(names), embeddings.clone().detach().cpu().numpy()

if __name__ == "__main__":
    start_time = time.time()
    args = get_arguments()
    seed_torch(args.seed)
    torch.cuda.set_device(args.gpu[0])

    data_path = '/root/gpufree-data/OpenPath-main/al_file/train.csv'
    if not os.path.exists(data_path):
        print(f"❌ 错误：找不到文件 {data_path}")
        sys.exit(1)
        
    train_files, train_dict = get_files(data_path)
    
    id_cls = args.id_cls
    ood_cls = args.ood_cls

    train_id_files = [item for item in train_files if item['label'] in id_cls]
    train_ood_files = [item for item in train_files if item['label'] in ood_cls]
    print(f"ID 类别数量: {len(train_id_files)}, OOD 类别数量: {len(train_ood_files)}")
    
    train_dataset_eval = Tumor_dataset_val_cls(args, files=train_files)
    train_eval_loader = get_loader(args, train_dataset_eval, shuffle=False)

    vlm_models = load_vlm_models()

    all_selected_records = []
    report_lines = []
    
    out_dir = "al_file"
    os.makedirs(out_dir, exist_ok=True)
    
    class_name_map = {
        0: "ADI", 1: "BACK", 2: "DEB", 3: "LYM", 
        4: "MUC", 5: "MUS", 6: "NORM", 7: "STR", 8: "TUM"
    }

    # 对每个指定的 ID 类别进行逐类高精度抽取
    for target_class in id_cls:
        c_name = class_name_map.get(target_class, str(target_class))
        print(f"\n" + "="*50)
        print(f"🔬 正在对 ID 类别 [Class {target_class}: {c_name}] 进行独立抽取...")
        print("="*50)
        
        true_class_imgs = [item['img'] for item in train_files if item['label'] == target_class]
        if len(true_class_imgs) == 0:
            print(f"⚠️ 警告: 数据集中未发现类别 {target_class} 的样本，跳过。")
            continue

        # 执行 1-vs-All 零样本推理
        y_pred_raw, _, names, embeds = zero_shot_inference_single_class(args, train_eval_loader, target_class, id_cls, vlm_models)
        
        # 筛选预测为该目标类（索引为0）的样本
        mask = (y_pred_raw == 0)
        names_id_vlm = names[mask]
        embeds_id = embeds[mask]
        
        vlm_raw_count = len(set(names_id_vlm) & set(true_class_imgs))
        raw_qp = (vlm_raw_count / len(names_id_vlm) * 100) if len(names_id_vlm) > 0 else 0.0
        print(f"  [阶段1] 初筛候选数: {len(names_id_vlm)}, 命中真命题数: {vlm_raw_count}, 初筛准确率: {raw_qp:.2f}%")

        if len(names_id_vlm) == 0:
            print(f"  ❌ 错误: 模型未能在候选集中找到任何类别 {target_class} ({c_name}) 的样本。")
            final_qp = 0.0
            actual_init_num = 0
            names_init_select = []
        else:
            actual_init_num = min(args.init_num, len(names_id_vlm))
            if actual_init_num < args.init_num:
                print(f"  ⚠️ 警告: 初筛数量 ({len(names_id_vlm)}) 小于目标抽取数 ({args.init_num})，将全量提取。")
                cluster_idx = np.arange(len(names_id_vlm))
            else:
                cluster_idx = kmean_cluster(embeds=embeds_id, n=actual_init_num)
                
            names_init_select = names_id_vlm[cluster_idx]
            
            # 计算最终 Query Accuracy (QA)
            true_selected_count = sum(1 for name in names_init_select if name in true_class_imgs)
            final_qp = (true_selected_count / actual_init_num) * 100 if actual_init_num > 0 else 0.0
            
            print(f"  [阶段2] KMeans++ 提取数: {actual_init_num}, 命中真实数: {true_selected_count}, 最终 Query Accuracy (QA): {final_qp:.2f}%")
            
            for img_path in names_init_select:
                all_selected_records.append({
                    'img': img_path,
                    'cls_label': train_dict[img_path]
                })

        report_lines.append(
            f"Class {target_class} ({c_name:<4}) | "
            f"Pool: {len(names_id_vlm):<4} (Raw QA: {raw_qp:>6.2f}%) | "
            f"Final QA ({actual_init_num:d}): {final_qp:>6.2f}%"
        )

    # 汇总输出
    end_time = time.time()
    total_seconds = end_time - start_time
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    report_text = f"""
======================================================
🕒 运行时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
🎲 随机种子 (Seed): {args.seed}
📈 每类目标抽取数 (init_num): {args.init_num}
------------------------------------------------------
📊 各 ID 类别独立抽取战报 (Per-Class QA Report):
"""
    for line in report_lines:
        report_text += f" {line}\n"

    report_text += f"""------------------------------------------------------
⏱️ 总运行耗时: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s
======================================================
"""

    print(report_text)
    
    log_file_path = os.path.join(out_dir, "crc100k_per_class_qa_experiment.log")
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(report_text)
    print(f"📝 汇总实验记录已追加至: {log_file_path}")
    
    if all_selected_records:
        df_selected = pd.DataFrame(all_selected_records)
        # 打乱顺序保存组合后的冷启动文件
        df_selected = df_selected.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        csv_path = args.save_csv
        df_selected.to_csv(csv_path, index=False)
        print(f"📁 最终提取的多类组合冷启动样本已保存至: {csv_path} (共 {len(df_selected)} 张图片)")