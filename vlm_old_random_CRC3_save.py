import sys
import os
import time

# 配置国内镜像源与关闭SSL验证（必须放在最开头）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
from dataset.alb_dataset2 import Tumor_dataset, Tumor_dataset_val, Tumor_dataset_val_cls, get_loader
import argparse
import torch

import numpy as np
import pandas as pd
import random
from open_clip import create_model_from_pretrained, get_tokenizer
from peft import LoraConfig, TaskType, get_peft_model, get_peft_config
from transformers import CLIPProcessor, CLIPModel
from sklearn import metrics
from sklearn.cluster import KMeans
import copy

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
    parser = argparse.ArgumentParser(
        description="xxxx Pytorch implementation")
    parser.add_argument("--num_class", type=int, default=9, help="Train class num")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--crop_size", default=224)
    parser.add_argument("--gpu", nargs="+", type=int, default=[0])
    parser.add_argument("--batch_size", type=int, default=512, help="Train batch size")
    parser.add_argument("--num_workers", default=6)
    parser.add_argument("--seed", default=92, type=int)
    parser.add_argument("--model_type", type=str, default='combine')
    # 💡 修改点：将默认的保存路径修改为指定的 query_round_5_old.csv
    parser.add_argument("--save_csv", type=str, default="/root/gpufree-data/OpenPath-main/al_file/query_round_5_old7.csv")
    parser.add_argument("--init_num", type=int, default=45)
    return parser.parse_args()

def get_files(data_csv):
    data = pd.read_csv(data_csv)
    data_name = data.iloc[:, 0]
    data_label = data.iloc[:, 1]
    data_label = np.array(data_label).astype(np.uint8)
    data_name = data_name.to_list()
    new_file = [{"img": img, "label": label} for img, label in zip(data_name, data_label)]
    new_dict = {k:v for k, v in zip(data_name, data_label)}
    return new_file, new_dict

def cal_acc(y_pred, y_true):
    test_accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = metrics.recall_score(y_true, y_pred, average='macro')
    print(metrics.recall_score(y_true, y_pred, average=None))
    print(f"Test Accuracy: {test_accuracy.item()}, f1:{f1}, precision:{p}, recall:{r}")

def kmean_cluster(embeds, n):
    cluster_learner = KMeans(n_clusters=n, init='k-means++', n_init='auto')
    cluster_learner.fit(embeds)
    cluster_idxs = cluster_learner.predict(embeds)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embeds - centers)**2
    dis = dis.sum(axis=1)
    q_idx = np.array([np.arange(embeds.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
    return q_idx

def zero_shot_inference_random(args, train_eval_loader, id_cls, model_type='BMC'):
    # biomedCLIP
    BMC_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    BMC_tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    BMC_model.cuda().eval()

    # PLIP
    PLIP_model = CLIPModel.from_pretrained("vinid/plip")
    PLIP_processor = CLIPProcessor.from_pretrained("vinid/plip")
    PLIP_model.cuda().eval()

    # 1. 9 个主类的 Prompt 严格对应类别索引 0~8
    base_text_prompt = [
        "An H&E image of Adipose",                  # 0: ADI
        "An H&E image of background",               # 1: BACK
        "An H&E image of debris",                   # 2: DEB
        "An H&E image of lymphocytes",              # 3: LYM
        "An H&E image of mucus",                    # 4: MUC
        "An H&E image of smooth muscle",            # 5: MUS
        "An H&E image of normal mucosa",            # 6: NORM
        "An H&E image of cancer-associated stroma", # 7: STR
        "An H&E image of adenocarcinoma epithelium" # 8: TUM
    ]
    
    # 2. 为每个主类分配病理学上高度相似的专属“混淆项” (Distractors)
    distractors_dict = {
        0: ['An H&E image of Vessels'], 
        1: ['An H&E image of fibrous'], 
        2: ['An H&E image of necrotic tissue'], 
        3: ['An H&E image of Inflammatory infiltrates'], 
        4: ['An H&E image of Submucosa'], 
        5: ['An H&E image of stroma'], 
        6: ['An H&E image of glandular tissue', 'An H&E image of squamous epithelium'], 
        7: ['An H&E image of Nerves'], 
        8: ['An H&E image of Dysplasia', 'An H&E image of Hyperplasia']
    }
    
    # 3. 提取随机选中的 3个已知类 (ID) 的目标 Prompt
    text_prompt = [base_text_prompt[i] for i in id_cls]
    
    # 4. 动态提取这 3个已知类 对应的专属混淆 Prompt，并加入列表
    dynamic_distractors = []
    for cls_idx in id_cls:
        dynamic_distractors.extend(distractors_dict[cls_idx])
    
    # 组合目标Prompt与混淆Prompt (保持目标Prompt在最前面)
    text_prompt += list(dict.fromkeys(dynamic_distractors)) # 顺便去重
    print("\n🎯 最终使用的 Prompt 列表:", text_prompt)

    with torch.no_grad():
        pred = np.zeros((args.num_class,))
        pred_all, prob_all = torch.zeros((1, )), torch.zeros((1, len(text_prompt)))
        if model_type == 'plip' or model_type == 'BMC' or model_type == 'CONCH':
            embeddings = torch.zeros((1, 512))
        if 'combine' in model_type:
            embeddings = torch.zeros((1, 1024))
        names = []
        for counter, sample in enumerate(train_eval_loader):
            x_batch = sample['img'].cuda()
            batch_names = sample['img_name']

            if counter == 0:
                print(batch_names[0])

            if model_type == 'BMC':
                texts = BMC_tokenizer(text_prompt).cuda()
                image_features, text_features, logit_scale = BMC_model(x_batch, texts)
                probs = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
                embeddings = torch.cat([embeddings, image_features.cpu()], dim=0)
            if model_type == 'plip':
                inputs = PLIP_processor(text=text_prompt, return_tensors="pt", padding=True)
                inputs['pixel_values'] = x_batch
                for key in inputs.keys():
                    inputs[key] = inputs[key].cuda()
                outputs = PLIP_model.forward(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                image_features = outputs.image_embeds.cpu()
                embeddings = torch.cat([embeddings, image_features.cpu()], dim=0)
            if model_type == 'combine':
                inputs = PLIP_processor(text=text_prompt, return_tensors="pt", padding=True)
                inputs['pixel_values'] = x_batch
                for key in inputs.keys():
                    inputs[key] = inputs[key].cuda()
                outputs = PLIP_model.forward(**inputs)
                logits_per_image = outputs.logits_per_image
                cur_probs1 = logits_per_image.softmax(dim=1)  
                texts = BMC_tokenizer(text_prompt).cuda()
                image_features, text_features, logit_scale = BMC_model(x_batch, texts)
                cur_probs2 = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
                probs = (cur_probs1+cur_probs2)/2
                embeddings = torch.cat((embeddings, torch.cat((outputs.image_embeds.cpu(), image_features.cpu()), dim=1)), dim=0)

            logits_hard = torch.argmax(probs, dim=1)
            pred_all = torch.cat((pred_all, logits_hard.cpu()), dim=0)
            prob_all = torch.cat((prob_all, probs.cpu()), dim=0)
            names += batch_names
            
    pred_all, prob_all, embeddings = pred_all[1:], prob_all[1:], embeddings[1:]
    y_pred = pred_all.numpy().astype(np.uint8)
    y_prob = prob_all

    return y_pred, y_prob, np.array(names), embeddings.clone().detach().cpu().numpy()

if __name__ == "__main__":
    start_time = time.time()
    args = get_arguments()
    seed_torch(args.seed)
    torch.cuda.set_device(args.gpu[0])

    # 💡 核心修改点：动态随机选择 3个ID 和 6个OOD
    all_classes = np.arange(9)
    # 使用给定的 seed 随机选 3 个
    args.id_cls = sorted(np.random.choice(all_classes, size=3, replace=False).tolist())
    args.ood_cls = sorted([c for c in all_classes if c not in args.id_cls])
    
    id_cls = args.id_cls
    ood_cls = args.ood_cls
    args.num_class = len(id_cls)

    print("="*50)
    print(f"🎲 随机种子 (Seed): {args.seed}")
    print(f"🟢 随机抽取的 3 个已知类 (ID): {id_cls}")
    print(f"🔴 剩下的 6 个未知类 (OOD): {ood_cls}")
    print("="*50)

    # dataset
    train_files, train_dict = get_files('/root/gpufree-data/OpenPath-main/al_file/train.csv')
    np.random.shuffle(train_files)

    train_id_files = [item for item in train_files if item['label'] in id_cls]
    train_ood_files = [item for item in train_files if item['label'] in ood_cls]
    print(f"ID样本数: {len(train_id_files)}, OOD样本数: {len(train_ood_files)}")
    
    train_files = copy.deepcopy(train_id_files) + copy.deepcopy(train_ood_files)
    np.random.shuffle(train_files)

    train_dataset_eval = Tumor_dataset_val_cls(args, files=train_files)
    train_eval_loader = get_loader(args, train_dataset_eval, shuffle=False)

    y_pred_raw, y_prob_raw, names, embeds = zero_shot_inference_random(args, train_eval_loader, id_cls, model_type=args.model_type)
    print(y_prob_raw.shape)

    re_id_cls = np.arange(len(id_cls))
    names_id_vlm = names[y_pred_raw<=len(id_cls)-1]
    names_id_gt = [item['img'] for item in train_id_files]

    names_idx = np.array([i for i, val in enumerate(names) if val in names_id_vlm])
    embeds_id = embeds[names_idx]

    # 计算阶段一 (Zero-Shot) 的战报数据
    vlm_raw_id_count = len(list(set(names_id_gt) & set(names_id_vlm)))
    vlm_raw_ood_count = len(names_id_vlm) - vlm_raw_id_count
    raw_qp = (vlm_raw_id_count / len(names_id_vlm) * 100) if len(names_id_vlm) > 0 else 0.0

    print(len(names_id_vlm), len(names_id_gt), vlm_raw_id_count, \
          raw_qp/100, 'vs.', len(train_id_files)/len(train_files))

    # kmeans++
    cluster_idx = kmean_cluster(embeds=embeds_id, n=args.init_num)
    names_init_select = names_id_vlm[cluster_idx]
    label_select = np.array([train_dict[item] for item in names_init_select])

    count = 0
    for name in names_init_select:
        if name in [item['img'] for item in train_id_files]:
            count += 1
    
    kmeans_qp = (count / args.init_num) * 100
    print(f'query precision: {kmeans_qp/100:.4f}')

    # 💡 修改点：确保保存路径文件夹存在并强制写入
    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        data_df = pd.DataFrame()
        data_df['img'] = names_init_select
        data_df['cls_label'] = label_select
        data_df.to_csv(args.save_csv, index=False)

    end_time = time.time()
    total_seconds = end_time - start_time
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # ==========================
    # 战报输出与日志追加
    # ==========================
    report_text = f"""
======================================================
🕒 运行时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
🎲 随机种子 (Seed): {args.seed}
🟢 选中已知类 (ID): {id_cls}
🔴 剩下未知类 (OOD): {ood_cls}
------------------------------------------------------
📊 阶段 1: Zero-Shot 零样本初筛池纯度 (QA/QP)
 ├─ 初筛总候选数: {len(names_id_vlm)}
 ├─ 真正目标类数量: {vlm_raw_id_count}
 ├─ 误入未知类数量: {vlm_raw_ood_count}
 └─ 💡 原始初筛精准度 (Raw QP): {raw_qp:.2f}%
------------------------------------------------------
🏆 阶段 2: 最终 {args.init_num} 张冷启动种子 QP
🔬 [Baseline: KMeans++ 聚类采样] (已保存至: {args.save_csv})
 ├─ 真正目标类(ID): {count} | 杂病(OOD): {args.init_num - count}
 └─ 💡 最终精准度(QP): {kmeans_qp:.2f}%
------------------------------------------------------
⏱️ 总运行耗时: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s
======================================================
"""

    # 打印到控制台
    print(report_text)
    
    # 💡 追加写入 experiment_results_old.log 文档
    log_file_path = "al_file/experiment_results_old.log" 
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"📝 Baseline 实验记录已成功追加至: {log_file_path}")