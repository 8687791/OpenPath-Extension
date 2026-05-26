import os

# 配置国内镜像源与关闭SSL验证（必须放在最开头）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import sys
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
    parser = argparse.ArgumentParser(description="SkinTissue CLIP-ZeroShot implementation")
    # 💡 核心修改一：将全量大盘分类修改为皮肤的 10 分类
    parser.add_argument("--num_class", type=int, default=10, help="Skin disease class num")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--crop_size", default=224)
    parser.add_argument("--gpu", nargs="+", type=int, default=[0])
    parser.add_argument("--batch_size", type=int, default=512, help="Train batch size")
    parser.add_argument("--num_workers", default=6)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model_type", type=str, default='combine')
    
    # 💡 核心修改二：按照你当前的设定：1=Melanoma (黑色素瘤), 4=Nevi (色素痣) 设为 ID 黄金已知类
    parser.add_argument("--id_cls", nargs="+", type=int, default=[1, 4])
    # 其余 8 类杂病全部退归为 OOD 拦截池
    parser.add_argument("--ood_cls", nargs="+", type=int, default=[0, 2, 3, 5, 6, 7, 8, 9])
    
    # 💡 核心修改三：默认将挑选出的 50 张冷启动种子保存到 al_file_skin 独立空间中
    parser.add_argument("--save_csv", type=str, default="al_file_skin/clip_query_round_1.csv")
    parser.add_argument("--init_num", type=int, default=50)
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


def cal_acc(y_pred, y_true):
    test_accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = metrics.recall_score(y_true, y_pred, average='macro')
    print("各类别召回率轨迹:", metrics.recall_score(y_true, y_pred, average=None))
    print(f"盲测准确率 Test Accuracy: {test_accuracy.item()}, F1-Score: {f1}, Precision: {p}, Recall: {r}")


def kmean_cluster(embeds, n):
    cluster_learner = KMeans(n_clusters=n, init='k-means++', n_init='auto', random_state=42)
    cluster_learner.fit(embeds)
    cluster_idxs = cluster_learner.predict(embeds)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embeds - centers) ** 2
    dis = dis.sum(axis=1)
    q_idx = np.array([np.arange(embeds.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n)])
    return q_idx


def zero_shot_inference_random(args, train_eval_loader, id_cls, model_type='combine'):
    # 加载本地缓存或镜像站的 BiomedCLIP 预训练模型
    BMC_model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    BMC_tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    BMC_model.cuda().eval()

    # 加载本地缓存的 PLIP 皮肤/病理预训练大模型
    PLIP_model = CLIPModel.from_pretrained("vinid/plip")
    PLIP_processor = CLIPProcessor.from_pretrained("vinid/plip")
    PLIP_model.cuda().eval()

    # 💡 核心修改四：全面重构提示词矩阵。将原有的肠癌组织文本，全面对齐替换为符合皮肤临床/病理特征的专属医学 Prompt 描述
    skin_text_prompts = [
        "A dermoscopy image of Eczema showing erythematous plaques",                    # 0. 湿疹
        "A dermoscopy image of malignant Melanoma with irregular pigmentation",          # 1. 黑色素瘤 (ID 目标)
        "A dermoscopy image of Atopic Dermatitis with pruritic lesions",                 # 2. 特应性皮炎
        "A dermoscopy image of Basal Cell Carcinoma with pearly papules",                # 3. 基底细胞癌
        "A dermoscopy image of Melanocytic Nevi showing symmetric benign nest",          # 4. 色素痣 (ID 目标)
        "A dermoscopy image of Benign Keratosis-like Lesions",                           # 5. 良性角化病样皮损
        "A dermoscopy image of Psoriasis showing silvery scales",                        # 6. 银屑病/扁平苔藓
        "A dermoscopy image of Seborrheic Keratoses with stuck-on appearance",           # 7. 脂溢性角化病
        "A dermoscopy image of Fungal Infection showing tinea ringworm patterns",        # 8. 真菌感染/癣
        "A dermoscopy image of Viral Infection showing warts or molluscum papules"       # 9. 病毒感染/疣
    ]

    # 💡 核心修改五：构建科学的 OOD 外部混淆文本干扰库（杂病噪音），用于迷惑 CLIP 分类边界，提取高质边界特征
    skin_text_prompts_random = [
        "A dermoscopy image of inflammatory dermatosis skin lesion",
        "A dermoscopy image of benign hyperplastic cutaneous lesion",
        "A dermoscopy image of atypical vascular structures in skin tissue",
        "A dermoscopy image of hyperkeratotic epidermal proliferation",
        "A dermoscopy image of superficial fungal tinea skin infection"
    ]

    # 抽取选定的目标已知类提示词，并无缝接合前 3 项随机干扰描述，构成严密的完形填空空间
    text_prompt = list(np.array(skin_text_prompts)[np.array(id_cls)])
    text_prompt += list(np.array(skin_text_prompts_random)[:3])
    print("\n🔬 [皮肤战场] 当前 CLIP 零样本推理空间使用的文本矩阵为:")
    for idx, prompt in enumerate(text_prompt):
        print(f"   ├─ Prompt {idx}: {prompt}")
    print("-" * 65)

    with torch.no_grad():
        pred_all, prob_all = torch.zeros((1,)), torch.zeros((1, len(text_prompt)))
        if model_type == 'plip' or model_type == 'BMC':
            embeddings = torch.zeros((1, 512))
        if 'combine' in model_type:
            embeddings = torch.zeros((1, 1024))
        names = []
        
        for counter, sample in enumerate(train_eval_loader):
            x_batch = sample['img'].cuda()
            batch_names = sample['img_name']

            if counter == 0:
                print(f"🚀 首批切片就位，开始大规模跨模态视觉特征提取: {batch_names[0]}")

            # 1. 执行 BiomedCLIP 双塔特征检索
            if model_type == 'BMC':
                texts = BMC_tokenizer(text_prompt).cuda()
                image_features, text_features, logit_scale = BMC_model(x_batch, texts)
                probs = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
                embeddings = torch.cat([embeddings, image_features.cpu()], dim=0)
                
            # 2. 执行 PLIP 双塔特征检索
            elif model_type == 'plip':
                inputs = PLIP_processor(text=text_prompt, return_tensors="pt", padding=True)
                inputs['pixel_values'] = x_batch
                for key in inputs.keys(): inputs[key] = inputs[key].cuda()
                
                outputs = PLIP_model.forward(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                image_features = outputs.image_embeds.cpu()
                embeddings = torch.cat([embeddings, image_features.cpu()], dim=0)
                
            # 3. 强效模式：融合双强地基模型，求取集成平滑概率 (Ensemble Probabilities)
            elif model_type == 'combine':
                inputs = PLIP_processor(text=text_prompt, return_tensors="pt", padding=True)
                inputs['pixel_values'] = x_batch
                for key in inputs.keys(): inputs[key] = inputs[key].cuda()
                outputs = PLIP_model.forward(**inputs)
                cur_probs1 = outputs.logits_per_image.softmax(dim=1)
                
                texts = BMC_tokenizer(text_prompt).cuda()
                image_features, text_features, logit_scale = BMC_model(x_batch, texts)
                cur_probs2 = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
                
                probs = (cur_probs1 + cur_probs2) / 2
                embeddings = torch.cat(
                    (embeddings, torch.cat((outputs.image_embeds.cpu(), image_features.cpu()), dim=1)), dim=0)

            logits_hard = torch.argmax(probs, dim=1)
            pred_all = torch.cat((pred_all, logits_hard.cpu()), dim=0)
            prob_all = torch.cat((prob_all, probs.cpu()), dim=0)
            names += batch_names

    pred_all, prob_all, embeddings = pred_all[1:], prob_all[1:], embeddings[1:]
    y_pred = pred_all.numpy().astype(np.uint8)
    y_prob = prob_all

    return y_pred, y_prob, np.array(names), embeddings.clone().detach().cpu().numpy()


if __name__ == "__main__":
    args = get_arguments()
    seed_torch(args.seed)
    torch.cuda.set_device(args.gpu[0])

    # 💡 核心修改六：数据表和账本读取完全对齐之前用 make_csv_skin.py 洗好的皮肤大表
    train_csv_skin = 'al_file_skin/train.csv'
    print(f"📂 正在解析皮肤全量训练池: {train_csv_skin}...")
    train_files, train_dict = get_files(train_csv_skin)
    np.random.shuffle(train_files)

    id_cls = args.id_cls
    ood_cls = args.ood_cls
    args.num_class = len(id_cls)

    # 精准隔离划分皮肤战场的 ID 样本与 OOD 噪声
    train_id_files = [item for item in train_files if item['label'] in id_cls]
    train_ood_files = [item for item in train_files if item['label'] in ood_cls]
    print(f"📊 [皮肤全盘面解析成功]: 真正目标类(ID)共 {len(train_id_files)} 张，外部混淆杂病(OOD)共 {len(train_ood_files)} 张。")
    
    train_files = copy.deepcopy(train_id_files) + copy.deepcopy(train_ood_files)
    np.random.shuffle(train_files)

    train_dataset_eval = Tumor_dataset_val_cls(args, files=train_files)
    train_loader = get_loader(args, train_dataset_eval, shuffle=False)

    # 启动零样本跨模态推理主轴
    y_pred_raw, y_prob_raw, names, embeds = zero_shot_inference_random(args, train_loader, id_cls,
                                                                       model_type=args.model_type)

    names_id_vlm = names[y_pred_raw <= len(id_cls) - 1]
    names_id_gt = [item['img'] for item in train_id_files]

    names_idx = np.array([i for i, val in enumerate(names) if val in names_id_vlm])
    embeds_id = embeds[names_idx]

    print("\n📊 --- 空间对齐与初筛纯度报告 ---")
    print(f" ├─ 双塔CLIP认出的已知类候选总数 (VLM Approved): {len(names_id_vlm)}")
    print(f" ├─ 硬盘物理存储的真实已知类总数 (Ground-Truth ID): {len(names_id_gt)}")
    
    intersection_count = len(list(set(names_id_gt) & set(names_id_vlm)))
    raw_purity = intersection_count / len(names_id_vlm) if len(names_id_vlm) > 0 else 0
    natural_purity = len(train_id_files) / len(train_files)
    
    print(f" ├─ 盲测交集命中有效数 (True Positives): {intersection_count}")
    print(f" ├─ 💡 初筛候选池天然纯度 (VLM Raw Purity): {raw_purity * 100:.2f}% (对比全盘盲选自然占比: {natural_purity * 100:.2f}%)")
    print("-" * 45)

    # 引入学术标准的多样性 K-Means++ 空间采样
    print(f"🎯 正在对初筛出的候选空间执行冷启动多样性 K-Means++ 空间约束采样...")
    cluster_idx = kmean_cluster(embeds=embeds_id, n=args.init_num)
    names_init_select = names_id_vlm[cluster_idx]
    label_select = np.array([train_dict[item] for item in names_init_select])

    # 验证最终挑选出的 50 张冷启动种子的终极纯度 (Query Precision)
    count = 0
    for name in names_init_select:
        if name in [item['img'] for item in train_id_files]:
            count += 1
    qp_score = (count / args.init_num) * 100
    print(f"\n✨ 【皮肤冷启动消融实验核心指标】")
    print(f" └─ 💡 最终选出的 {args.init_num} 张种子图片查询精准度 Query Precision (QP): {qp_score:.2f}%")
    print("-" * 55)

    # 结果安全物理落盘，保存于皮肤专属工作室内，严防污染
    data_df = pd.DataFrame()
    data_df['img'] = names_init_select
    data_df['cls_label'] = label_select
    
    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        data_df.to_csv(args.save_csv, index=False)
        print(f"✅ 皮肤战场第一轮 CLIP 零样本冷启动种子挑选完成！结果已保存在: {args.save_csv}")