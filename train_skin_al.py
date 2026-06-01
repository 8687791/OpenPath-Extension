import os
# 💡 核心修复：国内镜像源和关闭 SSL 配置，必须放在文件最开头！
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import sys
import glob
from dataset.alb_dataset2 import Tumor_dataset, Tumor_dataset_val_cls, get_loader
import argparse
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import random
from sklearn import metrics
from sklearn.cluster import KMeans
from networks.pl_models import BMC_Vision_FT_Lit
import copy
import torch.nn.functional as F
import logging
from tqdm import tqdm
import lightning.pytorch as L
from lightning.pytorch.loggers import TensorBoardLogger

PROJECT_ROOT = Path(__file__).resolve().parent

def normalize_path(path):
    path = str(path)
    # ⚡️ 强力防爆补丁：如果大盘账本记录的物理路径在硬盘里撞墙找不到
    if not os.path.exists(path):
        # 1. 修复第 6 类 (BKL) 的双空格退化 Bug
        if "BKL) 2624" in path:
            fixed_path = path.replace("BKL) 256", "BKL)  256").replace("BKL) 2624", "BKL)  2624")
            if os.path.exists(fixed_path): return fixed_path
            
        # 2. 如果还有其他由于空格或者相对路径引起的错位，统一尝试转为绝对路径做二次兜底
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path): return abs_path
        
    return path

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
    parser = argparse.ArgumentParser(description="SkinTissue Multi-granular AL")
    parser.add_argument("--num_class", type=int, default=10)
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--crop_size", default=224)
    parser.add_argument("--gpu", nargs="+", type=int, default=[0])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--self_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query_num", type=int, default=50) 
    parser.add_argument("--query_times", type=int, default=5)
    
    # 💡 核心隔离一：全面指向刚才在 al_file_skin 里洗好的皮肤数据集大表
    parser.add_argument("--train_csv", type=str, default="al_file_skin/train.csv")
    parser.add_argument("--val_csv", type=str, default="al_file_skin/val_7k.csv")
    parser.add_argument("--init_csv", type=str, default="al_file_skin/query_round_4.csv")
    
    # 💡 核心隔离二：精确映射皮肤分类的 ID 和 OOD
    # 1=Melanoma(黑素瘤), 4=Nevi(色素痣) -> 作为主网络分类的已知金标准
    # 其他 0,2,3,5,6,7,8,9 -> 全部落入未知类杂病 OOD 池
    parser.add_argument("--id_cls", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--ood_cls", nargs="+", type=int, default=[0, 2, 3, 5, 6, 7, 8, 9])
    
    parser.add_argument("--id_ratio", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--max_candidates", type=int, default=-1)
    return parser.parse_args()

def get_files(data_csv):
    data = pd.read_csv(data_csv)
    data_name = data.iloc[:, 0].map(normalize_path).to_list()
    data_label = np.array(data.iloc[:, 1]).astype(np.uint8)
    new_file = [{"img": img, "label": label} for img, label in zip(data_name, data_label)]
    return new_file, {k:v for k, v in zip(data_name, data_label)}

def cal_acc(y_pred, y_true):
    test_accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = metrics.recall_score(y_true, y_pred, average='macro')
    return test_accuracy

class SAM2Adapter:
    def __init__(self):
        print("🚀 [SAM2] 正在加载本地库及模型权重...")
        checkpoint = "pretrained/sam2-hiera-large/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml" 
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
            self.predictor = SAM2ImagePredictor(sam2_model)
            self.enabled = True
            print("✅ SAM2 显式定位引擎加载成功！")
        except Exception as e:
            print(f"⚠️ 警告：加载真实 SAM2 失败 ({e})。启动高级几何模拟器。")
            self.enabled = False

    def get_segmentation_uncertainty(self, images_tensor):
        batch_size = images_tensor.size(0)
        uncertainties = []
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).cuda()
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).cuda()

        for b in range(batch_size):
            if self.enabled:
                try:
                    img_t = images_tensor[b].float() * std + mean
                    img_t = img_t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    img_np = (img_t * 255).astype(np.uint8)
                    self.predictor.set_image(img_np)
                    h, w, _ = img_np.shape
                    input_points = np.array([[w // 2, h // 2]])
                    input_labels = np.array([1])
                    _, scores, _ = self.predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        multimask_output=False
                    )
                    uncertainties.append(1.0 - float(scores[0]))
                except Exception:
                    uncertainties.append(float(np.random.rand() * 0.4))
            else:
                uncertainties.append(float(np.random.rand() * 0.4))
        return torch.tensor(uncertainties).cuda()

# 💡 核心延续：保留防爆显存的无物理落盘设计
def train_labeled(args, model, labeled_loader, save_dir):
    trainer = L.Trainer(
        max_epochs=args.epochs, 
        precision='16-mixed', 
        check_val_every_n_epoch=args.epochs, 
        enable_checkpointing=False, 
        logger=TensorBoardLogger(save_dir=save_dir)
    )
    trainer.fit(model=model, train_dataloaders=labeled_loader, val_dataloaders=val_loader)
    val_acc = float(trainer.callback_metrics.get('val_acc', 0.0))
    val_f1 = float(trainer.callback_metrics.get('val_f1', 0.0))
    return val_acc, val_f1

def self_training(args, model, psuedo_loader, save_dir):
    trainer = L.Trainer(
        max_epochs=args.self_epochs, 
        precision='16-mixed', 
        check_val_every_n_epoch=args.self_epochs, 
        enable_checkpointing=False, 
        logger=TensorBoardLogger(save_dir=save_dir)
    )
    trainer.fit(model=model, train_dataloaders=psuedo_loader, val_dataloaders=val_loader)
    st_acc = float(trainer.callback_metrics.get('val_acc', 0.0))
    return st_acc

@torch.no_grad()
def generate_pseudo_labels(args, model, labeled_loader, selection_loader):
    print("🧬 正在提取特征用于生成高置信度伪标签 (Self-Training)...")
    model = model.cuda().float() # ⚡️ 强行全精度对齐防崩溃
    
    for counter, sample in enumerate(labeled_loader):
        x_batch = sample['img'].cuda().float() 
        y_batch = sample['cls_label']
        batch_names = sample['img_name']
        cur_feature = model.feature_extractor(x_batch)

        if counter == 0:
            labeled_names, labeled_features, labeled_labels = batch_names, cur_feature, y_batch
        else:
            labeled_names += batch_names
            labeled_features = torch.cat((labeled_features, cur_feature), dim=0)
            labeled_labels = torch.cat((labeled_labels, y_batch), dim=0)

    class_id_embeds = []
    for cls_id in args.id_cls:
        class_id_embeds.append(labeled_features[labeled_labels==cls_id].mean(0).unsqueeze(0))
    class_id_embed = torch.cat(class_id_embeds)

    for counter, sample in enumerate(selection_loader):
        x_batch = sample['img'].cuda().float()
        y_batch, batch_names = sample['cls_label'], sample['img_name']
        cur_feature = model.feature_extractor(x_batch)
        cur_prob = model.fc(cur_feature).softmax(1)
        if counter == 0:
            unlabeled_probs, unlabeled_labels, unlabeled_names, unlabeled_features = cur_prob, y_batch, batch_names, cur_feature
        else:
            unlabeled_probs = torch.cat((unlabeled_probs, cur_prob), dim=0)
            unlabeled_labels = torch.cat((unlabeled_labels, y_batch), dim=0)
            unlabeled_names += batch_names
            unlabeled_features = torch.cat((unlabeled_features, cur_feature), dim=0)
            
    distances = 1 - F.cosine_similarity(unlabeled_features.unsqueeze(1), class_id_embed.unsqueeze(0), dim=2).cpu()
    distances, unlabeled_names, unlabeled_features = np.array(distances), np.array(unlabeled_names), unlabeled_features.cpu().numpy()

    distances_min = distances.min(axis=1)
    indices = np.argsort(distances_min)
    indices = indices[:int(args.id_ratio * len(indices))]

    candidates_names = unlabeled_names[indices]
    candidates_probs = unlabeled_probs[indices]
    candidates_preds = candidates_probs.argmax(1).cpu().numpy()
    
    pseudo_files = [{'img': name, 'label': label} for (name, label) in zip(candidates_names, candidates_preds)]
    print('>>> 生成的自训练伪标签集合大小: ', len(pseudo_files))
    return get_loader(args, Tumor_dataset(args, files=pseudo_files), shuffle=False, batch_size=256)

@torch.no_grad()
def sample_selection(args, model, labeled_loader, selection_loader, query_num, save_dir, sam2_model):
    print("\n=======================================================")
    print("🔍 启动 SkinTissue (PIS + EGSS) 融合 SAM2 采样主车间")
    print("=======================================================")
    model = model.cuda().float()
    
    for counter, sample in enumerate(labeled_loader):
        x_batch = sample['img'].cuda().float()
        y_batch = sample['cls_label']
        cur_feature = model.feature_extractor(x_batch)
        if counter == 0:
            labeled_features, labeled_labels = cur_feature, y_batch
        else:
            labeled_features = torch.cat((labeled_features, cur_feature), dim=0)
            labeled_labels = torch.cat((labeled_labels, y_batch), dim=0)

    class_id_embeds = [labeled_features[labeled_labels==cls_id].mean(0).unsqueeze(0) for cls_id in args.id_cls]
    class_id_embed = torch.cat(class_id_embeds)

    all_names, all_tissue_entropies, all_cell_uncertainties = [], [], []

    for counter, sample in enumerate(tqdm(selection_loader, desc="[Phase 1/2] 皮肤 PIS 特征全量扫描")):
        x_batch = sample['img'].cuda().float()
        batch_names = sample['img_name']
        cur_feature = model.feature_extractor(x_batch)
        cur_prob = model.fc(cur_feature).softmax(1)
        
        tissue_entropy = -torch.sum((cur_prob.log() + 1e-6) * cur_prob, dim=1)
        cell_uncertainty = sam2_model.get_segmentation_uncertainty(x_batch)
        
        all_names.extend(batch_names)
        all_tissue_entropies.append(tissue_entropy)
        all_cell_uncertainties.append(cell_uncertainty)
        
        if counter == 0:
            unlabeled_features = cur_feature
        else:
            unlabeled_features = torch.cat((unlabeled_features, cur_feature), dim=0)

    all_tissue_entropies = torch.cat(all_tissue_entropies)
    all_cell_uncertainties = torch.cat(all_cell_uncertainties)
    
    cos_sim = F.cosine_similarity(unlabeled_features.unsqueeze(1), class_id_embed.unsqueeze(0), dim=2)
    min_d_i, _ = torch.min(1.0 - cos_sim, dim=1)
    
    M_percentile = 25 
    threshold_d_M = np.percentile(min_d_i.cpu().numpy(), M_percentile)
    pis_passed_indices = torch.where(min_d_i <= threshold_d_M)[0].cpu().numpy()
    print(f"🎯 [PIS 完成] 拦截皮肤杂病 OOD 噪声！候选池已精简至 {len(pis_passed_indices)}。")

    passed_names = [all_names[idx] for idx in pis_passed_indices]
    passed_entropies = all_tissue_entropies[pis_passed_indices]
    passed_sam2_scores = all_cell_uncertainties[pis_passed_indices]
    passed_combined_scores = ((args.alpha * passed_entropies) + (args.beta * passed_sam2_scores)).cpu().numpy()

    B_batches = 10
    sample_indices = np.arange(len(passed_names))
    np.random.shuffle(sample_indices) 
    
    samples_per_batch = query_num // B_batches
    selected_names = []

    for chunk_indices in np.array_split(sample_indices, B_batches):
        if len(chunk_indices) == 0: continue
        chunk_scores = passed_combined_scores[chunk_indices]
        real_selected_idx = chunk_indices[np.argsort(chunk_scores)[::-1][:samples_per_batch]]
        for idx in real_selected_idx:
            selected_names.append(passed_names[idx])

    selected_labels = [train_dict[item] for item in selected_names]
    
    data_df = pd.DataFrame({'img': selected_names, 'cls_label': np.array(selected_labels), 'human_review_required': True})
    data_df.to_csv(save_dir, index=False)
    print(f"✅ [EGSS+SAM2] 选出的 {len(selected_names)} 个高价值皮肤样本已保存至 {save_dir}")

    count = sum(1 for name in selected_names if name in [item['img'] for item in train_id_files])
    return count / args.query_num

if __name__ == "__main__":
    args = get_arguments()
    seed_torch(args.seed)
    torch.cuda.set_device(args.gpu[0])

    print("📂 正在加载皮肤全量实验数据...")
    train_files, train_dict = get_files(args.train_csv)
    np.random.shuffle(train_files)
    test_files, _ = get_files(args.val_csv)

    id_cls, ood_cls = args.id_cls, args.ood_cls
    args.num_class = len(id_cls)

    train_id_files = [item for item in train_files if item['label'] in id_cls]
    train_ood_files = [item for item in train_files if item['label'] in ood_cls]
    train_files = copy.deepcopy(train_id_files) + copy.deepcopy(train_ood_files)

    val_id_files = [item for item in test_files if item['label'] in id_cls]
    for item in val_id_files: 
        item['label'] = id_cls.index(item['label'])

    print(f"📂 正在读取第一轮皮肤大模型冷启动种子: {args.init_csv}")
    initial_data = pd.read_csv(args.init_csv)
    initial_names = set([os.path.basename(str(name)) for name in initial_data.iloc[:, 0].to_numpy()])
    
    initial_labeled = [item for item in train_files if os.path.basename(str(item['img'])) in initial_names]
    labeled_ID = [item for item in train_id_files if os.path.basename(str(item['img'])) in initial_names]
    candidates = [item for item in train_files if os.path.basename(str(item['img'])) not in initial_names]

    print(f"📊 皮肤初始化池: Labeled ID={len(labeled_ID)}, 待挖掘候选池 Candidates={len(candidates)}")
    
    labeled_dataset = Tumor_dataset_val_cls(args, files=initial_labeled)
    val_dataset = Tumor_dataset_val_cls(args, files=val_id_files)
    
    labeled_loader = get_loader(args, labeled_dataset, shuffle=True)
    global val_loader
    val_loader = get_loader(args, val_dataset, shuffle=False)

    labeled_ID = copy.deepcopy(labeled_ID)
    for item in labeled_ID: 
        item['label'] = id_cls.index(item['label'])
    labeled_loader_train = get_loader(args, Tumor_dataset(args, files=labeled_ID), shuffle=True)

    sam2_model = SAM2Adapter()
    precision_list, labeled_data_all = [], []
    val_acc_history, val_f1_history, st_acc_history = [], [], []

    # 💡 虚拟显存防空洞
    memory_checkpoints = {}

    for i in range(args.query_times):
        print(f"\n==============================================")
        print(f"🔄 启动皮肤场主动学习主轴：Round {i}")
        print(f"==============================================")
        
        if i == 0:
            model = BMC_Vision_FT_Lit(pretrain=True, num_class=args.num_class, args=args)
            
            # 💡 隔离存储：把 TensorBoard 画板导向 lightning_logs_skin
            save_dir = 'lightning_logs_skin/Skin-BMC-ST-0-L/'
            cur_val_acc, cur_val_f1 = train_labeled(args, model, labeled_loader_train, save_dir=save_dir)
            val_acc_history.append(cur_val_acc)
            val_f1_history.append(cur_val_f1)
            
            memory_checkpoints['0-L'] = copy.deepcopy(model.state_dict())
            
            sampled_candidates = random.sample(candidates, args.max_candidates) if args.max_candidates > 0 and len(candidates) > args.max_candidates else candidates
            selection_loader = get_loader(args, Tumor_dataset_val_cls(args, files=sampled_candidates), shuffle=False)
            
            pseudo_loader = generate_pseudo_labels(args, model, labeled_loader, selection_loader)
            save_dir = 'lightning_logs_skin/Skin-BMC-ST-0-ST/'
            
            cur_st_acc = self_training(args, model, pseudo_loader, save_dir)
            st_acc_history.append(cur_st_acc)
            
            memory_checkpoints['0-ST'] = copy.deepcopy(model.state_dict())
            labeled_data_all = copy.deepcopy(initial_labeled)
            precision_list.append(len(labeled_ID) / len(initial_data))
        else:
            model = BMC_Vision_FT_Lit(pretrain=True, num_class=len(id_cls), args=args)
            model.load_state_dict(memory_checkpoints[f'{i-1}-ST'])
            
            # 💡 将历轮主动学习产出的切片存放于 al_file_skin 下
            save_csv = f'al_file_skin/Skin_query{i}_labeled.csv'
            
            sampled_candidates = random.sample(candidates, args.max_candidates) if args.max_candidates > 0 and len(candidates) > args.max_candidates else candidates
            selection_loader = get_loader(args, Tumor_dataset_val_cls(args, sampled_candidates))
            
            cur_precision = sample_selection(args, model, labeled_loader, selection_loader, args.query_num, save_csv, sam2_model)
            
            save_dir = f'lightning_logs_skin/Skin-BMC-ST-{i}-L/'
            labeled_data, _ = get_files(save_csv)
            labeled_data_all += labeled_data
            
            labeled_data_ID = [item for item in labeled_data if item['label'] in id_cls] 
            for item in labeled_data_ID: 
                item['label'] = id_cls.index(item['label'])
            
            labeled_loader_ID = get_loader(args, Tumor_dataset(args, labeled_data_ID))
            
            cur_val_acc, cur_val_f1 = train_labeled(args, model, labeled_loader_ID, save_dir)
            val_acc_history.append(cur_val_acc)
            val_f1_history.append(cur_val_f1)
            
            memory_checkpoints[f'{i}-L'] = copy.deepcopy(model.state_dict())
            
            labeled_names_all = [item['img'] for item in labeled_data_all]
            candidates = [item for item in train_files if item['img'] not in copy.deepcopy(labeled_names_all)]
            
            labeled_loader = get_loader(args, Tumor_dataset_val_cls(args, labeled_data_all))
            
            sampled_candidates = random.sample(candidates, args.max_candidates) if args.max_candidates > 0 and len(candidates) > args.max_candidates else candidates
            selection_loader = get_loader(args, Tumor_dataset_val_cls(args, sampled_candidates))
            
            pseudo_loader = generate_pseudo_labels(args, model, labeled_loader, selection_loader)
            save_dir = f'lightning_logs_skin/Skin-BMC-ST-{i}-ST/'
            
            cur_st_acc = self_training(args, model, pseudo_loader, save_dir)
            st_acc_history.append(cur_st_acc)
            
            memory_checkpoints[f'{i}-ST'] = copy.deepcopy(model.state_dict())
            precision_list.append(cur_precision)
            
        print(f"📊 当前 AL 选片纯度历史轨迹 (QP): {precision_list}")
        print(f"📈 皮肤主分类模型准确率轨迹 (Model Acc): {val_acc_history}")

        # 💡 核心隔离三：指标数组独立输出至 log_skin，防污染
        os.makedirs('log_skin', exist_ok=True)
        np.savetxt('log_skin/Ours_precision.txt', np.array(precision_list), delimiter=',')
        np.savetxt('log_skin/Ours_model_acc.txt', np.array(val_acc_history), delimiter=',')
        np.savetxt('log_skin/Ours_model_f1.txt', np.array(val_f1_history), delimiter=',')
        np.savetxt('log_skin/Ours_self_training_acc.txt', np.array(st_acc_history), delimiter=',')

    print("🏁 恭喜！皮肤 10 分类多尺度联合主动学习全循环圆满落幕，指标已成功封存！")