import os
# 配置国内镜像源（必须放在最开头）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import sys
import glob
import copy
import argparse
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import lightning.pytorch as L
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# 导入原本的数据集加载器与模型
from dataset.alb_dataset2 import Tumor_dataset, Tumor_dataset_val_cls, get_loader
from networks.pl_models import BMC_Vision_FT_Lit

# 🌟 多尺度主动学习：尝试导入本地的 sam2 模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'sam2'))
try:
    from sam2.build_sam import build_sam2
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("⚠️ 未能在环境中检测到完整的 sam2 库，将降级使用单尺度采样。请确保 sam2 文件夹在根目录中。")


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
    options_parser = argparse.ArgumentParser(description="CRC PyTorch Implementation")
    # 🎯 修改点：NCT-CRC-HE-100K 共有 9 个类别
    options_parser.add_argument("--num_class", type=int, default=9, help="Total class num in dataset")
    options_parser.add_argument("--input_size", default=256, type=int)
    options_parser.add_argument("--crop_size", default=224, type=int)
    options_parser.add_argument("--gpu", nargs="+", type=int, default=[0])
    options_parser.add_argument("--batch_size", type=int, default=128, help="Train batch size")
    options_parser.add_argument("--num_workers", default=6, type=int)
    options_parser.add_argument("--epochs", type=int, default=50)
    options_parser.add_argument("--self_epochs", type=int, default=50)
    options_parser.add_argument("--seed", type=int, default=42)
    options_parser.add_argument("--query_num", type=int, default=50)
    options_parser.add_argument("--query_times", type=int, default=5)
    # 🎯 修改点：日志文件路径换为 log1_CRC
    options_parser.add_argument("--log", type=str, default='log1_CRC/train_crc_sup.txt')
    options_parser.add_argument("--lr", type=float, default=1e-3)
    options_parser.add_argument("--save_dir", type=str, default='lightning_logs')
    
    # 🎯 修改点：默认读取的冷启动初始种子路径
    options_parser.add_argument("--init_csv", type=str, default="al_file/query_round_5_graph.csv")
    
    # 🎯 修改点：ID (已知类) [3, 6, 8] 与 OOD (未知类) 剩余的 6 个类别
    options_parser.add_argument("--id_cls", nargs="+", type=int, default=[3, 6, 8])
    options_parser.add_argument("--ood_cls", nargs="+", type=int, default=[0, 1, 2, 4, 5, 7])
    
    # 多尺度主动学习：SAM2 模型路径
    options_parser.add_argument("--sam2_ckpt", type=str, default="pretrained/sam2_hiera_large.pt")
    options_parser.add_argument("--sam2_cfg", type=str, default="sam2_hiera_l.yaml")
    
    options_parser.add_argument("--id_ratio", type=float, default=1e-3)
    return options_parser.parse_args()


def get_files(data_csv):
    data = pd.read_csv(data_csv)
    data_name = data.iloc[:, 0]
    if data.shape[1] > 1:
        data_label = data.iloc[:, 1]
    else:
        data_label = [0] * len(data_name)
        
    data_label = np.array(data_label).astype(np.uint8)
    data_name = data_name.to_list()
    new_file = [{"img": img, "label": label} for img, label in zip(data_name, data_label)]
    new_dict = {k: v for k, v in zip(data_name, data_label)}
    return new_file, new_dict


def kmean_cluster(embeds, n):
    num_samples = embeds.shape[0]
    actual_n = min(n, num_samples)
    
    if actual_n == 0:
        print("⚠️ 警告: 候选样本为空，无法进行 K-Means 聚类！")
        return []
        
    if actual_n < n:
        print(f"⚠️ 提示: 剩余候选样本数 ({num_samples}) 小于设定的查询数 ({n})，"
              f"K-Means 聚类数已自动调整为 {actual_n}")

    cluster_learner = KMeans(n_clusters=actual_n, random_state=0, n_init=10)
    cluster_learner.fit(embeds)
    
    centers = cluster_learner.cluster_centers_
    dist_matrix = cdist(centers, embeds, metric='euclidean')
    closest_indices = np.argmin(dist_matrix, axis=1)
    
    closest_indices = list(set(closest_indices))
    return closest_indices


def cal_acc(y_pred, y_true):
    test_accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Test Accuracy: {test_accuracy}, f1:{f1}, precision:{p}, recall:{r}")
    return test_accuracy, f1, r, p


def cls_count(args, y_pred):
    count = [0] * args.num_class
    for item in y_pred:
        if item < args.num_class:
            count[item] += 1
    print(count)
    return count


def train_labeled(args, model, labeled_loader, val_loader, save_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        filename='best-model-{epoch:02d}-{val_acc:.3f}',
        save_last=False,
    )
    trainer = L.Trainer(max_epochs=args.epochs, precision='16-mixed', check_val_every_n_epoch=args.epochs, callbacks=[checkpoint_callback],
                        logger=TensorBoardLogger(save_dir=save_dir))
    trainer.fit(model=model, train_dataloaders=labeled_loader, val_dataloaders=val_loader)


def self_training(args, model, psuedo_loader, val_loader, save_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        filename='best-model-{epoch:02d}-{val_acc:.3f}',
        save_last=False,
    )
    trainer = L.Trainer(max_epochs=args.self_epochs, precision='16-mixed', check_val_every_n_epoch=args.self_epochs, callbacks=[checkpoint_callback],
                        logger=TensorBoardLogger(save_dir=save_dir))
    trainer.fit(model=model, train_dataloaders=psuedo_loader, val_dataloaders=val_loader)


@torch.no_grad()
def generate_pseudo_labels(args, model, labeled_loader, selection_loader, train_dict, id_cls):
    for counter, sample in enumerate(labeled_loader):
        x_batch = sample['img'].cuda()
        y_batch = sample['cls_label']
        batch_names = sample['img_name']
        cur_feature = model.feature_extractor(x_batch)

        if counter == 0:
            labeled_names = batch_names
            labeled_features = cur_feature
            labeled_labels = y_batch
        else:
            labeled_names += batch_names
            labeled_features = torch.cat((labeled_features, cur_feature), dim=0)
            labeled_labels = torch.cat((labeled_labels, y_batch), dim=0)

    for counter, cls_id in enumerate(id_cls):
        if counter == 0:
            class_id_embed = labeled_features[labeled_labels == cls_id].mean(0).unsqueeze(0)
        else:
            class_id_embed = torch.cat((class_id_embed, labeled_features[labeled_labels == cls_id].mean(0).unsqueeze(0)))

    for counter, sample in enumerate(selection_loader):
        x_batch = sample['img'].cuda()
        y_batch = sample['cls_label']
        batch_names = sample['img_name']
        cur_feature = model.feature_extractor(x_batch)
        cur_prob = model.fc(cur_feature).softmax(1)
        if counter == 0:
            unlabeled_probs = cur_prob
            unlabeled_labels = y_batch
            unlabeled_names = batch_names
            unlabeled_features = cur_feature
        else:
            unlabeled_probs = torch.cat((unlabeled_probs, cur_prob), dim=0)
            unlabeled_labels = torch.cat((unlabeled_labels, y_batch), dim=0)
            unlabeled_names += batch_names
            unlabeled_features = torch.cat((unlabeled_features, cur_feature), dim=0)
            
    distances = 1 - F.cosine_similarity(unlabeled_features.unsqueeze(1), class_id_embed.unsqueeze(0), dim=2).cpu()
    distances, unlabeled_names, unlabeled_features = np.array(distances), np.array(unlabeled_names), unlabeled_features.cpu().numpy()

    distances_min = distances.min(axis=1)
    indices = np.argsort(distances_min)
    indices = indices[:max(args.query_num * 2, int(args.id_ratio * len(indices)))] 

    candidates_features = unlabeled_features[indices]
    candidates_names = unlabeled_names[indices]
    candidates_probs = unlabeled_probs[indices]

    candidates_labels = []
    for item in candidates_names:
        actual_key = item.item() if hasattr(item, 'item') else item
        candidates_labels.append(train_dict.get(actual_key, 0))

    cand_prec = count = 0
    for item in candidates_labels:
        if int(item) in id_cls:
            count += 1
    cand_prec = count / len(candidates_labels) if len(candidates_labels) > 0 else 0
    print('ID/OOD candidates precision:', cand_prec)

    candidates_preds = candidates_probs.argmax(1).cpu().numpy()
    candidates_labels_re = candidates_labels.copy()
    for i in range(len(candidates_labels_re)):
        if candidates_labels_re[i] in id_cls:
            candidates_labels_re[i] = id_cls.index(candidates_labels_re[i])
        else:
            candidates_labels_re[i] = len(id_cls)
            
    test_accuracy, f1, r, p = cal_acc(candidates_preds, np.array(candidates_labels_re))

    pseudo_files = [{'img': name, 'label': label} for (name, label) in zip(candidates_names, candidates_preds)]
    print('pseudo labels size: ', len(pseudo_files))
    
    metrics_dict = {
        'candidate_precision': cand_prec,
        'test_acc': test_accuracy,
        'f1': f1,
        'recall': r,
        'precision': p
    }
    return get_loader(args, Tumor_dataset(args, files=pseudo_files), shuffle=False, batch_size=256), metrics_dict


@torch.no_grad()
def sample_selection(args, model, labeled_loader, selection_loader, query_num, save_dir, train_dict, id_cls, train_id_files):
    sam2_model = None
    if SAM2_AVAILABLE and os.path.exists(args.sam2_ckpt):
        print(f"🧬 正在加载 SAM2 基础大模型用于多尺度形态学对齐: {args.sam2_ckpt}")
        sam2_model = build_sam2(args.sam2_cfg, args.sam2_ckpt, device="cuda")
        sam2_model.eval()

    for counter, sample in enumerate(labeled_loader):
        x_batch = sample['img'].cuda()
        y_batch = sample['cls_label']
        batch_names = sample['img_name']
        cur_feature = model.feature_extractor(x_batch)

        if counter == 0:
            labeled_names = batch_names
            labeled_features = cur_feature
            labeled_labels = y_batch
        else:
            labeled_names += batch_names
            labeled_features = torch.cat((labeled_features, cur_feature), dim=0)
            labeled_labels = torch.cat((labeled_labels, y_batch), dim=0)

    for counter, cls_id in enumerate(id_cls):
        if counter == 0:
            class_id_embed = labeled_features[labeled_labels == cls_id].mean(0).unsqueeze(0)
        else:
            class_id_embed = torch.cat((class_id_embed, labeled_features[labeled_labels == cls_id].mean(0).unsqueeze(0)))

    for counter, sample in enumerate(selection_loader):
        x_batch = sample['img'].cuda()
        y_batch = sample['cls_label']
        batch_names = sample['img_name']
        
        cur_feature = model.feature_extractor(x_batch)
        cur_prob = model.fc(cur_feature).softmax(1)

        if sam2_model is not None:
            micro_batch_size = 4  
            cell_embeds_list = []
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for start_idx in range(0, x_batch.size(0), micro_batch_size):
                    end_idx = min(start_idx + micro_batch_size, x_batch.size(0))
                    mb_x = x_batch[start_idx:end_idx]
                    
                    mb_x_sam = F.interpolate(mb_x, size=(1024, 1024), mode='bilinear', align_corners=False)
                    mb_backbone_features = sam2_model.image_encoder(mb_x_sam)
                    
                    # 🛡️ 绝杀修复层：对官方原生 SAM2 字典拓扑树进行全自动对齐
                    if isinstance(mb_backbone_features, dict):
                        mb_feats = mb_backbone_features.get("vision_features", mb_backbone_features.get("image_embed", list(mb_backbone_features.values())[0]))
                    else:
                        mb_feats = mb_backbone_features
                        
                    mb_cell_level_feat = mb_feats[0] if isinstance(mb_feats, (list, tuple)) else mb_feats
                    
                    mb_cell_embed = F.adaptive_avg_pool2d(mb_cell_level_feat, (1, 1)).flatten(1)
                    cell_embeds_list.append(mb_cell_embed)
            
            cell_embed = torch.cat(cell_embeds_list, dim=0).to(cur_feature.dtype)
            fused_feature = torch.cat([cur_feature, cell_embed], dim=1)
        else:
            fused_feature = cur_feature

        if counter == 0:
            unlabeled_probs = cur_prob
            unlabeled_labels = y_batch
            unlabeled_names = batch_names
            unlabeled_features = fused_feature 
            unlabeled_tissue_features = cur_feature 
        else:
            unlabeled_probs = torch.cat((unlabeled_probs, cur_prob), dim=0)
            unlabeled_labels = torch.cat((unlabeled_labels, y_batch), dim=0)
            unlabeled_names += batch_names
            unlabeled_features = torch.cat((unlabeled_features, fused_feature), dim=0)
            unlabeled_tissue_features = torch.cat((unlabeled_tissue_features, cur_feature), dim=0)
            
    distances = 1 - F.cosine_similarity(unlabeled_tissue_features.unsqueeze(1), class_id_embed.unsqueeze(0), dim=2).cpu()
    distances, unlabeled_names = np.array(distances), np.array(unlabeled_names)
    unlabeled_features = unlabeled_features.cpu().numpy()

    distances_min = np.array(distances).min(axis=1)
    indices = np.argsort(distances_min)
    
    buffer_len = max(query_num * 5, int(args.id_ratio * len(indices)))
    indices = indices[:buffer_len]

    candidates_features = unlabeled_features[indices] 
    candidates_names = unlabeled_names[indices]
    candidates_probs = unlabeled_probs[indices]

    candidates_labels = []
    for item in candidates_names:
        actual_key = item.item() if hasattr(item, 'item') else item
        candidates_labels.append(train_dict.get(actual_key, 0))

    count = 0
    for item in candidates_labels:
        if int(item) in id_cls:
            count += 1
    print('ID/OOD candidates precision:', count / len(candidates_labels) if len(candidates_labels) > 0 else 0)

    print("🧬 正在执行多尺度对齐 K-Means 聚类以寻找结构模糊的长尾样本...")
    cluster_idx = kmean_cluster(embeds=candidates_features, n=query_num)
    
    selected_names = np.array(candidates_names)[cluster_idx]
    
    selected_labels = []
    for item in selected_names:
        actual_key = item.item() if hasattr(item, 'item') else item
        selected_labels.append(train_dict.get(actual_key, 0))
    
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    data_df = pd.DataFrame()
    data_df['img'] = selected_names
    data_df['cls_label'] = np.array(selected_labels)
    data_df.to_csv(save_dir, index=False)

    count = 0
    id_img_basenames = set([os.path.basename(str(item['img'])) for item in train_id_files])
    for name in selected_names:
        if os.path.basename(str(name)) in id_img_basenames:
            count += 1
            
    actual_query_num = len(selected_names)
    query_precision = count / actual_query_num if actual_query_num > 0 else 0
    print('query precision: ', query_precision)
    
    if sam2_model is not None:
        del sam2_model
        torch.cuda.empty_cache()
        
    return query_precision


if __name__ == "__main__":
    args = get_arguments()
    seed_torch(args.seed)
    torch.cuda.set_device(args.gpu[0])

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    l = logging.getLogger(__name__)
    fileHandler = logging.FileHandler(args.log, mode='a')
    l.setLevel(logging.INFO)
    l.addHandler(fileHandler)

    # 🎯 修改点：读取 CRC 数据的 CSV 文件，请确保你在 al_file 下准备好了这两个文件
    train_files, train_dict = get_files('al_file/train.csv')
    np.random.shuffle(train_files)
    train_all_files = train_files[:]
    test_files, _ = get_files('al_file/val_7k.csv')
    np.random.shuffle(test_files)

    cls_count(args, y_pred=np.array([item['label'] for item in train_files]))

    id_cls = args.id_cls
    ood_cls = args.ood_cls
    args.num_class = len(id_cls) 

    train_id_files = [item for item in train_files if item['label'] in id_cls]
    train_ood_files = [item for item in train_files if item['label'] in ood_cls]
    print(f"Train ID size: {len(train_id_files)}, Train OOD size: {len(train_ood_files)}")

    test_id_files = [item for item in test_files if item['label'] in id_cls]
    test_ood_files = [item for item in test_files if item['label'] in ood_cls]
    print(f"Test ID size: {len(test_id_files)}, Test OOD size: {len(test_ood_files)}")

    train_files = copy.deepcopy(train_id_files) + copy.deepcopy(train_ood_files)

    val_id_files = copy.deepcopy(test_id_files)[:]
    for item in val_id_files: 
        item['label'] = id_cls.index(item['label'])

    print(f"正在读取冷启动初筛种子库: {args.init_csv}")
    initial_data = pd.read_csv(args.init_csv)
    initial_names = initial_data.iloc[:, 0].to_numpy()
    
    initial_basenames = set([os.path.basename(str(name)) for name in initial_names])
    
    initial_labeled = [item for item in train_files if os.path.basename(str(item['img'])) in initial_basenames]
    labeled_ID = [item for item in train_id_files if os.path.basename(str(item['img'])) in initial_basenames]
    
    candidates = [item for item in train_files if os.path.basename(str(item['img'])) not in initial_basenames]
    candidates = copy.deepcopy(candidates)[:]
    print('initial labeled size: ', len(labeled_ID), 'candidates size: ', len(candidates))
    
    labeled_dataset = Tumor_dataset_val_cls(args, files=initial_labeled)
    val_dataset = Tumor_dataset_val_cls(args, files=val_id_files)
    
    labeled_loader = get_loader(args, labeled_dataset, shuffle=True)
    val_loader = get_loader(args, val_dataset, shuffle=False)

    labeled_ID = copy.deepcopy(labeled_ID)
    for item in labeled_ID: 
        item['label'] = id_cls.index(item['label'])
        
    cls_count(args, y_pred=np.array([item['label'] for item in labeled_ID]))
    labeled_loader_train = get_loader(args, Tumor_dataset(args, files=labeled_ID), shuffle=True)

    query_times = args.query_times
    
    precision_list = []                  
    candidate_precision_history = []     
    test_acc_history = []                
    f1_history = []                      
    recall_history = []                  
    precision_history = []               
    
    for i in range(query_times):
        if i == 0:
            model = BMC_Vision_FT_Lit(pretrain=True, num_class=args.num_class, args=args)
            # 🎯 修改点：模型存放路径前缀更换为 CRC
            save_dir = 'lightning_logs/CRC-BMC-ST-0-L/'
            train_labeled(args, model, labeled_loader_train, val_loader, save_dir=save_dir)
            
            version_dirs = glob.glob(save_dir + 'lightning_logs/version_*/')
            if version_dirs:
                version_dirs.sort(key=lambda x: int(x.split('version_')[-1].strip('/')))
                latest_version_dir = version_dirs[-1]
                checkpoint_dir = latest_version_dir + 'checkpoints/'
                checkpoint_files = os.listdir(checkpoint_dir)
                if checkpoint_files:
                    ckpt_path = checkpoint_dir + checkpoint_files[0]
                    print(f"找到checkpoint: {ckpt_path}")
                else:
                    print("错误：checkpoint目录为空")
                    sys.exit(1)
            else:
                print(f"错误：找不到version目录")
                sys.exit(1)

            model = BMC_Vision_FT_Lit.load_from_checkpoint(checkpoint_path=ckpt_path, pretrain=True,
                                                           num_class=len(id_cls), args=args)
                                                           
            global_selection_loader = get_loader(args, Tumor_dataset_val_cls(args, candidates), shuffle=False, batch_size=256)
            
            pseudo_loader, metrics_dict = generate_pseudo_labels(args, model, labeled_loader, global_selection_loader, train_dict, id_cls)
            # 🎯 修改点：模型存放路径前缀更换为 CRC
            save_dir = 'lightning_logs/CRC-BMC-ST-0-ST/'
            self_training(args, model, pseudo_loader, val_loader, save_dir)
            labeled_data_all = copy.deepcopy(initial_labeled)
            
            cur_precision = len(labeled_ID) / args.query_num if args.query_num > 0 else 0
            precision_list.append(cur_precision)
        else:
            # 🎯 修改点：模型存放路径前缀更换为 CRC
            save_dir = f'lightning_logs/CRC-BMC-ST-{i - 1}-ST/'
            version_dirs = glob.glob(save_dir + 'lightning_logs/version_*/')
            if version_dirs:
                version_dirs.sort(key=lambda x: int(x.split('version_')[-1].strip('/')))
                latest_version_dir = version_dirs[-1]
                checkpoint_dir = latest_version_dir + 'checkpoints/'
                checkpoint_files = os.listdir(checkpoint_dir)
                if checkpoint_files:
                    ckpt_path = checkpoint_dir + checkpoint_files[0]
                    print(f"找到checkpoint: {ckpt_path}")
                else:
                    print("错误：checkpoint目录为空")
                    sys.exit(1)
            else:
                print(f"错误：找不到version目录")
                sys.exit(1)

            model = BMC_Vision_FT_Lit.load_from_checkpoint(checkpoint_path=ckpt_path, pretrain=True,
                                                           num_class=len(id_cls), args=args)
            # 🎯 修改点：保存下一轮查询的 CSV 路径
            save_csv = f'al_file/BMC_query{i}_labeled.csv'
            
            global_selection_loader = get_loader(args, Tumor_dataset_val_cls(args, candidates), shuffle=False, batch_size=256)
            cur_precision = sample_selection(args, model, labeled_loader, global_selection_loader, query_num=args.query_num,
                                             save_dir=save_csv, train_dict=train_dict, id_cls=id_cls, train_id_files=train_id_files)
            # 🎯 修改点：模型存放路径前缀更换为 CRC
            save_dir = f'lightning_logs/CRC-BMC-ST-{i}-L/'
            labeled_data, _ = get_files(save_csv)
            labeled_data_all += labeled_data
            
            labeled_data_ID = [item for item in labeled_data if item['label'] in id_cls]
            labeled_data_ID = copy.deepcopy(labeled_data_ID)
            for item in labeled_data_ID:
                item['label'] = id_cls.index(item['label'])
            print(f"当前轮次新增选入ID样本数: {len(labeled_data_ID)}")
            
            labeled_loader_ID = get_loader(args, Tumor_dataset(args, labeled_data_ID))
            train_labeled(args, model, labeled_loader_ID, val_loader, save_dir)
            
            version_dirs = glob.glob(save_dir + 'lightning_logs/version_*/')
            if version_dirs:
                version_dirs.sort(key=lambda x: int(x.split('version_')[-1].strip('/')))
                latest_version_dir = version_dirs[-1]
                checkpoint_dir = latest_version_dir + 'checkpoints/'
                checkpoint_files = os.listdir(checkpoint_dir)
                if checkpoint_files:
                    ckpt_path = checkpoint_dir + checkpoint_files[0]
                    print(f"找到checkpoint: {ckpt_path}")
                else:
                    print("错误：checkpoint目录为空")
                    sys.exit(1)
            else:
                print(f"错误：找不到version目录")
                sys.exit(1)

            labeled_names_all = [item['img'] for item in labeled_data_all]
            labeled_basenames_all = set([os.path.basename(str(n)) for n in labeled_names_all])
            candidates = [item for item in train_files if os.path.basename(str(item['img'])) not in labeled_basenames_all]
            print('剩余未标注池 size: ', len(candidates))
            
            model = BMC_Vision_FT_Lit.load_from_checkpoint(checkpoint_path=ckpt_path, pretrain=True,
                                                           num_class=len(id_cls), args=args)
            labeled_loader = get_loader(args, Tumor_dataset_val_cls(args, labeled_data_all))
            global_selection_loader_for_pseudo = get_loader(args, Tumor_dataset_val_cls(args, candidates), shuffle=False, batch_size=256)
            
            pseudo_loader, metrics_dict = generate_pseudo_labels(args, model, labeled_loader, global_selection_loader_for_pseudo, train_dict, id_cls)
            # 🎯 修改点：模型存放路径前缀更换为 CRC
            save_dir = f'lightning_logs/CRC-BMC-ST-{i}-ST/'
            self_training(args, model, pseudo_loader, val_loader, save_dir)
            precision_list.append(cur_precision)
            
        candidate_precision_history.append(metrics_dict['candidate_precision'])
        test_acc_history.append(metrics_dict['test_acc'])
        f1_history.append(metrics_dict['f1'])
        recall_history.append(metrics_dict['recall'])
        precision_history.append(metrics_dict['precision'])

        print(f"🔄 截止当前轮次的精准度战报统计: {precision_list}")
        
    # 🎯 修改点：所有统计数据的落盘文件夹统一改为 log1_CRC
    os.makedirs('log1_CRC', exist_ok=True)
    np.savetxt('log1_CRC/query_precision.txt', np.array(precision_list), delimiter=',')
    np.savetxt('log1_CRC/candidate_precision.txt', np.array(candidate_precision_history), delimiter=',')
    np.savetxt('log1_CRC/test_accuracy.txt', np.array(test_acc_history), delimiter=',')
    np.savetxt('log1_CRC/f1_score.txt', np.array(f1_history), delimiter=',')
    np.savetxt('log1_CRC/recall.txt', np.array(recall_history), delimiter=',')
    np.savetxt('log1_CRC/precision.txt', np.array(precision_history), delimiter=',')
    
    summary_df = pd.DataFrame({
        'Round': [f"Round_{idx}" for idx in range(query_times)],
        'Query_Precision_QP': precision_list,
        'Candidate_Pool_Precision': candidate_precision_history,
        'Test_Accuracy': test_acc_history,
        'F1_Score_Macro': f1_history,
        'Recall_Macro': recall_history,
        'Precision_Macro': precision_history
    })
    summary_df.to_csv('log1_CRC/all_metrics_summary.csv', index=False)
    
    # 🎯 修改点：最终保存名字换成 CRC_Ours_precision.txt
    np.savetxt('log1_CRC/CRC_Ours_precision.txt', np.array(precision_list), delimiter=',')
    print("✅ 全程实验结束，所有精细化多分辨率及分类指标已完美落盘至 log1_CRC/ 目录与 all_metrics_summary.csv 账本中！")