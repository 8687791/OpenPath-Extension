import os
# 配置国内镜像源（必须放在最开头）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''

import sys
import time
from dataset.alb_dataset2 import Tumor_dataset, Tumor_dataset_val_cls, get_loader
import argparse
import torch
import numpy as np
import pandas as pd
import random
from sklearn import metrics
from sklearn.cluster import KMeans
from networks.pl_models import BMC_Vision_FT_Lit
from scipy.spatial.distance import mahalanobis, euclidean, cosine
import copy
import torch.nn.functional as F
import logging
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import glob

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
    parser = argparse.ArgumentParser(description="xxxx Pytorch implementation")
    parser.add_argument("--num_class", type=int, default=9, help="Train class num")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--crop_size", default=224)
    parser.add_argument("--gpu", nargs="+", type=int, default=[0])
    parser.add_argument("--batch_size", type=int, default=128, help="Train batch size")
    parser.add_argument("--num_workers", default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--self_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--query_num", type=int, default=50)
    parser.add_argument("--query_times", type=int, default=3)
    parser.add_argument("--log", type=str, default='log/train_sup.txt')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--init_csv", type=str)
    parser.add_argument("--id_cls", nargs="+", type=int, default=[3,6,8])
    parser.add_argument("--ood_cls", nargs="+", type=int, default=[0,1,2,4,5,7])
    parser.add_argument("--id_ratio", type=float, default=1e-3)
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

def kmean_cluster(embeds, n):
    cluster_learner = KMeans(n_clusters=n, init='k-means++', n_init='auto')
    cluster_learner.fit(embeds)
    cluster_idxs = cluster_learner.predict(embeds)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embeds - centers)**2
    dis = dis.sum(axis=1)
    q_idx = np.array([np.arange(embeds.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
    return q_idx

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def cal_acc(y_pred, y_true):
    test_accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = metrics.recall_score(y_true, y_pred, average='macro')
    print(f"Test Accuracy: {test_accuracy}, f1:{f1}, precision:{p}, recall:{r}")
    return test_accuracy, f1, r, p

def cls_count(args, y_pred):
    count = [0]*args.num_class
    for item in y_pred:
        count[item] += 1
    print(count)
    return count

def train_labeled(args, model, labeled_loader, save_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc', mode='max', save_top_k=1,
        filename='best-model-{epoch:02d}-{val_acc:.3f}', save_last=False,
    )
    trainer = L.Trainer(max_epochs=args.epochs, precision='16-mixed', check_val_every_n_epoch=args.epochs,
                        callbacks=[checkpoint_callback], logger=TensorBoardLogger(save_dir=save_dir))
    trainer.fit(model=model, train_dataloaders=labeled_loader, val_dataloaders=val_loader)
    
    val_acc = float(trainer.callback_metrics.get('val_acc', 0.0)) if 'val_acc' in trainer.callback_metrics else 0.0
    val_f1 = float(trainer.callback_metrics.get('val_f1', 0.0)) if 'val_f1' in trainer.callback_metrics else 0.0
    return val_acc, val_f1

def self_training(args, model, psuedo_loader, save_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc', mode='max', save_top_k=1,
        filename='best-model-{epoch:02d}-{val_acc:.3f}', save_last=False,
    )
    trainer = L.Trainer(max_epochs=args.self_epochs, precision='16-mixed', check_val_every_n_epoch=args.self_epochs,
                        callbacks=[checkpoint_callback], logger=TensorBoardLogger(save_dir=save_dir))
    trainer.fit(model=model, train_dataloaders=psuedo_loader, val_dataloaders=val_loader)
    
    val_acc = float(trainer.callback_metrics.get('val_acc', 0.0)) if 'val_acc' in trainer.callback_metrics else 0.0
    val_f1 = float(trainer.callback_metrics.get('val_f1', 0.0)) if 'val_f1' in trainer.callback_metrics else 0.0
    return val_acc, val_f1

@torch.no_grad()
def generate_pseudo_labels(args, model, labeled_loader, selection_loader):
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

    # 🔧 修复1：防止因缺失特定 ID 类别求 mean 时产生 NaN
    class_embeds = []
    for cls_id in args.id_cls:
        mask = (labeled_labels == cls_id)
        if mask.sum() > 0:
            class_embeds.append(labeled_features[mask].mean(0).unsqueeze(0))
    
    if len(class_embeds) > 0:
        class_id_embed = torch.cat(class_embeds, dim=0)
    else:
        # 极端兜底：如果全部 ID 类都缺失，使用所有标记样本的平均特征防止崩溃
        class_id_embed = labeled_features.mean(0).unsqueeze(0)

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
    indices = indices[:int(args.id_ratio*len(indices))]

    candidates_features = unlabeled_features[indices]
    candidates_names = unlabeled_names[indices]
    candidates_probs = unlabeled_probs[indices]

    candidates_labels = [train_dict[item] for item in candidates_names]
    count = 0
    for item in candidates_labels:
        if int(item) in id_cls:
            count += 1
            
    id_ood_prec = count / len(candidates_labels) if len(candidates_labels) > 0 else 0
    print('ID/OOD candidates precision:', id_ood_prec)

    candidates_preds = candidates_probs.argmax(1).cpu().numpy()
    candidates_labels_re = candidates_labels.copy()
    for i in range(len(candidates_labels_re)):
        if candidates_labels_re[i] in id_cls:
            candidates_labels_re[i] = id_cls.index(candidates_labels_re[i])
        else:
            candidates_labels_re[i] = len(id_cls)
            
    acc, f1, r, p = cal_acc(candidates_preds, np.array(candidates_labels_re))

    pseudo_files = [{'img': name, 'label': label} for (name, label) in zip(candidates_names, candidates_preds)]
    print('pseudo labels size: ', len(pseudo_files))
    
    return get_loader(args, Tumor_dataset(args, files=pseudo_files), shuffle=False, batch_size=256), \
           {"Acc": acc, "F1": f1, "Recall": r, "Precision": p, "ID_OOD_Prec": id_ood_prec}

@torch.no_grad()
def sample_selection(args, model, labeled_loader, selection_loader, query_num, save_dir):
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

    # 🔧 修复1：防止因缺失特定 ID 类别求 mean 时产生 NaN
    class_embeds = []
    for cls_id in args.id_cls:
        mask = (labeled_labels == cls_id)
        if mask.sum() > 0:
            class_embeds.append(labeled_features[mask].mean(0).unsqueeze(0))
    
    if len(class_embeds) > 0:
        class_id_embed = torch.cat(class_embeds, dim=0)
    else:
        # 极端兜底：如果全部 ID 类都缺失，使用所有标记样本的平均特征防止崩溃
        class_id_embed = labeled_features.mean(0).unsqueeze(0)

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
    distances, unlabeled_names, unlabeled_features = distances, np.array(unlabeled_names), unlabeled_features.cpu().numpy()

    # 增加一个小修复：防止 distances 中仍然存在 NaN 导致 max(1) 报错
    distances = np.nan_to_num(distances, nan=1.0) 

    distance_probs = torch.tensor(distances).softmax(1)
    distance_entropy = -torch.sum((distance_probs.log()+1e-6)*distance_probs, dim=1)
    
    # 动态适应 class_id_embed 的维度大小，防止出现越界错误
    num_centers = class_id_embed.shape[0]
    entropies = []
    if num_centers >= 2:
        for i in range(num_centers):
            for j in range(i+1, num_centers):
                d_probs = torch.tensor(distances)[:,[i,j]].softmax(1)
                entropies.append(-torch.sum((d_probs.log()+1e-6)*d_probs, dim=1).unsqueeze(1))
        if entropies:
            distance_entropy_pairwise = torch.cat(entropies, dim=1)
            distance_entropy_pairwise, _ = distance_entropy_pairwise.max(dim=1)
            distance_entropy = distance_entropy_pairwise

    distances_min = np.array(distances).min(axis=1)
    indices = np.argsort(distances_min)
    indices = indices[:int(args.id_ratio*len(indices))]

    candidates_features = unlabeled_features[indices]
    candidates_names = unlabeled_names[indices]
    candidates_probs = unlabeled_probs[indices]

    candidates_labels = [train_dict[item] for item in candidates_names]
    count = 0
    for item in candidates_labels:
        if int(item) in id_cls:
            count += 1
            
    id_ood_prec = count / len(candidates_labels) if len(candidates_labels) > 0 else 0
    print('ID/OOD candidates precision:', id_ood_prec)

    candidates_preds = candidates_probs.argmax(1).cpu().numpy()
    candidates_labels_re = candidates_labels.copy()
    for i in range(len(candidates_labels_re)):
        if candidates_labels_re[i] in id_cls:
            candidates_labels_re[i] = id_cls.index(candidates_labels_re[i])
        else:
            candidates_labels_re[i] = len(id_cls)
            
    acc, f1, r, p = cal_acc(candidates_preds, np.array(candidates_labels_re))

    cluster_idx = kmean_cluster(embeds=candidates_features, n=query_num)
    selected_names = np.array(candidates_names)[cluster_idx]
    selected_labels = [train_dict[item] for item in selected_names]
    
    data_df = pd.DataFrame()
    data_df['img'] = selected_names
    data_df['cls_label'] = np.array(selected_labels)
    data_df.to_csv(save_dir, index=False)

    count = 0
    for name in selected_names:
        if name in [item['img'] for item in train_id_files]:
            count += 1
            
    query_prec = count / args.query_num
    print('query precision: ', query_prec)
    
    return query_prec, {"Acc": acc, "F1": f1, "Recall": r, "Precision": p, "ID_OOD_Prec": id_ood_prec, "Query_Prec": query_prec}

if __name__ == "__main__":
    start_time_all = time.time() 
    args = get_arguments()
    seed_torch(args.seed)
    torch.cuda.set_device(args.gpu[0])
    
    log_metrics = []

    os.makedirs('log', exist_ok=True)
    l = logging.getLogger(__name__)
    fileHandler = logging.FileHandler(args.log, mode='a')
    l.setLevel(logging.INFO)
    l.addHandler(fileHandler)

    train_files, train_dict = get_files('/root/gpufree-data/OpenPath-main/al_file/train.csv')
    np.random.shuffle(train_files)
    train_all_files = train_files[:]
    test_files, _ = get_files('/root/gpufree-data/OpenPath-main/al_file/val_7k.csv')
    np.random.shuffle(test_files)

    cls_count(args, y_pred=np.array([itme['label'] for itme in train_files]))

    id_cls = args.id_cls
    ood_cls = args.ood_cls
    args.num_class = len(id_cls)

    train_id_files = [item for item in train_files if item['label'] in id_cls]
    train_ood_files = [item for item in train_files if item['label'] in ood_cls]
    print(len(train_id_files), len(train_ood_files))

    test_id_files = [item for item in test_files if item['label'] in id_cls]
    test_ood_files = [item for item in test_files if item['label'] in ood_cls]
    print(len(test_id_files), len(test_ood_files))

    train_files = copy.deepcopy(train_id_files)+copy.deepcopy(train_ood_files)
    global val_loader
    val_id_files = copy.deepcopy(test_id_files)[:]
    for item in val_id_files: 
        item['label'] = id_cls.index(item['label'])

    initial_data = pd.read_csv(args.init_csv)
    if len(initial_data) > 10:
        print(f"⚠️ 从 {args.init_csv} 中随机选取 10 张图片进行训练...")
        random.seed(None)
        selected_idx = random.sample(range(len(initial_data)), 10)
        initial_data = initial_data.iloc[selected_idx].reset_index(drop=True)
    
    initial_names = initial_data.iloc[:, 0].to_numpy()
    selected_10_images = initial_names.tolist() 

    initial_basenames = set([os.path.basename(str(name)) for name in initial_names])
    initial_labeled = [item for item in train_files if os.path.basename(str(item['img'])) in initial_basenames]
    labeled_ID = [item for item in train_id_files if os.path.basename(str(item['img'])) in initial_basenames]
    
    candidates = [item for item in train_files if item['img'] not in initial_names]
    candidates = copy.deepcopy(candidates)[:]
    print('initial labeled size: ', len(labeled_ID), 'candidates size: ', len(candidates))
    
    labeled_dataset = Tumor_dataset_val_cls(args, files=initial_labeled)
    selection_dataset = Tumor_dataset_val_cls(args, files=candidates)
    val_dataset = Tumor_dataset_val_cls(args, files=val_id_files)
    
    labeled_loader = get_loader(args, labeled_dataset, shuffle=True)
    selection_loader = get_loader(args, selection_dataset, shuffle=False)
    val_loader = get_loader(args, val_dataset, shuffle=False)

    labeled_ID = copy.deepcopy(labeled_ID)
    for item in labeled_ID: 
        item['label'] = id_cls.index(item['label'])
    cls_count(args, y_pred=np.array([itme['label'] for itme in labeled_ID]))

    query_times = args.query_times
    precision_list = []
    
    for i in range(query_times):
        if i == 0:
            model = BMC_Vision_FT_Lit(pretrain=True, num_class=args.num_class, args=args)
            save_dir = 'lightning_logs/CRC100K-BMC-ST-0-L/'
            
            # 🔧 修复2：防范初始化抽样中一个 ID 数据都没有导致崩溃的情况
            if len(labeled_ID) > 0:
                labeled_loader_train = get_loader(args, Tumor_dataset(args, files=labeled_ID), shuffle=True)
                val_acc_L, val_f1_L = train_labeled(args, model, labeled_loader_train, save_dir=save_dir)
                
                version_dirs = glob.glob(save_dir + 'lightning_logs/version_*/')
                if version_dirs:
                    version_dirs.sort(key=lambda x: int(x.split('version_')[-1].strip('/')))
                    ckpt_path = version_dirs[-1] + 'checkpoints/' + os.listdir(version_dirs[-1] + 'checkpoints/')[0]
                else:
                    print(f"错误：找不到version目录")
                    sys.exit(1)
            else:
                print("⚠️ 严重警告：冷启动 10 张图片中没有任何 ID 类别！强制退出，请检查冷启动池或更换随机种子。")
                sys.exit(1)

            model = BMC_Vision_FT_Lit.load_from_checkpoint(checkpoint_path=ckpt_path, pretrain=True, num_class=len(id_cls), args=args)
            
            pseudo_loader, cur_metrics = generate_pseudo_labels(args, model, labeled_loader, selection_loader)
            cur_metrics["Val_Acc_L"] = val_acc_L
            cur_metrics["Val_F1_L"] = val_f1_L
            log_metrics.append({"Round": i, "Phase": "Generate Pseudo Labels", "Metrics": cur_metrics})
            
            save_dir = 'lightning_logs/CRC100K-BMC-ST-0-ST/'
            val_acc_ST, val_f1_ST = self_training(args, model, pseudo_loader, save_dir)
            log_metrics[-1]["Metrics"]["Val_Acc_ST"] = val_acc_ST
            log_metrics[-1]["Metrics"]["Val_F1_ST"] = val_f1_ST
            
            labeled_data_all = copy.deepcopy(initial_labeled)
            precision_list.append(len(labeled_ID) / 10) 
        else:
            save_dir = f'lightning_logs/CRC100K-BMC-ST-{i - 1}-ST/'
            version_dirs = glob.glob(save_dir + 'lightning_logs/version_*/')
            if version_dirs:
                version_dirs.sort(key=lambda x: int(x.split('version_')[-1].strip('/')))
                ckpt_path = version_dirs[-1] + 'checkpoints/' + os.listdir(version_dirs[-1] + 'checkpoints/')[0]
            else:
                print(f"错误：找不到version目录")
                sys.exit(1)

            model = BMC_Vision_FT_Lit.load_from_checkpoint(checkpoint_path=ckpt_path, pretrain=True, num_class=len(id_cls), args=args)
            save_csv = f'al_file/BMC_query{i}_labeled.csv'
            
            cur_precision, cur_metrics_1 = sample_selection(args, model, labeled_loader, selection_loader, query_num=args.query_num, save_dir=save_csv)
            log_metrics.append({"Round": i, "Phase": "Sample Selection", "Metrics": cur_metrics_1})
            
            save_dir = f'lightning_logs/CRC100K-BMC-ST-{i}-L/'
            labeled_data, _ = get_files(save_csv)
            labeled_data_all += labeled_data
            labeled_data_ID = [item for item in labeled_data if item['label'] in id_cls]
            labeled_data_ID = copy.deepcopy(labeled_data_ID)
            
            # 🔧 修复2：防范 query_prec 为 0 时 Dataloader 发生 num_samples=0 的崩溃
            if len(labeled_data_ID) > 0:
                for item in labeled_data_ID:
                    item['label'] = id_cls.index(item['label'])
                print(len(labeled_data_ID))
                labeled_loader_ID = get_loader(args, Tumor_dataset(args, labeled_data_ID))
                
                val_acc_L, val_f1_L = train_labeled(args, model, labeled_loader_ID, save_dir)
                version_dirs = glob.glob(save_dir + 'lightning_logs/version_*/')
                if version_dirs:
                    version_dirs.sort(key=lambda x: int(x.split('version_')[-1].strip('/')))
                    ckpt_path = version_dirs[-1] + 'checkpoints/' + os.listdir(version_dirs[-1] + 'checkpoints/')[0]
                else:
                    print(f"错误：找不到version目录")
                    sys.exit(1)
            else:
                print("⚠️ 警告：本轮未筛选到任何 ID 数据！跳过 L 阶段训练，沿用上一阶段模型。")
                val_acc_L, val_f1_L = 0.0, 0.0
                # ckpt_path 保持为 ST 阶段的模型，不改变

            labeled_names_all = [item['img'] for item in labeled_data_all]
            candidates = [item for item in train_files if item['img'] not in copy.deepcopy(labeled_names_all)]
            print('candidates size: ', len(candidates))
            print(len(labeled_data_all), len(candidates))
            
            model = BMC_Vision_FT_Lit.load_from_checkpoint(checkpoint_path=ckpt_path, pretrain=True, num_class=len(id_cls), args=args)
            labeled_loader = get_loader(args, Tumor_dataset_val_cls(args, labeled_data_all))
            selection_loader = get_loader(args, Tumor_dataset_val_cls(args, candidates))
            
            pseudo_loader, cur_metrics_2 = generate_pseudo_labels(args, model, labeled_loader, selection_loader)
            cur_metrics_2["Val_Acc_L"] = val_acc_L
            cur_metrics_2["Val_F1_L"] = val_f1_L
            log_metrics.append({"Round": i, "Phase": "Generate Pseudo Labels", "Metrics": cur_metrics_2})
            
            save_dir = f'lightning_logs/CRC100K-BMC-ST-{i}-ST/'
            val_acc_ST, val_f1_ST = self_training(args, model, pseudo_loader, save_dir)
            log_metrics[-1]["Metrics"]["Val_Acc_ST"] = val_acc_ST
            log_metrics[-1]["Metrics"]["Val_F1_ST"] = val_f1_ST
            
            precision_list.append(cur_precision)
        
        print(precision_list)

    np.savetxt('log/Ours_precision.txt', np.array(precision_list), delimiter=',')

    total_time = time.time() - start_time_all
    hours, rem = divmod(total_time, 3600)
    mins, secs = divmod(rem, 60)
    
    experiment_log_path = "log/training_experiment_results.log"
    
    with open(experiment_log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"🚀 实验运行时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"⏱ 训练总耗时: {int(hours):02d}h {int(mins):02d}m {secs:05.2f}s\n")
        f.write(f"⚙️ 随机种子: {args.seed} | 读取的文件: {args.init_csv}\n")
        f.write(f"🖼 抽取的冷启动 10 张图片:\n")
        for idx, img_name in enumerate(selected_10_images, 1):
            f.write(f"   {idx}. {os.path.basename(img_name)}\n")
        f.write(f"\n📊 Precision List 最终记录: {precision_list}\n")
        f.write(f"\n📈 各轮候选集 (Candidates) 指标详情:\n")
        for log_m in log_metrics:
            metrics_dict = log_m["Metrics"]
            phase = log_m["Phase"]
            
            line = (f"   [Round {log_m['Round']} - {phase}] "
                    f"ID/OOD Purity: {metrics_dict.get('ID_OOD_Prec', 0):.4f}, "
                    f"Acc: {metrics_dict['Acc']:.4f}, "
                    f"F1: {metrics_dict['F1']:.4f}, "
                    f"Prec: {metrics_dict['Precision']:.4f}, "
                    f"Recall: {metrics_dict['Recall']:.4f}")
            
            if "Query_Prec" in metrics_dict:
                line += f", Query Prec: {metrics_dict['Query_Prec']:.4f}"
                
            if "Val_Acc_L" in metrics_dict:
                line += f" | 验证集(L) Acc: {metrics_dict['Val_Acc_L']:.3f}, F1: {metrics_dict['Val_F1_L']:.3f}"
            if "Val_Acc_ST" in metrics_dict:
                line += f" | 验证集(ST) Acc: {metrics_dict['Val_Acc_ST']:.3f}, F1: {metrics_dict['Val_F1_ST']:.3f}"
                
            f.write(line + "\n")
        f.write(f"{'='*80}\n")
    
    print(f"\n✅ 训练结束！完整的实验结果已追加至: {experiment_log_path}")