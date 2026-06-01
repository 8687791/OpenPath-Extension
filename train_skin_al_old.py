import os
import sys

# 配置国内镜像源与关闭SSL验证（必须放在最开头）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

from dataset.alb_dataset2 import Tumor_dataset, Tumor_dataset_val_cls, get_loader
import argparse
import torch
import numpy as np
import pandas as pd
import random
from sklearn import metrics
from sklearn.cluster import KMeans
from networks.pl_models import BMC_Vision_FT_Lit
import torch.nn.functional as F
import logging
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import glob
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
    parser = argparse.ArgumentParser(description="SkinTissue Active Learning Core Pipeline")
    # 💡 核心修改一：大盘分类完全切入皮肤 10 分类
    parser.add_argument("--num_class", type=int, default=10, help="Skin disease total classes")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--crop_size", default=224)
    parser.add_argument("--gpu", nargs="+", type=int, default=[0])
    parser.add_argument("--batch_size", type=int, default=128, help="Train batch size")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--self_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query_num", type=int, default=50)
    parser.add_argument("--query_times", type=int, default=5)
    
    # 💡 核心修改二：动态注册之前遗漏的两个 CSV 参数，彻底解决 usage 报错
    parser.add_argument("--train_csv", type=str, default='al_file_skin/train.csv', help="Path to train csv")
    parser.add_argument("--val_csv", type=str, default='al_file_skin/val_7k.csv', help="Path to validation csv")
    parser.add_argument("--init_csv", type=str, default='al_file_skin/clip_query_round_1.csv')
    
    # 💡 核心修改三：日志保存路径与大模型微调 log 默认死锁皮肤车间
    parser.add_argument("--log", type=str, default='log_skin/train_sup_openpath_clip.txt')
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--save_dir", type=str, default='lightning_logs_skin/OpenPath_Vanilla_CLIP/')
    
    # 💡 核心修改四：锁定皮肤消融实验已知类阵营 [1, 4] (1=Melanoma, 4=Nevi)
    parser.add_argument("--id_cls", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--ood_cls", nargs="+", type=int, default=[0, 2, 3, 5, 6, 7, 8, 9])
    
    # 💡 提示：为了给聚类算法留足样本池缓冲，建议在运行命令中将此值由 1e-3 适当调大（如 0.05）
    parser.add_argument("--id_ratio", type=float, default=1e-3)
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
    # ⚡️ 安全防护补丁 1：如果经过过滤放行的候选池样本总数少于预期的聚类类簇数，执行自适应优雅降维
    if embeds.shape[0] < n:
        print(f"⚠️ [防爆警报] 当前过滤后候选池的真实样本数({embeds.shape[0]})少于预设查询数({n})！")
        print(f"   └─ 已启动自适应兜底，强行将 K-Means 聚类数下调至 {embeds.shape[0]}，全量收纳剩余黄金样本。")
        n = embeds.shape[0]
        
    cluster_learner = KMeans(n_clusters=n, init='k-means++', n_init='auto', random_state=42)
    cluster_learner.fit(embeds)
    cluster_idxs = cluster_learner.predict(embeds)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embeds - centers) ** 2
    dis = dis.sum(axis=1)
    q_idx = np.array([np.arange(embeds.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n)])
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
    print(f"🎯 内部筛选性能评价: Accuracy: {test_accuracy:.4f}, Macro-F1: {f1:.4f}, Precision: {p:.4f}, Recall: {r:.4f}")
    return test_accuracy, f1, r, p 

def cls_count(args, y_pred):
    count = [0] * args.num_class
    for item in y_pred:
        count[item] += 1
    print("当前各病种实体真实计数分布:", count)
    return count

def train_labeled(args, model, labeled_loader, val_loader, save_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        filename='best-model-{epoch:02d}-{val_acc:.3f}',
        save_last=False,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs, 
        precision='16-mixed', 
        check_val_every_n_epoch=args.epochs, 
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger(save_dir=save_dir)
    )
    trainer.fit(model=model, train_dataloaders=labeled_loader, val_dataloaders=val_loader)

def self_training(args, model, psuedo_loader, val_loader, save_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        filename='best-model-{epoch:02d}-{val_acc:.3f}',
        save_last=False,
    )
    trainer = L.Trainer(
        max_epochs=args.self_epochs, 
        precision='16-mixed', 
        check_val_every_n_epoch=args.self_epochs, 
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger(save_dir=save_dir)
    )
    trainer.fit(model=model, train_dataloaders=psuedo_loader, val_dataloaders=val_loader)

@torch.no_grad()
def generate_pseudo_labels(args, model, labeled_loader, selection_loader, train_dict):
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

    for counter, cls_id in enumerate(args.id_cls):
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
    indices = indices[:int(args.id_ratio * len(indices))]

    candidates_names = unlabeled_names[indices]
    candidates_probs = unlabeled_probs[indices]
    candidates_labels = [train_dict[item] for item in candidates_names]
    
    count = 0
    for item in candidates_labels:
        if int(item) in args.id_cls:
            count += 1
    print('ID/OOD伪标签初选池纯度 (Candidates Precision):', count / len(candidates_labels))

    candidates_preds = candidates_probs.argmax(1).cpu().numpy()
    candidates_labels_re = candidates_labels.copy()
    for i in range(len(candidates_labels_re)):
        if candidates_labels_re[i] in args.id_cls:
            candidates_labels_re[i] = args.id_cls.index(candidates_labels_re[i])
        else:
            candidates_labels_re[i] = len(args.id_cls)
    cal_acc(candidates_preds, np.array(candidates_labels_re))

    pseudo_files = [{'img': name, 'label': label} for (name, label) in zip(candidates_names, candidates_preds)]
    print('生成的伪标签队列总体积 (Pseudo labels size): ', len(pseudo_files))
    return get_loader(args, Tumor_dataset(args, files=pseudo_files), shuffle=False, batch_size=256)

@torch.no_grad()
def sample_selection(args, model, labeled_loader, selection_loader, query_num, save_dir, train_dict, train_id_files):
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

    for counter, cls_id in enumerate(args.id_cls):
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
    distances, unlabeled_names, unlabeled_features = distances, np.array(unlabeled_names), unlabeled_features.cpu().numpy()

    distances_min = np.array(distances).min(axis=1)
    indices = np.argsort(distances_min)
    indices = indices[:int(args.id_ratio * len(indices))]

    candidates_features = unlabeled_features[indices]
    candidates_names = unlabeled_names[indices]
    candidates_probs = unlabeled_probs[indices]
    candidates_labels = [train_dict[item] for item in candidates_names]
    
    count = 0
    for item in candidates_labels:
        if int(item) in args.id_cls:
            count += 1
    print('ID/OOD主动学习精选池纯度:', count / len(candidates_labels))

    candidates_preds = candidates_probs.argmax(1).cpu().numpy()
    candidates_labels_re = candidates_labels.copy()
    for i in range(len(candidates_labels_re)):
        if candidates_labels_re[i] in args.id_cls:
            candidates_labels_re[i] = args.id_cls.index(candidates_labels_re[i])
        else:
            candidates_labels_re[i] = len(args.id_cls)
    cal_acc(candidates_preds, np.array(candidates_labels_re))

    # 引用防爆改进后的聚类器
    cluster_idx = kmean_cluster(embeds=candidates_features, n=query_num)
    selected_names = np.array(candidates_names)[cluster_idx]
    selected_labels = [train_dict[item] for item in selected_names]
    
    data_df = pd.DataFrame()
    data_df['img'] = selected_names
    data_df['cls_label'] = np.array(selected_labels)
    
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    data_df.to_csv(save_dir, index=False)

    count = 0
    for name in selected_names:
        if name in [item['img'] for item in train_id_files]:
            count += 1
            
    # ⚡️ 安全防护补丁 2：将固定分母修改为动态实际选中长度，防止硬截断状态下出现除零错误
    current_query_precision = count / max(1, len(selected_names))
    print('💡 本轮人在回路主动查询样本绝对精度 (Query Precision): ', current_query_precision)
    return current_query_precision

if __name__ == "__main__":
    args = get_arguments()
    seed_torch(args.seed)
    torch.cuda.set_device(args.gpu[0])

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    l = logging.getLogger(__name__)
    fileHandler = logging.FileHandler(args.log, mode='a')
    l.setLevel(logging.INFO)
    l.addHandler(fileHandler)

    # 💡 核心对齐：全面绑定动态注册的皮肤大账本
    print(f"📂 正在解析皮肤全量池大账本: {args.train_csv}...")
    train_files, train_dict = get_files(args.train_csv)
    np.random.shuffle(train_files)
    
    test_files, _ = get_files(args.val_csv)
    np.random.shuffle(test_files)

    print("--- 皮肤训练大池全盘自然病种普查 ---")
    cls_count(args, y_pred=np.array([item['label'] for item in train_files]))

    id_cls = args.id_cls
    ood_cls = args.ood_cls
    args.num_class = len(id_cls)

    train_id_files = [item for item in train_files if item['label'] in id_cls]
    train_ood_files = [item for item in train_files if item['label'] in ood_cls]
    print(f"📊 数据就位：真目标(ID)共 {len(train_id_files)} 张，外部未知干扰(OOD)共 {len(train_ood_files)} 张。")

    test_id_files = [item for item in test_files if item['label'] in id_cls]
    test_ood_files = [item for item in test_files if item['label'] in ood_cls]

    train_files = copy.deepcopy(train_id_files) + copy.deepcopy(train_ood_files)

    val_id_files = copy.deepcopy(test_id_files)[:]
    for item in val_id_files: 
        item['label'] = id_cls.index(item['label'])

    print(f"📂 正在读取第一阶段冷启动开辟的黄金启动盘: {args.init_csv}...")
    initial_data = pd.read_csv(args.init_csv)
    initial_names = initial_data.iloc[:, 0].to_numpy()
    initial_basenames = set([os.path.basename(str(name)) for name in initial_names])
    
    initial_labeled = [item for item in train_files if os.path.basename(str(item['img'])) in initial_basenames]
    labeled_ID = [item for item in train_id_files if os.path.basename(str(item['img'])) in initial_basenames]
    
    candidates = [item for item in train_files if os.path.basename(str(item['img'])) not in initial_basenames]
    candidates = copy.deepcopy(candidates)[:]
    print('💡 初始化有监督集基数: ', len(labeled_ID), '剩余待探索候选集基数: ', len(candidates))
    
    labeled_dataset = Tumor_dataset_val_cls(args, files=initial_labeled)
    selection_dataset = Tumor_dataset_val_cls(args, files=candidates)
    val_dataset = Tumor_dataset_val_cls(args, files=val_id_files)
    
    labeled_loader = get_loader(args, labeled_dataset, shuffle=True)
    selection_loader = get_loader(args, selection_dataset, shuffle=False)
    val_loader = get_loader(args, val_dataset, shuffle=False)

    labeled_ID = copy.deepcopy(labeled_ID)
    for item in labeled_ID: 
        item['label'] = id_cls.index(item['label'])
        
    print("--- 初始冷启动已标注集合内部各品类分布 ---")
    cls_count(args, y_pred=np.array([item['label'] for item in labeled_ID]))
    labeled_loader_train = get_loader(args, Tumor_dataset(args, files=labeled_ID), shuffle=True)

    query_times = args.query_times
    precision_list = []
    
    for i in range(query_times):
        if i == 0:
            model = BMC_Vision_FT_Lit(pretrain=True, num_class=args.num_class, args=args)
            save_dir = os.path.join(args.save_dir, 'Skin-BMC-ST-0-L/')
            train_labeled(args, model, labeled_loader_train, val_loader, save_dir=save_dir)
            
            version_dirs = glob.glob(os.path.join(save_dir, 'lightning_logs/version_*/'))
            if version_dirs:
                version_dirs.sort(key=lambda x: int(x.split('version_')[-1].strip('/')))
                latest_version_dir = version_dirs[-1]
                checkpoint_dir = os.path.join(latest_version_dir, 'checkpoints/')
                checkpoint_files = os.listdir(checkpoint_dir)
                if checkpoint_files:
                    ckpt_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                    print(f"✅ 成功锚定第 0 轮有监督最优 Checkpoint: {ckpt_path}")
                else:
                    print("错误：checkpoint目录为空"); sys.exit(1)
            else:
                print("错误：找不到version目录"); sys.exit(1)

            model = BMC_Vision_FT_Lit.load_from_checkpoint(checkpoint_path=ckpt_path, pretrain=True, num_class=len(id_cls), args=args)
            pseudo_loader = generate_pseudo_labels(args, model, labeled_loader, selection_loader, train_dict)
            save_dir = os.path.join(args.save_dir, 'Skin-BMC-ST-0-ST/')
            self_training(args, model, pseudo_loader, val_loader, save_dir)
            labeled_data_all = copy.deepcopy(initial_labeled)
            
            # 💡 自适应对齐分母
            precision_list.append(len(labeled_ID) / len(initial_names))
        else:
            save_dir = os.path.join(args.save_dir, f'Skin-BMC-ST-{i - 1}-ST/')
            version_dirs = glob.glob(os.path.join(save_dir, 'lightning_logs/version_*/'))
            if version_dirs:
                version_dirs.sort(key=lambda x: int(x.split('version_')[-1].strip('/')))
                latest_version_dir = version_dirs[-1]
                checkpoint_dir = os.path.join(latest_version_dir, 'checkpoints/')
                checkpoint_files = os.listdir(checkpoint_dir)
                if checkpoint_files:
                    ckpt_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                    print(f"✅ 成功锚定第 {i} 轮自训练最优 Checkpoint: {ckpt_path}")
                else:
                    print("错误：checkpoint目录为空"); sys.exit(1)
            else:
                print("错误：找不到version目录"); sys.exit(1)

            model = BMC_Vision_FT_Lit.load_from_checkpoint(checkpoint_path=ckpt_path, pretrain=True, num_class=len(id_cls), args=args)
            save_csv = f'al_file_skin/BMC_query{i}_labeled.csv'
            cur_precision = sample_selection(args, model, labeled_loader, selection_loader, query_num=args.query_num, save_dir=save_csv, train_dict=train_dict, train_id_files=train_id_files)
            
            save_dir = os.path.join(args.save_dir, f'Skin-BMC-ST-{i}-L/')
            labeled_data, _ = get_files(save_csv)
            labeled_data_all += labeled_data
            labeled_data_ID = [item for item in labeled_data if item['label'] in id_cls]
            labeled_data_ID = copy.deepcopy(labeled_data_ID)
            for item in labeled_data_ID:
                item['label'] = id_cls.index(item['label'])
            print(f"第 {i} 轮人在回路实际采纳并标记的真实 ID 增量样本数: {len(labeled_data_ID)}")
            
            labeled_loader_ID = get_loader(args, Tumor_dataset(args, labeled_data_ID))
            train_labeled(args, model, labeled_loader_ID, val_loader, save_dir)
            
            version_dirs = glob.glob(os.path.join(save_dir, 'lightning_logs/version_*/'))
            if version_dirs:
                version_dirs.sort(key=lambda x: int(x.split('version_')[-1].strip('/')))
                latest_version_dir = version_dirs[-1]
                checkpoint_dir = os.path.join(latest_version_dir, 'checkpoints/')
                checkpoint_files = os.listdir(checkpoint_dir)
                if checkpoint_files:
                    ckpt_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                    print(f"✅ 成功锚定第 {i} 轮有监督增量最优 Checkpoint: {ckpt_path}")
                else:
                    print("错误：checkpoint目录为空"); sys.exit(1)
            else:
                print("错误：找不到version目录"); sys.exit(1)

            labeled_names_all = [item['img'] for item in labeled_data_all]
            candidates = [item for item in train_files if item['img'] not in copy.deepcopy(labeled_names_all)]
            print('当前池内剩余待探索候选集大小: ', len(candidates))
            
            model = BMC_Vision_FT_Lit.load_from_checkpoint(checkpoint_path=ckpt_path, pretrain=True, num_class=len(id_cls), args=args)
            labeled_loader = get_loader(args, Tumor_dataset_val_cls(args, labeled_data_all))
            selection_loader = get_loader(args, Tumor_dataset_val_cls(args, candidates))
            pseudo_loader = generate_pseudo_labels(args, model, labeled_loader, selection_loader, train_dict)
            save_dir = os.path.join(args.save_dir, f'Skin-BMC-ST-{i}-ST/')
            self_training(args, model, pseudo_loader, val_loader, save_dir)
            precision_list.append(cur_precision)
            
        print("💡 当前全周期人在回路大循环主动学习精确度轨迹线:", precision_list)
        
    os.makedirs('log_skin', exist_ok=True)
    np.savetxt('log_skin/OpenPath_Vanilla_Precision.txt', np.array(precision_list), delimiter=',')
    print("✅ 实验圆满收盘！数据已固化落盘至 log_skin/OpenPath_Vanilla_Precision.txt")