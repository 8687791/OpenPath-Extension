import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 实验配置与账本对齐
# ==========================================
CACHE_PATH = "al_file_skin/vlm_id_candidates.pkl"
TRAIN_CSV = "al_file_skin/train.csv"
INIT_NUM = 50
GOLD_ID_CLASSES = [1, 4]  # 1=Melanoma, 4=Nevi

if not os.path.exists(CACHE_PATH):
    raise FileNotFoundError(f"❌ 找不到初筛固化文件 {CACHE_PATH}，请先运行第一步代码大模型保存！")

# 1. 载入固化的全量候选池与真值表
with open(CACHE_PATH, "rb") as f:
    cache_data = pickle.load(f)
names_id_vlm = np.array(cache_data["names"])
embeds_id = cache_data["embeds"]

train_df = pd.read_csv(TRAIN_CSV)
truth_dict = {os.path.basename(row.iloc[0]): int(row.iloc[1]) for _, row in train_df.iterrows()}

print(f"📊 成功载入固化候选池。当前待处理有效样本数: {len(names_id_vlm)}\n")

# ==================================================
# 🧪 算法 A: 经典 K-Means++ 采样
# ==================================================
def run_kmeans_plus_plus(names, embeds):
    kmeans = KMeans(n_clusters=INIT_NUM, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(embeds)
    selected = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(embeds - center, axis=1)
        selected.append(names[np.argmin(distances)])
    return selected

# ==================================================
# 🧪 算法 B: 密度自适应核心采样 (Density + FPS)
# ==================================================
def run_density_core_sampling(names, embeds, k_neighbors=5, noise_percentile=15):
    n_samples = embeds.shape[0]
    dot_product = np.dot(embeds, embeds.T)
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    dist_matrix = np.sqrt(np.maximum(norms**2 + norms.T**2 - 2 * dot_product, 0.0))
    
    knn_dists = np.sort(dist_matrix, axis=1)[:, 1:k_neighbors + 1].mean(axis=1)
    threshold = np.percentile(knn_dists, 100 - noise_percentile)
    valid_indices = np.where(knn_dists <= threshold)[0]
    if len(valid_indices) < INIT_NUM: valid_indices = np.arange(n_samples)
    
    selected_indices = [valid_indices[np.argmin(knn_dists[valid_indices])]]
    min_dists = dist_matrix[selected_indices[0], :]
    
    while len(selected_indices) < INIT_NUM:
        candidate_indices = [i for i in valid_indices if i not in selected_indices]
        best_next = candidate_indices[np.argmax(min_dists[candidate_indices])]
        selected_indices.append(best_next)
        min_dists = np.minimum(min_dists, dist_matrix[best_next, :])
    return names[selected_indices].tolist()

# ==================================================
# 🧪 算法 C: 图论中心度与 NMS 抑制采样 (Graph Centrality)
# ==================================================
def run_graph_centrality_nms(names, embeds, sim_thresh=0.75, penalty=0.5):
    # 计算余弦相似度图
    sim_matrix = cosine_similarity(embeds)
    # 斩断弱连接（剔除边缘噪声的连边）
    sim_matrix[sim_matrix < sim_thresh] = 0 
    # 计算每个节点的度中心性（即处于多稠密的区域）
    centrality = sim_matrix.sum(axis=1) 
    
    selected_indices = []
    valid_mask = np.ones(len(names), dtype=bool)
    
    for _ in range(INIT_NUM):
        masked_centrality = centrality * valid_mask
        if masked_centrality.max() <= 0:
            best_idx = np.random.choice(np.where(valid_mask)[0]) # 容错
        else:
            best_idx = np.argmax(masked_centrality)
            
        selected_indices.append(best_idx)
        valid_mask[best_idx] = False
        
        # NMS 抑制：降低与刚被选中节点高度相似的周边节点的中心度，强迫多样性
        centrality -= sim_matrix[best_idx] * penalty
        centrality = np.maximum(centrality, 0)
        
    return names[selected_indices].tolist()

# ==================================================
# 🧪 算法 D: K-Center Greedy 核心集 (经典基线，纯最远点)
# ==================================================
def run_k_center_greedy(names, embeds):
    # 寻找几何中心点作为起点
    centroid = embeds.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(embeds - centroid, axis=1)
    first_idx = np.argmin(dists)
    
    selected_indices = [first_idx]
    min_dists = np.linalg.norm(embeds - embeds[first_idx:first_idx+1], axis=1)
    
    while len(selected_indices) < INIT_NUM:
        # 纯粹寻找距离当前集合最远的点（极易采到 OOD 噪声）
        best_next = np.argmax(min_dists)
        selected_indices.append(best_next)
        new_dists = np.linalg.norm(embeds - embeds[best_next:best_next+1], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        
    return names[selected_indices].tolist()


# ==================================================
# 🚀 计算精度并开榜
# ==================================================
def calc_qp(selected_list):
    id_count = sum([1 for n in selected_list if truth_dict.get(os.path.basename(n)) in GOLD_ID_CLASSES])
    return (id_count / len(selected_list)) * 100

print(f"🏆 ================= 采样算法横向消融大对标 (50张预算) =================")
print(f" ├─ 方案 A: 经典 K-Means++ (易受稀疏空间边缘影响)       QP: {calc_qp(run_kmeans_plus_plus(names_id_vlm, embeds_id)):.2f}%")
print(f" ├─ 方案 D: K-Center Greedy (顶会基线/反面教材，极易吸噪)  QP: {calc_qp(run_k_center_greedy(names_id_vlm, embeds_id)):.2f}%")
print(f" ├─ 方案 B: Density-Core FPS (局部密度剔除+最远点)       QP: {calc_qp(run_density_core_sampling(names_id_vlm, embeds_id)):.2f}%")
print(f" └─ 方案 C: Graph Centrality NMS (图论流形挖掘+侧向抑制)  QP: {calc_qp(run_graph_centrality_nms(names_id_vlm, embeds_id)):.2f}%")
print(f"=======================================================================")