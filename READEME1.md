这是一份更新后的 README 文本。已在 **“第二步”之前** 完美嵌入了 Miniconda 的一键安装与环境初始化步骤，并保持了其余所有工程细节、避坑参数和后续实验启动命令完全不变：

---

# 🛠️ OpenPath + SAM2 多模态主动学习环境从零搭建全流程

### 第一步：彻底清理旧环境残留（防止路径冲突）

我们在实验中发现，系统原生 `virtualenv` 和 `Conda` 叠加激活会导致 `PATH` 错乱，引发找不到 `torch` 的奇葩 Bug。在配新环境前，先在终端执行以下命令彻底退出所有叠加态：

```bash
# 连续执行，直到终端最左侧没有任何括号（如 (base) 或 (openpath)）为止
conda deactivate
deactivate

```

### 第二步：安装并初始化 Miniconda 基础环境

若系统内尚未配置 Conda 环境，请直接下载并安装 Python 3.10 版本的 Miniconda：

```bash
# 1. 下载 Miniconda 安装包
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh

# 2. 运行安装程序（一路按空格跳过协议，输入 yes 确认并接受默认路径）
bash Miniconda3-py310_23.11.0-2-Linux-x86_64.sh

# 3. 激活 Conda 环境变量（执行后命令行最左侧将显示 (base)）
source ~/.bashrc

```

### 第三步：创建并激活全新的 Conda 虚拟环境

基于你模型中使用的 Python 3.10 版本，创建一个干净、独立的虚拟环境：

```bash
# 1. 创建名为 openpath 的 Python 3.10 环境
conda create -n openpath python=3.10 -y

# 2. 干净地激活该环境
conda activate openpath

```

### 第四步：系统盘“防爆”预处理（将缓存锁死在数据盘）

由于云容器的根目录（系统盘 `/`）通常只有 30GB，而大模型权重和缓存极其庞大。在安装任何包之前，必须把缓存目录重定向到空间充足的数据盘 `/root/gpufree-data`：

```bash
# 1. 在大容量数据盘创建专用的缓存目录
mkdir -p /root/gpufree-data/hf_cache

# 2. 强行在系统盘原有缓存位置建立软链接（快捷方式）
rm -rf /root/.cache/huggingface
ln -s /root/gpufree-data/hf_cache /root/.cache/huggingface

```

### 第五步：准备并安装 `requirements1.txt` 基础依赖

在你的项目根目录下新建一个名为 `requirements1.txt` 的文件，将我们对齐好版本的依赖项粘贴进去：

```text
# 基础深度学习与核心训练框架
torch>=2.0.1
torchvision
lightning>=2.1.0

# 跨模态大模型与文本分词依赖 (锁定版本以完美向下兼容 PyTorch 2.0.1，防止系统自动禁用 Torch)
transformers>=4.36.0,<4.40.0
huggingface-hub>=0.19.0,<1.0
sentencepiece
tiktoken

# 视觉特征提取与张量高级重排 (InternVL 与 BiomedCLIP 强依赖)
timm
einops

# 科学计算与医学图像数据增强
albumentations
pandas
numpy
tqdm
scikit-learn

```

保存文件后，在终端执行批量安装（国内清华镜像源加速）：

```bash
pip install -r requirements1.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

```

### 第六步：源码编译安装 SAM2 核心引擎（终极抗假死命令）

**【切勿直接用 pip 默认命令安装】**。为了防止 `pip` 默认的隔离构建机制在系统盘 `/tmp` 目录下创建几 GB 的临时 PyTorch 沙盒导致系统瞬间憋死，必须加上 `--no-build-isolation` 参数，强制利用我们刚刚在 `openpath` 环境里装好的 PyTorch 和工具进行硬件级 CUDA 算子联编：

```bash
python -m pip install git+https://github.com/facebookresearch/segment-anything-2.git --no-build-isolation

```

*注：当终端停在 `Running setup.py install for segment-anything-2 ...` 时，是在调用 NVCC 编译器编译 CUDA，会静止 3~5 分钟，请耐心等待其弹出 `Successfully installed`。*

### 第七步：环境黄金健全性验证（Sanity Check）

环境全部配置完成后，新开一个终端，运行以下命令验证 GPU 算力通道与大模型组件是否全线贯通：

```bash
python -c "
import torch
print('1. CUDA 是否可用:', torch.cuda.is_available())
print('2. 当前显卡型号:', torch.cuda.get_device_name(0))

try:
    from sam2.build_sam import build_sam2
    print('3. SAM2 核心库链接: 成功对齐！')
except Exception as e:
    print('3. SAM2 链接失败:', e)

try:
    import transformers
    print('4. Transformers 降维兼容层: 完美激活！')
except Exception as e:
    print('4. Transformers 激活失败:', e)
"

```

---

## 🚀 欢呼吧！环境已就绪，一键启动实验

当上述验证全部输出正确后，你就可以直接在项目根目录下，按照逻辑顺序无缝推进你的两阶段主动学习研究了：

1. **第一阶段：跑大模型冷启动与新病种发掘**

```bash
python vlm_generative_ood.py

```

2. **第二阶段：跑双尺度人在回路主动学习主轴大循环**

```bash
python train_multigranular_al.py

```

整套流水线从这一刻起彻底闭环，可以放心让你的 Tesla T4 显卡满载轰鸣了！