import os
from huggingface_hub import snapshot_download

# 1. 配置国内镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 2. 强行关闭 SSL 证书验证
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# 修改提示信息，体积减小到约 4~5GB
print("🚀 开始下载轻量级 InternVL2.5-2B 模型，文件约 4~5GB，请耐心等待...")

snapshot_download(
    repo_id="OpenGVLab/InternVL2_5-2B",          # 这里改成了 2B 版本的官方仓库名
    local_dir="pretrained/InternVL2.5-2B",       # 这里改成了新的本地保存路径
    local_dir_use_symlinks=False
)

print("✅ 下载完成！模型已保存在 pretrained/InternVL2.5-2B 目录下。")