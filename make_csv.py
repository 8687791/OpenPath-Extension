import os
import pandas as pd

# 【修改点 1】确保路径指向你当前的实际数据存放位置
# 如果你的数据在 /root/gpufree-data/OpenPath-main/data/ 目录下，使用以下相对路径
train_data_path = "./data/NCT-CRC-HE-100K"
val_data_path = "./data/CRC-VAL-HE-7K"

classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

def generate(root_path, save_name):
    files = []
    # 检查根目录是否存在，避免报错
    if not os.path.exists(root_path):
        print(f"错误：找不到目录 {root_path}，请检查路径是否正确。")
        return

    for idx, cls in enumerate(classes):
        cls_folder = os.path.join(root_path, cls)
        if os.path.exists(cls_folder):
            for img in os.listdir(cls_folder):
                if img.endswith(('.tif', '.jpg', '.png')):
                    # 使用 abspath 会生成类似 /root/gpufree-data/OpenPath-main/data/... 的全路径
                    files.append({
                        'img': os.path.abspath(os.path.join(cls_folder, img)),
                        'label': idx
                    })
        else:
            print(f"警告：未找到类别目录 {cls_folder}")

    if len(files) > 0:
        df = pd.DataFrame(files)
        os.makedirs('al_file', exist_ok=True)
        df.to_csv(f'al_file/{save_name}', index=False)
        print(f"已生成 al_file/{save_name}，共 {len(df)} 张图。")
    else:
        print(f"未在 {root_path} 中发现任何图像文件。")

# 【修改点 2】将输出文件名改为 train.csv
# 因为你的训练脚本 train_sup_crc100k.py 默认寻找的是 al_file/train.csv
generate(train_data_path, "train.csv")
generate(val_data_path, "val_7k.csv")