import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 💡 核心修改一：根据截图，精准锁定图片文件夹的上一级父目录
skin_data_root = "./data/SkinTissue/IMG_CLASSES"

# 💡 核心修改二：精准对应你截图里的 10 大全量皮肤疾病文件夹名称
classes = [
    '1. Eczema 1677',
    '2. Melanoma 15.75k',
    '3. Atopic Dermatitis - 1.25k',
    '4. Basal Cell Carcinoma (BCC) 3323',
    '5. Melanocytic Nevi (NV) - 7970',
    '6. Benign Keratosis-like Lesions (BKL) 2624',
    '7. Psoriasis pictures Lichen Planus and related diseases - 2k',
    '8. Seborrheic Keratoses and other Benign Tumors - 1.8k',
    '9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k',
    '10. Warts Molluscum and other Viral Infections - 2.1k'
]

def generate_skin_csv():
    files = []
    if not os.path.exists(skin_data_root):
        print(f"❌ 错误：找不到皮肤核心数据目录 {skin_data_root}，请检查路径大小写是否与截图一致。")
        return

    print("📂 正在全量扫描 10 大皮肤疾病车间图片...")
    for idx, cls_folder_name in enumerate(classes):
        cls_folder_path = os.path.join(skin_data_root, cls_folder_name)
        
        if os.path.exists(cls_folder_path):
            # 扫描当前疾病文件夹下的所有病理/临床图像
            img_list = os.listdir(cls_folder_path)
            counter = 0
            for img_name in img_list:
                if img_name.lower().endswith(('.tif', '.jpg', '.jpeg', '.png')):
                    img_full_path = os.path.abspath(os.path.join(cls_folder_path, img_name))
                    files.append({
                        'img': img_full_path,
                        'label': idx  # 映射为 0 到 9 的数字标签
                    })
                    counter += 1
            print(f"   - 发现类别 [{cls_folder_name}] 包含 {counter} 张合规图片")
        else:
            print(f"⚠️ 警告：未在硬盘中检索到目录 {cls_folder_path}")

    if len(files) > 0:
        total_df = pd.DataFrame(files)
        
        # 💡 核心修改三：将账本物理隔离输出到 al_file_skin 文件夹
        os.makedirs('al_file_skin', exist_ok=True)
        
        # 打乱全量数据集
        total_df = total_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 💡 核心修改四：学术标准 8:2 划分（80% 用于主动学习池，20% 留作期末测试）
        train_df, val_df = train_test_split(total_df, test_size=0.2, random_state=42, stratify=total_df['label'])
        
        # 落盘封存
        train_df.to_csv("al_file_skin/train.csv", index=False)
        val_df.to_csv("al_file_skin/val_7k.csv", index=False)
        
        print(f"\n🎉 10分类皮肤数据集洗档圆满成功！")
        print(f"📊 【当前独立大盘数据盘点】:")
        print(f"   - 待挖掘主动学习池 (al_file_skin/train.csv): {len(train_df)} 张图")
        print(f"   - 独立期末验证集   (al_file_skin/val_7k.csv): {len(val_df)} 张图")
    else:
        print("❌ 严重错误：未在指定目录下扫描到任何有效的图像文件！")

if __name__ == "__main__":
    generate_skin_csv()