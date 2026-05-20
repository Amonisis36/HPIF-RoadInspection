import os
import shutil
from pathlib import Path
from tqdm import tqdm

def extract_clean_dataset(source_dir, target_dir):
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 我们需要严格保留的文件后缀清单
    keep_suffixes = [
        '_synthetic_road.npy',
        '_pure_shading.npy',
        '_aligned_normal.npy',
        '_aligned_albedo.npy',
        '_road_spliced_mask_padded.npy',
        '_info.json',
        #'_aligned_verts.npy',
        #'_aligned_roughness.npy'
    ]
    
    # 遍历 Train 和 Val
    splits = ['Train', 'Val']
    
    for split in splits:
        split_source = source_path / split
        if not split_source.exists():
            print(f"⚠️ 找不到文件夹: {split_source}")
            continue
            
        print(f"\n🚀 正在处理 {split} 集...")
        
        # 遍历比如 L-00034905-00000005_reflect 这样的序列文件夹
        seq_folders = [f for f in split_source.iterdir() if f.is_dir()]
        
        for seq_folder in tqdm(seq_folders, desc=f"复制 {split} 文件"):
            # 在目标路径创建同名文件夹
            seq_target = target_path / split / seq_folder.name
            seq_target.mkdir(parents=True, exist_ok=True)
            
            # 遍历文件夹内的文件，检查后缀
            for file_path in seq_folder.iterdir():
                if file_path.is_file():
                    # 检查文件名是否以我们需要的后缀结尾
                    if any(file_path.name.endswith(suffix) for suffix in keep_suffixes):
                        # 复制文件到新目录 (copy2 会保留文件的创建时间等元数据)
                        target_file = seq_target / file_path.name
                        if not target_file.exists():
                            shutil.copy2(file_path, target_file)

if __name__ == "__main__":
    # TODO: 确认你的原始路径
    SOURCE_DIRECTORY = "/home/rmt/datadisk/Data" 
    
    # TODO: 这是你要新建的、用于打包开源的干净路径
    TARGET_DIRECTORY = "/home/rmt/datadisk/ActiveLight-PBR_Release"
    
    print(f"开始提取干净数据集...")
    print(f"源路径: {SOURCE_DIRECTORY}")
    print(f"目标路径: {TARGET_DIRECTORY}")
    
    extract_clean_dataset(SOURCE_DIRECTORY, TARGET_DIRECTORY)
    
    print("\n✅ 提取完成！请去目标路径检查文件。")