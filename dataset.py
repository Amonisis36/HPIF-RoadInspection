import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import glob
import json

class RoadScanDataset(Dataset):
    def __init__(self, 
                 data_root,
                 prior_root=None,     
                 target_width=1864,   
                 crop_height=256,     
                 is_train=True,       
                 default_pixel_size_mm=0.07,
                 file_keyword=None    
                 ):
        super().__init__()
        self.data_root = data_root
        self.target_width = target_width
        self.crop_height = crop_height
        self.is_train = is_train
        self.default_pixel_size = default_pixel_size_mm
        
        self.prior_dir = prior_root if prior_root else data_root
        left_prior_path = os.path.join(self.prior_dir, "Global_Left_Prior.npy")
        right_prior_path = os.path.join(self.prior_dir, "Global_Right_Prior.npy")
        
        self.left_prior = None
        self.right_prior = None
        
        if os.path.exists(left_prior_path):
            self.left_prior = torch.from_numpy(np.load(left_prior_path)).float()
            print(f"[Dataset] Loaded Global Left Prior: {self.left_prior.shape}")
        else:
            print(f"⚠️ [Dataset] Warning: Global_Left_Prior.npy not found in {self.prior_dir}")
            
        if os.path.exists(right_prior_path):
            self.right_prior = torch.from_numpy(np.load(right_prior_path)).float()
            print(f"[Dataset] Loaded Global Right Prior: {self.right_prior.shape}")
        else:
            print(f"⚠️ [Dataset] Warning: Global_Right_Prior.npy not found in {self.prior_dir}")
        # ==========================================
        
        # 1. 扫描所有 .jpg 文件
        all_files = sorted(glob.glob(os.path.join(data_root, "*.jpg")))
        if len(all_files) == 0:
            all_files = sorted(glob.glob(os.path.join(data_root, "*.JPG")))
            
        # 2. 按照关键字过滤
        if file_keyword is not None:
            self.image_paths = [f for f in all_files if file_keyword in os.path.basename(f)]
            print(f"[Dataset] Filtered by '{file_keyword}': {len(self.image_paths)} / {len(all_files)} images kept.")
        else:
            self.image_paths = all_files
            print(f"[Dataset] No filter. Using all {len(self.image_paths)} images.")

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_root} with keyword '{file_keyword}'")

        # 3. 加载标定字典 (可选)
        self.calib_dict = {}
        calib_path = os.path.join(data_root, "calibration.json")
        if os.path.exists(calib_path):
            with open(calib_path, 'r') as f:
                self.calib_dict = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def _load_and_process_image(self, path):
        img_pil = Image.open(path).convert('L') 
        W, H = img_pil.size
        
        if W != self.target_width:
            img_pil = img_pil.resize((self.target_width, H), Image.BICUBIC)
            
        img_arr = np.array(img_pil, dtype=np.float32) / 255.0
        
        # 反 Gamma 校正 (Linearize)
        img_linear = np.power(img_arr, 2.2)
        
        return torch.from_numpy(img_linear), H

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        file_name = os.path.basename(img_path)
        
        full_image_tensor, H_full = self._load_and_process_image(img_path)
        
        # 随机切片
        if self.is_train and H_full > self.crop_height:
            start_y = np.random.randint(0, H_full - self.crop_height)
            image_crop = full_image_tensor[start_y : start_y + self.crop_height, :]
        else:
            start_y = 0
            image_crop = full_image_tensor
            

        # 判断当前图像是左目还是右目 (兼容你的 L- 序列前缀，以及 L/Left 等命名法)
        is_left_camera = ("L-" in file_name or "_L" in file_name or "Left" in file_name)
        selected_prior = self.left_prior if is_left_camera else self.right_prior
        
        # 创建一个安全的 Default Dummy (如果没找到 prior 就用零填充，防止代码崩溃)
        prior_crop = torch.zeros((4, self.crop_height, self.target_width), dtype=torch.float32)
        
        if selected_prior is not None:
            # 鲁棒切片：因为物理先验在 Y 轴上是完美的刚体直线 (Y=0)
            prior_H, prior_W = selected_prior.shape[1], selected_prior.shape[2]
            
            # 安全截取，防止边界溢出
            crop_h = min(self.crop_height, prior_H)
            crop_w = min(self.target_width, prior_W)
            prior_crop[:, :crop_h, :crop_w] = selected_prior[:, :crop_h, :crop_w]
        # ==========================================
            
        pixel_size = self.calib_dict.get(file_name, self.default_pixel_size)
        
        return {
            # 加 unsqueeze(0) 是为了变成网络需要的 [1, H, W] 形状
            "gt_image": image_crop.unsqueeze(0),         
            "light_prior": prior_crop,           
            "pixel_size_mm": float(pixel_size),
            "start_y_idx": int(start_y),
            "name": file_name
        }

if __name__ == "__main__":
    
    data_dir = 'data/trap'
    # 实例化 Dataset
    ds = RoadScanDataset(data_root=data_dir, target_width=1864, crop_height=256)
    
    sample = ds[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"GT Shape: {sample['gt_image'].shape} (Should be [1, 256, 1864])")
    print(f"Prior Shape: {sample['light_prior'].shape} (Should be [4, 256, 1864])")
    print(f"Pixel Size: {sample['pixel_size_mm']}")
    print(f"Max Value (Linear): {sample['gt_image'].max():.4f}")