import os
import argparse
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

# 导入咱们脱敏后的主网络
from models import HPIF_LightGuidedUNet

def load_image(img_path, device):
    """
    读取 JPG/PNG 图像，执行物理逆向渲染必须的反 Gamma 校正，并转为 Tensor
    """
    img_pil = Image.open(img_path).convert('L')
    W_orig, H_orig = img_pil.size
    
    # 转为 0~1 的 float，然后执行线性化 (Linearize)
    img_arr = np.array(img_pil, dtype=np.float32) / 255.0
    img_linear = np.power(img_arr, 2.2)
    
    # 归一化到 [-1, 1] 喂给网络
    img_tensor = (torch.from_numpy(img_linear).unsqueeze(0).unsqueeze(0) - 0.5) / 0.5
    
    # 为了 UNet 的下采样结构，长宽必须是 16 的整数倍
    pad_h = (16 - H_orig % 16) % 16
    pad_w = (16 - W_orig % 16) % 16
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='replicate')
        
    return img_tensor.to(device), (H_orig, W_orig)

def inference():
    parser = argparse.ArgumentParser(description="HPIF Official Inference Script")
    parser.add_argument('--input', type=str, required=True, help="Path to input JPG/PNG image")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained best.pth")
    parser.add_argument('--side', type=str, default='L', choices=['L', 'R'], help="Camera side: L or R")
    parser.add_argument('--prior_dir', type=str, default='.', help="Directory containing Global_Prior.npy files")
    parser.add_argument('--out_dir', type=str, default='./results', help="Directory to save predictions")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Using device: {device}")

    # ==========================================
    # 1. 实例化模型并加载权重
    # ==========================================
    print(f"📦 Loading model from {args.checkpoint}...")
    model = HPIF_LightGuidedUNet(in_channels=1, prior_channels=4).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # 兼容完整的存档字典或纯净的 state_dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 自动处理 DataParallel 的 'module.' 前缀问题
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()

    # ==========================================
    # 2. 准备数据 (图像 + 全局物理先验)
    # ==========================================
    print(f"🖼️ Processing input image: {args.input}")
    img_tensor, (H_orig, W_orig) = load_image(args.input, device)
    _, _, H_pad, W_pad = img_tensor.shape

    prior_name = "Global_Left_Prior.npy" if args.side == 'L' else "Global_Right_Prior.npy"
    prior_path = os.path.join(args.prior_dir, prior_name)
    
    if not os.path.exists(prior_path):
        raise FileNotFoundError(f"❌ Missing physical prior file: {prior_path}. Please check --prior_dir.")
        
    prior_full = torch.from_numpy(np.load(prior_path)).float().to(device)
    
    # 物理先验截取 (与图像对齐)
    prior_crop = prior_full[:, :H_pad, :W_pad].unsqueeze(0)

    # ==========================================
    # 3. 执行物理逆向推断
    # ==========================================
    print("🚀 Running physics-guided inference...")
    with torch.no_grad():
        pred_norm, pred_alb = model(img_tensor, prior_crop)

    # 裁掉之前为了 UNet 填充的边界，恢复真实尺寸
    pred_norm = pred_norm[:, :, :H_orig, :W_orig]
    pred_alb = pred_alb[:, :, :H_orig, :W_orig]

    # ==========================================
    # 4. 可视化并保存结果
    # ==========================================
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # A. 保存法向图 (Normal)
    # 将法线 [-1, 1] 映射到 [0, 255] RGB 空间
    norm_vis = (pred_norm[0].cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
    norm_vis = (norm_vis * 255.0).astype(np.uint8)
    norm_save_path = os.path.join(args.out_dir, f"{base_name}_normal.png")
    Image.fromarray(norm_vis).save(norm_save_path)
    
    # B. 保存本色图 (Albedo)
    # 本色范围在 [0.08, 0.8] 之间，进行线性拉伸方便观察
    alb_vis = pred_alb[0, 0].cpu().numpy()
    alb_vis = np.clip((alb_vis - 0.08) / 0.72, 0, 1) # 拉伸到 0-1
    alb_vis = (alb_vis * 255.0).astype(np.uint8)
    alb_save_path = os.path.join(args.out_dir, f"{base_name}_albedo.png")
    Image.fromarray(alb_vis).save(alb_save_path)

    print(f"✅ Success! Results saved to: {args.out_dir}/")
    print(f"   -> Normal: {os.path.basename(norm_save_path)}")
    print(f"   -> Albedo: {os.path.basename(alb_save_path)}")

if __name__ == "__main__":
    inference()