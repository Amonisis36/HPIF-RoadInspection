import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# ==========================================
# 1. 导入依赖
# ==========================================
from common_utils import ssim, compute_metrics_tensor
from models import HPIF_LightGuidedUNet  # 我们上一轮脱敏好的主网络

# ==========================================
# 2. 开源数据集定制 Dataloader
# ==========================================
class ActiveLightTrainDataset(Dataset):
    def __init__(self, data_root, split='Train', crop_size=256, seq_length=8, epoch_multiplier=10):
        super().__init__()
        self.data_root = Path(data_root)
        self.split_dir = self.data_root / split
        self.crop_size = crop_size
        self.seq_length = seq_length
        # 🌟 恢复 10 倍增器
        self.epoch_multiplier = epoch_multiplier if split == 'Train' else 1
        
        self.left_prior = torch.from_numpy(np.load(self.data_root / "Global_Left_Prior.npy")).float()
        self.right_prior = torch.from_numpy(np.load(self.data_root / "Global_Right_Prior.npy")).float()
        
        self.samples = []
        if self.split_dir.exists():
            for seq_folder in self.split_dir.iterdir():
                if seq_folder.is_dir():
                    for f in seq_folder.glob('*_synthetic_road.npy'):
                        prefix = f.name.replace('_synthetic_road.npy', '')
                        self.samples.append({
                            'img': f,
                            'normal': seq_folder / f"{prefix}_aligned_normal.npy",
                            'is_left': prefix.startswith('L-')
                        })

    def __len__(self):
        return len(self.samples) * self.epoch_multiplier

    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        
        img = np.load(sample['img']).astype(np.float32)
        normal = np.load(sample['normal']).astype(np.float32)
        H, W = img.shape[:2]
        
        # Y 轴随机切
        start_y = np.random.randint(0, H - self.crop_size) if H > self.crop_size else 0
        
        # X 轴 8 连切
        x_starts = np.linspace(0, W - self.crop_size, self.seq_length).astype(int)
        
        prior_full = self.left_prior if sample['is_left'] else self.right_prior
        
        seq_img, seq_norm, seq_prior = [], [], []
        
        for start_x in x_starts:
            crop = np.s_[start_y : start_y + self.crop_size, start_x : start_x + self.crop_size, :]
            
            img_patch = torch.from_numpy(img[crop]).permute(2, 0, 1)
            norm_patch = torch.from_numpy(normal[crop]).permute(2, 0, 1)
            prior_patch = prior_full[:, start_y:start_y+self.crop_size, start_x:start_x+self.crop_size]
            
            seq_img.append((img_patch - 0.5) / 0.5)
            seq_norm.append(norm_patch)
            seq_prior.append(prior_patch)
            
        # 返回的是 [8, C, 256, 256] 的序列张量！
        return {
            'input_img': torch.stack(seq_img, dim=0),
            'gt_normal': torch.stack(seq_norm, dim=0),
            'light_prior': torch.stack(seq_prior, dim=0)
        }


# ==========================================
# 3. 辅助可视化函数
# ==========================================
def save_epoch_visualization(epoch, input_img, gt_normal, pred_normal, visual_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    vis_img = (input_img[0].cpu().permute(1,2,0).numpy() * 0.5) + 0.5
    axes[0,0].imshow(vis_img.clip(0,1))
    axes[0,0].set_title("Input Image")
    
    axes[0,1].imshow(gt_normal[0, 0].cpu().numpy(), cmap='bwr', vmin=-0.2, vmax=0.2)
    axes[0,1].set_title("GT Nx (Slope)")
    axes[0,2].imshow(gt_normal[0, 1].cpu().numpy(), cmap='bwr', vmin=-0.2, vmax=0.2)
    axes[0,2].set_title("GT Ny (Slope)")
    
    pred_vis = (pred_normal[0].cpu().permute(1,2,0).numpy() + 1.0) / 2.0
    axes[1,0].imshow(pred_vis)
    axes[1,0].set_title("Pred Normal (RGB)")
    
    axes[1,1].imshow(pred_normal[0, 0].cpu().numpy(), cmap='bwr', vmin=-0.2, vmax=0.2)
    axes[1,1].set_title("Pred Nx")
    axes[1,2].imshow(pred_normal[0, 1].cpu().numpy(), cmap='bwr', vmin=-0.2, vmax=0.2)
    axes[1,2].set_title("Pred Ny")
    
    for ax in axes.flat: ax.axis('off')
    plt.suptitle(f"Epoch {epoch} Prior Monitoring")
    plt.tight_layout()
    plt.savefig(f"{visual_dir}/epoch_{epoch:03d}.png")
    plt.close()

# ==========================================
# 4. 主训练循环
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Train HPIF Inverse Rendering Network")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to ActiveLight-PBR_Release")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--exp_name', type=str, default="HPIF_Official_Release")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Experiment: {args.exp_name} on {device}")

    # --- 目录初始化 ---
    ckpt_dir = f'checkpoints/{args.exp_name}'
    viz_dir = f'sanity_check_results/{args.exp_name}'
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # --- 数据加载 ---
    train_dataset = ActiveLightTrainDataset(args.data_dir, split='Train')
    val_dataset = ActiveLightTrainDataset(args.data_dir, split='Val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- 模型初始化 ---
    model = HPIF_LightGuidedUNet(in_channels=3, prior_channels=4).to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = torch.optim.AdamW(actual_model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_ang = float('inf')

    # --- 循环体 ---
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_mae_accum, train_ssim_accum = 0.0, 0.0, 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            input_img = batch['input_img'].to(device)
            gt_normal = batch['gt_normal'].to(device)
            light_prior = batch['light_prior'].to(device) 
            
            B, Seq, C, H, W = input_img.shape
            input_img = input_img.view(B * Seq, C, H, W)
            gt_normal = gt_normal.view(B * Seq, 3, H, W)
            light_prior = light_prior.view(B * Seq, 4, H, W) 

            optimizer.zero_grad()
            
            pred_norm, pred_alb = model(input_img, light_prior)

            # 损失计算
            loss_n_l1 = F.l1_loss(pred_norm, gt_normal)
            loss_n_cos = (1.0 - F.cosine_similarity(pred_norm, gt_normal, dim=1)).mean()
            # Dummy albedo loss (因为纯光照模态下 a 锁死为 0.2)
            loss_a_flat = F.l1_loss(pred_alb, torch.full_like(pred_alb, 0.2))

            loss = loss_n_l1 + loss_n_cos * 0.5 + loss_a_flat * 0.1
            loss.backward()
            optimizer.step()

            # 指标监控
            with torch.no_grad():
                cos_sim = F.cosine_similarity(pred_norm, gt_normal, dim=1).clamp(-1, 1)
                mae_batch = torch.acos(cos_sim).mean() * (180.0 / 3.14159)
                train_mae_accum += mae_batch.item()

                pred_nx = (pred_norm[:, 0:1, :, :] + 1.0) / 2.0
                gt_nx = (gt_normal[:, 0:1, :, :] + 1.0) / 2.0
                ssim_batch = ssim(pred_nx, gt_nx, data_range=1.0)
                train_ssim_accum += ssim_batch.item()
            
            train_loss += loss.item()
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}", 
                'MAE': f"{(train_mae_accum/(batch_idx + 1)):.1f}°"
            })

        # --- 验证阶段 ---
        model.eval()
        val_loss, val_ang, val_ssim_accum = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_img = batch['input_img'].to(device)
                gt_normal = batch['gt_normal'].to(device)
                light_prior = batch['light_prior'].to(device) 

                pred_norm, pred_alb = model(input_img, light_prior)

                l_l1 = F.l1_loss(pred_norm, gt_normal)
                l_cos = (1.0 - F.cosine_similarity(pred_norm, gt_normal, dim=1)).mean()
                l_a = F.l1_loss(pred_alb, torch.full_like(pred_alb, 0.2))
                loss = l_l1 + l_cos * 0.5 + l_a * 0.1

                ang_all, _, _, azimuth_hf = compute_metrics_tensor(pred_norm, gt_normal)
                
                pred_nx = (pred_norm[:, 0:1, :, :] + 1.0) / 2.0
                gt_nx = (gt_normal[:, 0:1, :, :] + 1.0) / 2.0
                ssim_batch = ssim(pred_nx, gt_nx, data_range=1.0)

                val_loss += loss.item()
                val_ang += ang_all.item()
                val_ssim_accum += ssim_batch.item() 

                if batch_idx == 0:
                    save_epoch_visualization(epoch+1, input_img, gt_normal, pred_norm, viz_dir)

        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae_accum / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ang = val_ang / len(val_loader)
        avg_val_ssim = val_ssim_accum / len(val_loader)

        print(f"\n🏁 Epoch {epoch+1} Summary:")
        print(f"   [Train] Loss: {avg_train_loss:.4f} | MAE: {avg_train_mae:.1f}°")
        print(f"   [Val]   Loss: {avg_val_loss:.4f} | MAE: {avg_val_ang:.1f}° | SSIM: {avg_val_ssim:.4f}\n")

        is_best = avg_val_ang < best_val_ang
        if is_best:
            best_val_ang = avg_val_ang

        save_state = {
            'epoch': epoch,
            'model_state_dict': actual_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_ang': best_val_ang
        }
        
        torch.save(save_state, f'{ckpt_dir}/latest.pth')
        if is_best:
            torch.save(save_state, f'{ckpt_dir}/best.pth')
            print(f"🌟 New Record! Best Checkpoint saved (MAE: {best_val_ang:.2f}°)")

if __name__ == '__main__':
    main()