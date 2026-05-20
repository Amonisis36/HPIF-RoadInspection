import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2

# ==========================================
# 1. 核心评估指标 (Metrics)
# ==========================================

from skimage.metrics import structural_similarity as skimage_ssim

def ssim(pred_tensor, gt_tensor, data_range=1.0):
    """
    官方 skimage SSIM 的 Tensor 包装器。
    接收 [B, C, H, W] 的 PyTorch Tensor，自动转换为 NumPy 并在 CPU 上使用官方库计算平均 SSIM。
    """
    # 1. 安全脱离计算图，转移到 CPU 并转为 NumPy
    pred = pred_tensor.detach().cpu().numpy()
    gt = gt_tensor.detach().cpu().numpy()
    
    batch_size = pred.shape[0]
    total_ssim = 0.0
    
    # 2. 逐样本计算
    for i in range(batch_size):
        # [C, H, W] -> [H, W, C] (skimage 的默认偏好)
        p = np.transpose(pred[i], (1, 2, 0))
        g = np.transpose(gt[i], (1, 2, 0))
        
        # 自动适配单通道(如 Nx)和多通道(如 Albedo)
        if p.shape[-1] == 1:
            p = p.squeeze(-1)
            g = g.squeeze(-1)
            # 单通道图像不需要 channel_axis
            val = skimage_ssim(p, g, data_range=data_range)
        else:
            # 多通道图像必须指定 channel_axis
            val = skimage_ssim(p, g, data_range=data_range, channel_axis=-1)
            
        total_ssim += val
        
    return total_ssim / batch_size

def compute_metrics_tensor(pred_n, gt_n, hf_percentile=0.90):
    pred_n_unit = pred_n / (torch.norm(pred_n, dim=1, keepdim=True) + 1e-8)
    gt_n_unit = gt_n / (torch.norm(gt_n, dim=1, keepdim=True) + 1e-8)
    
    dot = torch.clamp(torch.sum(pred_n_unit * gt_n_unit, dim=1, keepdim=True), -1.0, 1.0)
    ang_err_map = torch.acos(dot) * (180.0 / math.pi)
    eval_norm_ang_all = torch.mean(ang_err_map)
    
    pred_nz = torch.clamp(pred_n_unit[:, 2:3, :, :], -1.0, 1.0)
    gt_nz = torch.clamp(gt_n_unit[:, 2:3, :, :], -1.0, 1.0)
    zenith_err_map = torch.abs(torch.acos(pred_nz) - torch.acos(gt_nz)) * (180.0 / math.pi)

    pred_xy = pred_n_unit[:, :2, :, :]
    gt_xy = gt_n_unit[:, :2, :, :]
    pred_xy_norm = pred_xy / torch.clamp(torch.norm(pred_xy, dim=1, keepdim=True), min=1e-6)
    gt_xy_norm = gt_xy / torch.clamp(torch.norm(gt_xy, dim=1, keepdim=True), min=1e-6)
    dot_xy = torch.clamp(torch.sum(pred_xy_norm * gt_xy_norm, dim=1, keepdim=True), -1.0, 1.0)
    azimuth_err_map = torch.acos(dot_xy) * (180.0 / math.pi)

    diff_x = torch.abs(gt_n_unit[:, :, :, 1:] - gt_n_unit[:, :, :, :-1])
    diff_y = torch.abs(gt_n_unit[:, :, 1:, :] - gt_n_unit[:, :, :-1, :])
    gt_grad_mag = torch.sum(F.pad(diff_x, (0,1,0,0)) + F.pad(diff_y, (0,0,0,1)), dim=1, keepdim=True)
    
    dynamic_thresh = torch.quantile(gt_grad_mag.view(-1).float(), hf_percentile)
    hf_mask = (gt_grad_mag > dynamic_thresh).float()
    
    valid_hf_pixels = torch.sum(hf_mask) + 1e-8
    eval_norm_ang_hf = torch.sum(ang_err_map * hf_mask) / valid_hf_pixels
    eval_zenith_hf = torch.sum(zenith_err_map * hf_mask) / valid_hf_pixels
    eval_azimuth_hf = torch.sum(azimuth_err_map * hf_mask) / valid_hf_pixels
    
    return eval_norm_ang_all, eval_norm_ang_hf, eval_zenith_hf, eval_azimuth_hf

# ==========================================
# 2. Loss Functions
# ==========================================

def compute_tv_loss(img_tensor):
    """全变分损失 (Total Variation)，用于抑制高频噪声。"""
    diff_h = torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :])
    diff_w = torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1])
    return torch.mean(diff_h) + torch.mean(diff_w)

def compute_normal_sparsity_loss(pred_norm):
    """法线稀疏化损失，鼓励法线在非边缘区域保持平整。"""
    nx = pred_norm[:, 0:1, :, :]
    ny = pred_norm[:, 1:2, :, :]
    nx_grad_x = torch.abs(nx[:, :, :, 1:] - nx[:, :, :, :-1])
    ny_grad_y = torch.abs(ny[:, :, 1:, :] - ny[:, :, :-1, :])
    return torch.mean(nx_grad_x) + torch.mean(ny_grad_y)

def compute_curl_loss(pred_norm):
    """旋度损失 (Curl Loss)，强制法线场符合可积性 (Integrability) 约束。"""
    nx = pred_norm[:, 0:1, :, :]
    ny = pred_norm[:, 1:2, :, :]
    dnx_dy = nx[:, :, 1:, :-1] - nx[:, :, :-1, :-1]
    dny_dx = ny[:, :, :-1, 1:] - ny[:, :, :-1, :-1]
    return torch.mean(torch.abs(dnx_dy - dny_dx))

def compute_l2_smoothness_loss(pred_norm):
    """L2 平滑损失，对突变施加指数级惩罚，用于获取极低频的宏观基底。"""
    dx = torch.mean((pred_norm[:, :, :, 1:] - pred_norm[:, :, :, :-1])**2)
    dy = torch.mean((pred_norm[:, :, 1:, :] - pred_norm[:, :, :-1, :])**2)
    return dx + dy

# ==========================================
# 3. Retinex 等对比基线
# ==========================================

def multi_scale_retinex(img, sigma_list=[15, 80, 250]):
    """多尺度 Retinex (MSR)，用于传统方法对比实验。"""
    def ssr(img_in, sig):
        img_in = np.float64(img_in) + 1.0
        illum = cv2.GaussianBlur(img_in, (0, 0), sig)
        return np.log10(img_in) - np.log10(illum)

    retinex = np.zeros_like(img, dtype=np.float64)
    for sigma in sigma_list:
        retinex += ssr(img, sigma)
    
    retinex = retinex / len(sigma_list)
    mean, std = np.mean(retinex), np.std(retinex)
    retinex = (retinex - (mean - 2 * std)) / (4 * std) * 255.0
    return np.clip(retinex, 0, 255).astype(np.uint8)