import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# 1. 解耦注意力组件 
# ==========================================================

class Y_AxialAttention(nn.Module):
    """
    Y轴单向注意力机制 (纵向列注意力)
    物理意义：在线扫系统纵向光照恒定的庇护下，沿着 Y 轴提取纯粹的几何高频坑洼一致性。
    """
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, in_channels)
        )
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # [B, C, H, W] -> [B * W, H, C]
        x_reshaped = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)
        
        normed_x = self.norm1(x_reshaped)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x)
        out = x_reshaped + attn_out

        ffn_out = self.ffn(self.norm2(out))
        out = out + ffn_out
        
        # [B * W, H, C] -> [B, C, H, W]
        out = out.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()
        return out

class BottleneckSequenceAttention(nn.Module):
    """
    序列注意力机制 (底层宏观视野)
    """
    def __init__(self, channels=1024, seq_len=8):
        super().__init__()
        self.seq_len = seq_len
        if self.seq_len > 1:
            self.attn = nn.TransformerEncoderLayer(
                d_model=channels, 
                nhead=8, 
                dim_feedforward=channels * 2, 
                batch_first=True
            )

    def forward(self, x):
        if self.seq_len <= 1:
            return x
        B_seq, C, H, W = x.shape
        B = B_seq // self.seq_len
        x_seq = x.view(B, self.seq_len, C, H, W)
        global_context = x_seq.mean(dim=[-1, -2]) 
        attended_context = self.attn(global_context) 
        attended_context = attended_context.view(B, self.seq_len, C, 1, 1)
        x_out = x_seq + attended_context
        return x_out.view(B_seq, C, H, W)

class SpatialLightingModulation(nn.Module):
    def __init__(self, feature_channels, light_channels=4):
        super().__init__()
        self.light_encoder = nn.Sequential(
            nn.Conv2d(light_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feature_channels * 2, kernel_size=3, padding=1)
        )
        # Zero-Initialization 
        nn.init.zeros_(self.light_encoder[-1].weight)
        nn.init.zeros_(self.light_encoder[-1].bias)

    def forward(self, x, physical_light_map):
        if physical_light_map.shape[-2:] != x.shape[-2:]:
            physical_light_map = F.interpolate(physical_light_map, size=x.shape[-2:], mode='bilinear', align_corners=False)

        gamma_beta = self.light_encoder(physical_light_map)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return x * (1.0 + gamma) + beta

# ==========================================================
# 2. 基础 UNet 
# ==========================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class BaseUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = DoubleConv(1024 + 512, 512)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = DoubleConv(512 + 256, 256)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = DoubleConv(256 + 128, 128)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up4 = DoubleConv(128 + 64, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

# ==========================================================
# 3.  HPIF 
# ==========================================================

class HPIF_LightGuidedUNet(BaseUNet):
    def __init__(self, in_channels=1, out_channels=4, seq_len=1, prior_channels=4):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        
        # 1. 宏观去偏置注意力 (底层)
        self.seq_attn = BottleneckSequenceAttention(channels=1024, seq_len=seq_len)
        
        # 2. 微观几何注意力 (浅层)
        self.y_axial_attn = Y_AxialAttention(in_channels=128, num_heads=4)
        
        # 3. 物理先验注入器 (高分辨率层)
        self.modulator = SpatialLightingModulation(feature_channels=64, light_channels=prior_channels)
        
        # 4. 彻底的物理拆解解码头 (N-Disentanglement)
        self.head_nx = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.head_ny = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.head_nz_alb = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x, light_prior):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) 
        
        # 底层宏观特征融合
        x5 = self.seq_attn(x5)
        
        # Decoder
        dx = self.up1(x5)
        dx = self.conv_up1(torch.cat([dx, x4], dim=1))
        dx = self.up2(dx)
        dx = self.conv_up2(torch.cat([dx, x3], dim=1))
        dx = self.up3(dx)
        dx = self.conv_up3(torch.cat([dx, x2], dim=1))
        
        # 浅层微观特征开会 (强制提取坑洼)
        dx = self.y_axial_attn(dx)
        
        dx = self.up4(dx)
        features_64 = self.conv_up4(torch.cat([dx, x1], dim=1)) 
        
        # 🌟 核心注入点：用脱敏提供的物理先验调制深层特征
        modulated_features = self.modulator(features_64, light_prior)

        # 物理拆解输出
        out_nx = self.head_nx(modulated_features)
        out_ny = self.head_ny(modulated_features)
        out_nz_alb = self.head_nz_alb(modulated_features)
        
        # [B, 4, H, W] -> nx(1), ny(1), nz_alb(2)
        raw_out = torch.cat([out_nx, out_ny, out_nz_alb], dim=1)
        
        # 后处理物理截断
        raw_norm = raw_out[:, 0:3, :, :]
        raw_alb = raw_out[:, 3:4, :, :]
        
        nx = torch.tanh(raw_norm[:, 0:1, :, :]) 
        ny = torch.tanh(raw_norm[:, 1:2, :, :])
        nz = torch.sigmoid(raw_norm[:, 2:3, :, :]) + 1e-4 
        norm = F.normalize(torch.cat([nx, ny, nz], dim=1), p=2, dim=1)
        
        gray_alb = 0.08 + 0.72 * torch.sigmoid(raw_alb)
        
        return norm, gray_alb

# ==========================================================
# 单元测试
# ==========================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HPIF_LightGuidedUNet(in_channels=3, prior_channels=4, seq_len=8).to(device)

    # 模拟输入 (单通道灰度图) 和 物理先验图 (4通道)
    dummy_img = torch.randn(2, 1, 256, 1864).to(device)
    dummy_prior = torch.randn(2, 4, 256, 1864).to(device)
    
    pred_norm, pred_alb = model(dummy_img, dummy_prior)
    
    print(f"Network Output - Normal Shape: {pred_norm.shape}")
    print(f"Network Output - Albedo Shape: {pred_alb.shape}")