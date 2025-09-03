import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .preprocessing import wavelet_transform  # Relative import from same package

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1) * x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion),
            nn.GELU(),
            nn.Linear(dim * ffn_expansion, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> (h w) b c')
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_flat + attn_out
        x = x + self.ffn(self.norm2(x))
        return rearrange(x, '(h w) b c -> b c h w', h=H, w=W)

class RobustEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Wavelet preprocessing branch
        self.wavelet_prep = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1),  # 12 channels from wavelet transform
            nn.ReLU(),
            ChannelAttention(64)
        )
        
        # RGB processing path
        self.rgb_path = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )
        
        # Feature fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, 1),  # 64 (RGB) + 64 (wavelet) = 128
            nn.ReLU()
        )
        
        # Downsample blocks
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            ChannelAttention(128),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            ChannelAttention(256),
            nn.ReLU()
        )
        
        # Transformer processing
        self.transformer = TransformerBlock(256)
        
        # Message integration (for 32x32 input)
        self.msg_proj = nn.Sequential(
            nn.Conv2d(1, 256, 1),
            nn.Sigmoid()
        )
        
        # Upsample blocks
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            ChannelAttention(128),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            ChannelAttention(64),
            nn.ReLU()
        )
        
        # Final output
        self.final_conv = nn.Conv2d(64, 3, 3, padding=1)
        self.tanh = nn.Tanh()
    
    def forward(self, cover, msg):
        # Wavelet processing path
        wavelet_coeffs = wavelet_transform(cover)
        wavelet_feat = self.wavelet_prep(wavelet_coeffs)
        
        # RGB processing path
        rgb_feat = self.rgb_path(cover)
        
        # Resize wavelet features to match RGB spatial dimensions
        wavelet_feat = F.interpolate(wavelet_feat, 
                                     size=rgb_feat.shape[2:], 
                                     mode='bilinear', 
                                     align_corners=False)
        
        # Feature fusion
        fused = torch.cat([rgb_feat, wavelet_feat], dim=1)
        x = self.fuse(fused)
        
        # Continue with encoder flow
        d1 = self.down1(x)
        d2 = self.down2(d1)
        trans_out = self.transformer(d2)
        
        # Message integration (upsample 32x32 to match trans_out size)
        msg_up = F.interpolate(msg, size=trans_out.shape[2:], mode='bilinear', align_corners=False)
        msg_feat = self.msg_proj(msg_up)
        fused = trans_out * msg_feat
        
        u1 = self.up1(fused)
        u2 = self.up2(u1 + d1)
        
        # Residual connection with controlled perturbation
        perturbation = self.tanh(self.final_conv(u2 + x)) * 0.1
        return torch.clamp(cover + perturbation, 0, 1)