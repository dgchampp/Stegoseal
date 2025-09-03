import torch
import torch.nn as nn
import torch.nn.functional as F
from .preprocessing import wavelet_transform, gradient_transform  # Fixed relative import

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )
        
    def forward(self, x):
        return x + self.conv(x)

class FrequencyDomainBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dct_conv = nn.Conv2d(2 * in_channels, 2 * in_channels, 3, padding=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels//8), 1),
            nn.ReLU(),
            nn.Conv2d(max(1, in_channels//8), in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # DCT-like processing
        x_dct = torch.fft.rfft2(x, norm='ortho')
        x_dct = torch.cat([x_dct.real, x_dct.imag], dim=1)
        x_dct = self.dct_conv(x_dct)
        real, imag = torch.chunk(x_dct, 2, dim=1)
        x_dct = torch.fft.irfft2(torch.complex(real, imag), s=x.shape[-2:], norm='ortho')
        
        # Attention fusion
        attn = self.attention(x)
        return x * attn + x_dct * (1 - attn)

class EdgeAwareRecovery(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.grad_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        gradients = gradient_transform(x)
        return x + self.grad_conv(gradients)

class RobustDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Wavelet processing path
        self.wavelet_path = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1),  # 12 channels from wavelet transform
            FrequencyDomainBlock(64)
        )
        
        # RGB processing path
        self.rgb_path = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )
        
        # Feature fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.ReLU()
        )
        
        # Feature extractor
        self.blocks = nn.Sequential(
            FrequencyDomainBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            FrequencyDomainBlock(256),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(),
            FrequencyDomainBlock(512)
        )
        
        # Edge-aware recovery
        self.edge_recovery = EdgeAwareRecovery(512)
        
        # Message reconstruction
        self.msg_recon = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.AdaptiveAvgPool2d((32, 32))
        )
        
    def forward(self, stego):
        # Wavelet processing
        wavelet_coeffs = wavelet_transform(stego)
        wavelet_feat = self.wavelet_path(wavelet_coeffs)
        
        # RGB processing
        rgb_feat = self.rgb_path(stego)
        
        # Upsample wavelet features to match RGB spatial dimensions
        wavelet_feat = F.interpolate(wavelet_feat, 
                                     size=rgb_feat.shape[2:], 
                                     mode='bilinear', 
                                     align_corners=False)
        
        # Feature fusion
        fused = torch.cat([rgb_feat, wavelet_feat], dim=1)
        x = self.fuse(fused)
        
        # Feature extraction
        features = self.blocks(x)
        
        # Edge-enhanced recovery
        edge_enhanced = self.edge_recovery(features)
        
        return self.msg_recon(edge_enhanced)