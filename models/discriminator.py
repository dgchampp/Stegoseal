import torch
import torch.nn as nn

class SpatialDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1))
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        self.conv3 = nn.utils.spectral_norm(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.conv4 = nn.utils.spectral_norm(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1))
        self.conv5 = nn.utils.spectral_norm(
            nn.Conv2d(512, 1, kernel_size=16, stride=1, padding=0))
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.conv5(x)
        return x.view(x.size(0), -1)
        
class SpectralDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        # Enhanced for 32x32 message input
        self.conv0 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1))
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1))
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        self.conv3 = nn.utils.spectral_norm(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.conv4 = nn.utils.spectral_norm(
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preprocess
        X_fft = torch.fft.fft2(x, norm="ortho")
        X_mag = torch.abs(X_fft)
        X_log = torch.log1p(X_mag)
        
        # Process through CNN
        x = self.lrelu(self.conv0(X_log))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.conv4(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)