import torch
import torch.nn as nn
import torch.nn.functional as F
def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Enhanced convolution block with instance normalization"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    """
    Enhanced U-Net architecture with attention gates
    """
    def __init__(self, in_channels: int, base_filters: int = 64):
        super().__init__()
        # Down-sampling path
        self.conv1 = conv_block(in_channels, base_filters)
        self.conv2 = conv_block(base_filters, base_filters*2)
        self.conv3 = conv_block(base_filters*2, base_filters*4)
        self.conv4 = conv_block(base_filters*4, base_filters*8)
        self.bottom = conv_block(base_filters*8, base_filters*8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Attention gates
        self.attn1 = nn.Conv2d(base_filters*8, 1, kernel_size=1)
        self.attn2 = nn.Conv2d(base_filters*4, 1, kernel_size=1)
        self.attn3 = nn.Conv2d(base_filters*2, 1, kernel_size=1)
        self.attn4 = nn.Conv2d(base_filters, 1, kernel_size=1)

        # Up-sampling path
        self.up_conv1 = conv_block(base_filters*8 + base_filters*8, base_filters*8)
        self.up_conv2 = conv_block(base_filters*8 + base_filters*4, base_filters*4)
        self.up_conv3 = conv_block(base_filters*4 + base_filters*2, base_filters*2)
        self.up_conv4 = conv_block(base_filters*2 + base_filters, base_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsample with conv blocks and pooling
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        xb = self.bottom(self.pool(x4))

        # Upsample with attention gates
        x_up = F.interpolate(xb, scale_factor=2, mode='nearest')
        attn = torch.sigmoid(self.attn1(x4))
        x_up = torch.cat([x4 * attn, x_up], dim=1)
        x_up = self.up_conv1(x_up)

        x_up = F.interpolate(x_up, scale_factor=2, mode='nearest')
        attn = torch.sigmoid(self.attn2(x3))
        x_up = torch.cat([x3 * attn, x_up], dim=1)
        x_up = self.up_conv2(x_up)

        x_up = F.interpolate(x_up, scale_factor=2, mode='nearest')
        attn = torch.sigmoid(self.attn3(x2))
        x_up = torch.cat([x2 * attn, x_up], dim=1)
        x_up = self.up_conv3(x_up)

        x_up = F.interpolate(x_up, scale_factor=2, mode='nearest')
        attn = torch.sigmoid(self.attn4(x1))
        x_up = torch.cat([x1 * attn, x_up], dim=1)
        x_up = self.up_conv4(x_up)

        return x_up