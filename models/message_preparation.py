import torch
import torch.nn as nn

class MessagePrepNetwork(nn.Module):
    """
    Enhanced Message Preparation Network with residual connections
    """
    def __init__(self):
        super().__init__()
        # Input: 15-channel 16x16 (from wavelet+original message)
        self.conv_in = nn.Conv2d(15, 64, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        
        # Upsampling blocks with residual connections
        self.up1 = self._upsample_block(64, 64)
        self.up2 = self._upsample_block(64, 64)
        self.up3 = self._upsample_block(64, 64)
        self.up4 = self._upsample_block(64, 64)
        
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def _upsample_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, msg_freq: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv_in(msg_freq))
        
        # Upsampling with skip connections
        x = x + self.up1(x)
        x = x + self.up2(x)
        x = x + self.up3(x)
        x = x + self.up4(x)
        
        out = self.conv_out(x)
        return out