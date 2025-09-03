import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyDomainLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha  # Weight for phase loss
        
    def forward(self, img1, img2):
        # Compute FFT
        fft1 = torch.fft.rfft2(img1, norm='ortho')
        fft2 = torch.fft.rfft2(img2, norm='ortho')
        
        # Amplitude and phase
        amp1 = torch.abs(fft1)
        phase1 = torch.angle(fft1)
        amp2 = torch.abs(fft2)
        phase2 = torch.angle(fft2)
        
        # Amplitude loss (log-space for perceptual relevance)
        amp_loss = F.mse_loss(torch.log1p(amp1), torch.log1p(amp2))
        
        # Phase loss (circular distance)
        phase_diff = torch.cos(phase1 - phase2)
        phase_loss = 1 - phase_diff.mean()
        
        return amp_loss + self.alpha * phase_loss