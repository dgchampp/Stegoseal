import torch
import torch.nn as nn

class DCTSpectralLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, decoded: torch.Tensor, target: torch.Tensor):
        # FFT2 for DCT-style spectral energy comparison
        dct_dec = torch.fft.fft2(decoded, norm="ortho")
        dct_tgt = torch.fft.fft2(target, norm="ortho")

        # Compute power spectrum
        power_dec = torch.abs(dct_dec) ** 2
        power_tgt = torch.abs(dct_tgt) ** 2

        return self.mse(power_dec, power_tgt)
