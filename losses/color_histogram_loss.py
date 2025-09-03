import torch
import torch.nn as nn

class ColorHistogramLoss(nn.Module):
    """
    Color histogram loss.
    Compares the color distribution of two images by computing smooth color histograms and taking L2 distance.
    """
    def __init__(self, bins: int = 64, sigma: float = 0.02):
        super().__init__()
        self.bins = bins
        self.sigma = sigma
        # Prepare bin centers in [0,1]
        self.register_buffer('bin_centers', torch.linspace(0.0, 1.0, bins).view(1, 1, bins))

    def _soft_histogram(self, img: torch.Tensor) -> torch.Tensor:
        # img: (N,3,H,W) in [0,1]
        N, C, H, W = img.shape
        pixels = img.reshape(N, C, -1)  # (N,3, H*W)
        # Compute soft histogram for each channel
        # Using Gaussian kernel around each bin center
        # Expand dims for broadcasting: pixels (N, C, P, 1), bin_centers (1,1,bins)
        pixels = pixels.unsqueeze(-1)  # (N,3,P,1)
        centers = self.bin_centers  # (1,1,bins)
        diff = pixels - centers  # shape (N,3,P,bins)
        # Gaussian weighting for each bin
        weights = torch.exp(- (diff / self.sigma)**2)  # (N,3,P,bins)
        # Sum over pixels
        hist = weights.sum(dim=2)  # (N,3,bins)
        # Normalize histograms to probability distribution (sum=1 per channel)
        hist = hist / (hist.sum(dim=2, keepdim=True) + 1e-6)
        return hist  # shape (N,3,bins)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are in [0,1] range
        img1 = torch.clamp(img1, 0.0, 1.0)
        img2 = torch.clamp(img2, 0.0, 1.0)
        # Compute soft histograms for each image
        hist1 = self._soft_histogram(img1)
        hist2 = self._soft_histogram(img2)
        # Compute L2 loss between the two histograms, averaged over channels
        loss = torch.mean((hist1 - hist2)**2)
        return loss
