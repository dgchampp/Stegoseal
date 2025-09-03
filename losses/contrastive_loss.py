import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, recovered_bits: torch.Tensor, original_bits: torch.Tensor) -> torch.Tensor:
        # Handle 32x32 messages (1024 bits)
        rec = torch.sigmoid(recovered_bits).detach()
        orig = original_bits
        N = rec.size(0)
        
        # Flatten 32x32 to 1024-dim vectors
        rec_vec = rec.view(N, -1)
        orig_vec = orig.view(N, -1)
        
        # Normalize
        rec_vec = F.normalize(rec_vec, dim=1)
        orig_vec = F.normalize(orig_vec, dim=1)
        
        # Similarity matrix
        sim_matrix = rec_vec @ orig_vec.t() / self.temperature
        labels = torch.arange(N, device=rec.device)
        return F.cross_entropy(sim_matrix, labels)