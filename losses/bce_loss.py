import torch.nn as nn

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, target_bits):
        # Now handles 32x32 messages
        return self.bce(pred_logits, target_bits)