import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class LPIPSLoss(nn.Module):
    """
    LPIPS perceptual loss using a pre-trained VGG16 network.
    Measures perceptual similarity between two images.
    """
    def __init__(self):
        super().__init__()
        # Load pre-trained VGG16 and select feature layers
        vgg = models.vgg16(pretrained=True).features
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg_layers = vgg  # we'll iterate through these layers on forward
        # Define which layer indices to capture features from (after ReLU activations of conv blocks)
        self.capture_indices = {3, 8, 15, 22, 29}  # conv1_2, conv2_2, conv3_3, conv4_3, conv5_3 (after ReLUs)
        # Mean and std for normalization (ImageNet)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # Ensure input in range [0,1] then normalize to VGG expected range
        x1 = (img1 - self.mean) / self.std
        x2 = (img2 - self.mean) / self.std
        # Extract features at the specified layers
        feats1 = []
        feats2 = []
        x1_in = x1; x2_in = x2
        for i, layer in enumerate(self.vgg_layers):
            x1_in = layer(x1_in)
            x2_in = layer(x2_in)
            if i in self.capture_indices:
                feats1.append(x1_in)
                feats2.append(x2_in)
        # Compute LPIPS as average L2 distance across all captured feature maps
        loss = 0.0
        for f1, f2 in zip(feats1, feats2):
            # L2 difference normalized by number of elements
            loss += F.mse_loss(f1, f2, reduction='mean')
        return loss
