import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur  # Add this import


# Import robust components
from models.encoder import RobustEncoder as Encoder
from models.decoder import RobustDecoder as Decoder
from models.discriminator import SpatialDiscriminator, SpectralDiscriminator
from losses.lpips_loss import LPIPSLoss
from losses.color_histogram_loss import ColorHistogramLoss
from losses.contrastive_loss import ContrastiveLoss
from losses.bce_loss import BinaryCrossEntropyLoss
from losses.frequency_loss import FrequencyDomainLoss
from models.preprocessing import wavelet_transform, inverse_wavelet_transform, get_wavelet_bands, gradient_transform

# Hyperparameters
data_dir = "/home/rio/Desktop/day3 (Working)/deep2/data/30000"
epochs = 100
batch_size =4 
lr = 1e-4
lr_decoder = 5e-4 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss coefficients
lambda_color = 1.0
lambda_lpips = 1.0
lambda_adv_spatial = 1.0
lambda_bce = 8.0
lambda_contrast = 1.5
lambda_adv_spectral = 0.5
lambda_freq = 0.5
lambda_wavelet = 0.7  # New wavelet consistency loss

# Distortion parameters
DISTORTION_PROB = 0.7  # 70% of images will be distorted
DISTORTION_LEVELS = [0.6, 0.7, 0.8, 0.9]  # Variable distortion intensities

transform = transforms.Compose([
    # Random horizontal flip (50% probability)
    transforms.RandomHorizontalFlip(p=0.5),
    
    # Random resized crop with controlled aspect ratio
    transforms.RandomResizedCrop(
        size=256,
        scale=(0.9, 1.0),  # 90-100% of original area
        ratio=(0.98, 1.02)  # Aspect ratio range
    ),
    
    # Color jitter (brightness/contrast/saturation/hue)
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
        hue=0.05
    ),
    
    # Gaussian blur
    transforms.GaussianBlur(
        kernel_size=5,
        sigma=(0.1, 2.0)
    ),
    
    # Convert to tensor
    transforms.ToTensor()
])

class CoverDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.files = []
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        for fname in os.listdir(image_dir):
            if any(fname.lower().endswith(ext) for ext in exts):
                self.files.append(os.path.join(image_dir, fname))
        self.num_files = len(self.files)
        assert self.num_files > 0, "No training images found."

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        img_path = self.files[idx % self.num_files]
        image = Image.open(img_path).convert('RGB')
        cover_tensor = transform(image)
        # 32x32 message (1024 bits)
        msg_bits = torch.randint(0, 2, (1, 32, 32), dtype=torch.float32)
        return cover_tensor, msg_bits

# Enhanced print-scan simulation

def robust_print_scan(image, distortion_level=0.8):
    """Physics-based print-scan simulation with dimension handling"""
    # Preserve original dimensions
    original_dim = image.dim()
    if original_dim == 3:
        image = image.unsqueeze(0)  # add batch dim for processing
    
    # 1. Convert to wavelet domain
    coeffs = wavelet_transform(image)
    bands = get_wavelet_bands(coeffs)
    
    # 2. Apply distortions to vulnerable bands
    # - Ink bleed (affects high frequencies)
    bands[11] += distortion_level * 0.1 * torch.randn_like(bands[11])  # HH band
    bands[7]  += distortion_level * 0.05 * torch.randn_like(bands[7])   # HL band
    
    # - Paper texture (affects mid frequencies)
    if distortion_level > 0.5:
        texture = torch.rand_like(bands[3]) * distortion_level * 0.3
        bands[3] += texture  # LH band
        bands[4] += texture  # HL band
    
    # 3. Reconstruct image
    modified_coeffs = torch.cat(bands, dim=1)
    reconstructed = inverse_wavelet_transform(modified_coeffs)
    
    # 4. Color shifts
    reconstructed = torch.clamp(reconstructed + distortion_level * 0.1 * torch.randn(1, 3, 1, 1).to(device), 0, 1)
    
    # 5. Gaussian blur (scanner effect)
    kernel_size = int(3 + distortion_level * 4) * 2 + 1
    kernel_size = max(3, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    reconstructed = gaussian_blur(reconstructed, kernel_size=[kernel_size, kernel_size], 
                                 sigma=distortion_level)
    
    # Restore original dimensions
    if original_dim == 3:
        return reconstructed.squeeze(0)
    return reconstructed

# Wavelet consistency loss
class WaveletConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, stego, cover):
        stego_wave = wavelet_transform(stego)
        cover_wave = wavelet_transform(cover)
        
        # Focus on robust bands (LL, LH, HL)
        loss = 0
        # LL bands (indices 0,1,2 for R,G,B)
        loss += self.mse(stego_wave[:, 0:3], cover_wave[:, 0:3])
        # LH bands (indices 3,4,5)
        loss += self.mse(stego_wave[:, 3:6], cover_wave[:, 3:6])
        # HL bands (indices 6,7,8)
        loss += self.mse(stego_wave[:, 6:9], cover_wave[:, 6:9])
        return loss / 3

# Initialize components
dataset = CoverDataset(data_dir)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

encoder = Encoder().to(device)
decoder = Decoder().to(device)
disc_spatial = SpatialDiscriminator().to(device)
disc_spectral = SpectralDiscriminator(in_channels=1).to(device)

# Losses
lpips_loss = LPIPSLoss().to(device)
hist_loss = ColorHistogramLoss().to(device)
contrast_loss = ContrastiveLoss().to(device)
bce_loss = BinaryCrossEntropyLoss().to(device)
freq_loss = FrequencyDomainLoss().to(device)
wavelet_loss = WaveletConsistencyLoss().to(device)

# Optimizers
optimizer_D = optim.Adam(list(disc_spatial.parameters()) + list(disc_spectral.parameters()), 
                        lr=1e-5, betas=(0.0, 0.9))
optimizer_enc = optim.Adam(encoder.parameters(), lr=lr, betas=(0.0, 0.9))
optimizer_dec = optim.Adam(decoder.parameters(), lr=lr_decoder, betas=(0.0, 0.9))

# Metrics
best_acc = 0.0
bce_log = []
acc_log = []
wavelet_loss_log = []

# Training loop
for epoch in range(1, epochs+1):
    epoch_bce = []
    epoch_acc = []
    epoch_wavelet = []
    
    for i, (cover_imgs, msg_bits) in enumerate(loader):
        cover_imgs = cover_imgs.to(device)
        msg_bits = msg_bits.to(device)
        batch_size_curr = cover_imgs.size(0)

        # Generate clean stego
        stego_clean = encoder(cover_imgs, msg_bits)
        
        # Apply print-scan simulation with variable intensity
        stego_imgs_ps = []
        for img in stego_clean:
            if random.random() < DISTORTION_PROB:
                # Random distortion level
                level = random.choice(DISTORTION_LEVELS)
                tensor_img = robust_print_scan(img, distortion_level=level)
            else:
                tensor_img = img
            stego_imgs_ps.append(tensor_img)
        stego_imgs_ps = torch.stack(stego_imgs_ps)
        
        # Decode from mixed batch (clean + distorted)
        decoded_logits = decoder(stego_imgs_ps)
        
        # Update discriminators
        optimizer_D.zero_grad()
        
        # Spatial discriminator (use clean stego only)
        real_score = disc_spatial(cover_imgs)
        fake_score = disc_spatial(stego_clean.detach())
        loss_D_spat = fake_score.mean() - real_score.mean()
        
        # Gradient penalty
        epsilon = torch.rand(batch_size_curr, 1, 1, 1, device=device)
        interp = epsilon * cover_imgs + (1 - epsilon) * stego_clean.detach()
        interp.requires_grad_(True)
        interp_score = disc_spatial(interp)
        grad = torch.autograd.grad(outputs=interp_score.sum(), inputs=interp, 
                                create_graph=True, retain_graph=True)[0]
        grad_norm = grad.view(grad.size(0), -1).norm(2, dim=1)
        grad_penalty = ((grad_norm - 1)**2).mean()
        loss_gp_spat = 10.0 * grad_penalty
        
        # Spectral discriminator
        orig_msg_img = msg_bits.to(device)
        fake_msg_img = torch.sigmoid(decoded_logits.detach())
        real_score_m = disc_spectral(orig_msg_img)
        fake_score_m = disc_spectral(fake_msg_img)
        loss_D_spec = fake_score_m.mean() - real_score_m.mean()
        
        # Spectral gradient penalty
        epsilon_m = torch.rand(batch_size_curr, 1, 1, 1, device=device)
        interp_msg = (epsilon_m * orig_msg_img + (1 - epsilon_m) * fake_msg_img).detach()
        interp_msg.requires_grad_(True)
        interp_msg_score = disc_spectral(interp_msg)
        grad_m = torch.autograd.grad(outputs=interp_msg_score.sum(), inputs=interp_msg,
                                  create_graph=True, retain_graph=True)[0]
        grad_m_norm = grad_m.view(grad_m.size(0), -1).norm(2, dim=1)
        grad_penalty_m = ((grad_m_norm - 1)**2).mean()
        loss_gp_spec = 10.0 * grad_penalty_m
        
        # Total discriminator loss
        loss_D_total = loss_D_spat + loss_gp_spat + loss_D_spec + loss_gp_spec
        loss_D_total.backward()
        optimizer_D.step()
        
        # Update generator (encoder + decoder)
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        
        # Loss components
        # Use clean stego for encoder losses
        loss_perceptual = lpips_loss(stego_clean, cover_imgs)
        loss_color = hist_loss(stego_clean, cover_imgs)
        loss_freq = freq_loss(stego_clean, cover_imgs)
        loss_wave = wavelet_loss(stego_clean, cover_imgs)  # New wavelet loss
        
        adv_spatial_score = disc_spatial(stego_clean)
        loss_adv_spatial = -adv_spatial_score.mean()
        
        # Use decoded results from mixed batch for decoder losses
        loss_bce = bce_loss(decoded_logits, msg_bits.to(device))
        loss_contrast = contrast_loss(decoded_logits, msg_bits.to(device))
        
        adv_spectral_score = disc_spectral(torch.sigmoid(decoded_logits))
        loss_adv_spectral = -adv_spectral_score.mean()
        
        # Total losses
        loss_enc = (lambda_color * loss_color + 
                  lambda_lpips * loss_perceptual + 
                  lambda_adv_spatial * loss_adv_spatial +
                  lambda_freq * loss_freq +
                  lambda_wavelet * loss_wave)  # Include wavelet loss
        
        loss_dec = (lambda_bce * loss_bce + 
                  lambda_contrast * loss_contrast + 
                  lambda_adv_spectral * loss_adv_spectral)
        
        # Backpropagate with gradient clipping
        loss_dec.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer_dec.step()
        
        loss_enc.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer_enc.step()
        
        # Metrics (32x32)
        with torch.no_grad():
            pred_probs = torch.sigmoid(decoded_logits)
            pred_bits = (pred_probs >= 0.5).float()
            bit_accuracy = (pred_bits == msg_bits).float().mean().item()
        
        print(f"[Epoch {epoch}|{epochs}][Batch {i+1}|{len(loader)}] "
              f"Enc: {loss_enc.item():.4f} | Dec: {loss_dec.item():.4f} | "
              f"D: {loss_D_total.item():.4f} | BCE: {loss_bce.item():.4f} | "
              f"Wave: {loss_wave.item():.4f} | "
              f"Acc: {bit_accuracy*100:.2f}%")
        
        # Log metrics
        epoch_bce.append(loss_bce.item())
        epoch_acc.append(bit_accuracy)
        epoch_wavelet.append(loss_wave.item())
    
    # Epoch logging
    avg_bce = np.mean(epoch_bce)
    avg_acc = np.mean(epoch_acc)
    avg_wavelet = np.mean(epoch_wavelet)
    
    bce_log.append(avg_bce)
    acc_log.append(avg_acc)
    wavelet_loss_log.append(avg_wavelet)
    
    # Save best model
    if avg_acc >= best_acc:
        best_acc = avg_acc
        torch.save(encoder.state_dict(), "best_encoder.pth")
        torch.save(decoder.state_dict(), "best_decoder.pth")
        print(f"✅ Saved best model ({avg_acc*100:.2f}% acc)")
    
    # Save checkpoint every 10 epochs
    if epoch % 100 == 0:
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'disc_spatial_state_dict': disc_spatial.state_dict(),
            'disc_spectral_state_dict': disc_spectral.state_dict(),
            'optimizer_enc_state_dict': optimizer_enc.state_dict(),
            'optimizer_dec_state_dict': optimizer_dec.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'best_acc': best_acc,
            'bce_log': bce_log,
            'acc_log': acc_log,
            'wavelet_loss_log': wavelet_loss_log
        }, f"checkpoint_epoch_{epoch}.pth")

# Final save and plotting
torch.save(encoder.state_dict(), "final_encoder.pth")
torch.save(decoder.state_dict(), "final_decoder.pth")

# Plot training metrics
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(acc_log, label="Bit Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(bce_log, label="BCE Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(wavelet_loss_log, label="Wavelet Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_metrics.png")
print("✅ Training complete and metrics saved")