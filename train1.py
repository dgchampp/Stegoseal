import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Import the model components from model.py
from model import StampOneEncoder, StampOneDecoder, SteganographyDiscriminator, SpectralDiscriminator

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class for images in a directory
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = []
        exts = ('.png', '.jpg', '.jpeg', '.bmp')
        for root, _, files in os.walk(folder_path):
            for fname in files:
                if fname.lower().endswith(exts):
                    self.image_files.append(os.path.join(root, fname))
        self.transform = transform
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Path to training images
train_dir = "/home/rio/Desktop/1024 (Copy)/data/test"
# Transformation: resize to 256x256 and convert to tensor in [0,1]
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
# Load dataset
dataset = ImageFolderDataset(train_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True)

# Initialize models
encoder = StampOneEncoder().to(device)
decoder = StampOneDecoder().to(device)
disc_image = SteganographyDiscriminator().to(device)
disc_spectral = SpectralDiscriminator().to(device)

# Optimizers
optimizer_gen = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)
optimizer_disc_image = torch.optim.Adam(disc_image.parameters(), lr=0.0001)
optimizer_disc_spectral = torch.optim.Adam(disc_spectral.parameters(), lr=0.0001)

# Loss weights (from paper)
lambda_color = 1.0
lambda_perceptual = 2.0
lambda_stegDisc = 1.0
lambda_ce = 1.0
lambda_qs = 1.0
lambda_specDisc = 1.0

# If LPIPS library is available, initialize it for perceptual loss
lpips_loss = None
try:
    import lpips
    lpips_loss = lpips.LPIPS(net='vgg').to(device)
except ImportError:
    print("LPIPS library not found. Perceptual loss will be approximate.")
    # Optionally: use a pretrained VGG for approximate perceptual loss if available
    from torchvision.models import vgg16, VGG16_Weights
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).to(device)
    vgg.eval()
    # Choose some intermediate layers for features (for approximation)
    feat_layers = [0, 5, 10, 17, 24]  # conv layers from VGG

# Helper: compute color histogram loss in log-chroma space
def color_histogram_loss(img1, img2, num_bins=30):
    # img1, img2: tensors of shape (N,3,H,W) in [0,1]
    # Convert to log-chromaticity (u = log(R/G), v = log(B/G))
    eps = 1e-6
    R1, G1, B1 = img1[:,0,:,:] + eps, img1[:,1,:,:] + eps, img1[:,2,:,:] + eps
    R2, G2, B2 = img2[:,0,:,:] + eps, img2[:,1,:,:] + eps, img2[:,2,:,:] + eps
    u1 = torch.log10(R1) - torch.log10(G1)
    v1 = torch.log10(B1) - torch.log10(G1)
    u2 = torch.log10(R2) - torch.log10(G2)
    v2 = torch.log10(B2) - torch.log10(G2)
    # Histogram (differentiable approximation using Gaussian kernel for each bin)
    # Define bin centers
    bin_centers = torch.linspace(-3, 3, steps=num_bins, device=img1.device)
    sigma = (bin_centers[1] - bin_centers[0]).item() * 0.5
    # Compute hist for u and v separately
    def compute_hist(vals):
        # vals: (N,H,W) values
        N, H, W = vals.shape
        vals_flat = vals.reshape(N, -1)  # shape (N, H*W)
        # Expand dims for bin centers
        diff = vals_flat.unsqueeze(-1) - bin_centers  # (N, pixels, num_bins)
        # Gaussian weights for each bin
        weights = torch.exp(-0.5 * (diff / sigma) ** 2)  # (N, pixels, num_bins)
        # Sum over pixels
        hist = weights.sum(dim=1)  # (N, num_bins)
        # Normalize histogram (so differences not scale with image size)
        hist = hist / (H * W)
        return hist
    hist_u1 = compute_hist(u1)
    hist_v1 = compute_hist(v1)
    hist_u2 = compute_hist(u2)
    hist_v2 = compute_hist(v2)
    # Compute Euclidean distance between histograms
    loss_u = F.mse_loss(hist_u1, hist_u2, reduction='mean')
    loss_v = F.mse_loss(hist_v1, hist_v2, reduction='mean')
    return (loss_u + loss_v)

# Prepare contrastive learning (QS-Attn) projector for decoder features
proj_dim = 128
proj_net = torch.nn.Linear(16*16*3, proj_dim).to(device)  # project flattened 768-dim message to 128-dim
# Training loop
num_epochs = 100  # adjust as needed
best_bit_acc = 0.0

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    disc_image.train()
    disc_spectral.train()
    running_correct_bits = 0
    total_bits = 0
    for images in dataloader:
        images = images.to(device)
        N = images.size(0)
        # Generate random 256-bit message for each image
        # shape (N, 256), values 0 or 1
        message_bits = torch.randint(0, 2, (N, 256), dtype=torch.float32, device=device)
        # Reshape to 16x16 and create 3-channel message image
        message_img = message_bits.view(N, 1, 16, 16)
        message_img = message_img.repeat(1, 3, 1, 1)  # duplicate across 3 channels for RGB format
        # Forward pass: Encode and then decode
        encoded = encoder(images, message_img)  # encoded stego image
        # Simulate noise perturbations (noise simulation pipeline)
        # We apply a random selection of distortions to the encoded image to improve robustness
        enc_noisy = encoded.detach()  # avoid gradients through simulated noise
        # JPEG compression simulation (by saving and reloading with quality loss)
        if torch.rand(1).item() < 0.3:
            # Convert to PIL and back with lower quality
            enc_noisy_cpu = (enc_noisy.clamp(0,1) * 255).cpu().detach().to(torch.uint8)
            new_imgs = []
            for i in range(enc_noisy_cpu.size(0)):
                im = transforms.ToPILImage()(enc_noisy_cpu[i])
                # Save to JPEG in memory
                buffer = bytes()
                from io import BytesIO
                buffer_io = BytesIO()
                quality = torch.randint(40, 80, ()).item()
                im.save(buffer_io, format='JPEG', quality=quality)
                buffer_io.seek(0)
                im_jpeg = Image.open(buffer_io).convert('RGB')
                im_jpeg = transform(im_jpeg)  # apply same transform (resize, to tensor)
                new_imgs.append(im_jpeg)
            enc_noisy = torch.stack(new_imgs).to(device)
        # Gaussian noise
        if torch.rand(1).item() < 0.5:
            noise_std = 0.01 * torch.rand(1).item()
            enc_noisy = enc_noisy + noise_std * torch.randn_like(enc_noisy)
            enc_noisy = enc_noisy.clamp(0, 1)
        # Random brightness/contrast/hue adjustments
        if torch.rand(1).item() < 0.5:
            # Random brightness and contrast
            brightness_factor = 1 + 0.2 * (2*torch.rand(1).item()-1)
            contrast_factor = 1 + 0.2 * (2*torch.rand(1).item()-1)
            enc_noisy = transforms.functional.adjust_brightness(transforms.functional.adjust_contrast(enc_noisy, contrast_factor), brightness_factor)
        # Random affine (small rotation/translation)
        if torch.rand(1).item() < 0.3:
            angle = torch.normal(0.0, 5.0, ()).item()  # small rotation
            translations = (torch.randn(1).item()*0.02, torch.randn(1).item()*0.02)
            enc_noisy = transforms.functional.affine(enc_noisy, angle=angle, translate=(int(translations[0]*256), int(translations[1]*256)), scale=1.0, shear=0)
        # Dithering simulation (approximate by adding quantization noise)
        if torch.rand(1).item() < 0.2:
            levels = 8
            enc_noisy = torch.round(enc_noisy * levels) / levels + (torch.rand_like(enc_noisy) - 0.5) / levels
            enc_noisy = enc_noisy.clamp(0, 1)
        # Continue with decoder using the noisy encoded image
        recovered_msg = decoder(enc_noisy)
        # Ground truth message image (target) is message_img (N,3,16,16) with 0/1
        target_msg = message_img  # (N,3,16,16)
        # Compute losses
        # 1. Encoder loss components
        # Color histogram loss
        loss_color = color_histogram_loss(images, encoded)
        # Perceptual (LPIPS) loss
        if lpips_loss is not None:
            loss_lpips = lpips_loss(images, encoded).mean()
        else:
            # Fallback: use simple VGG-based perceptual loss (just an approximation)
            # Extract features from a few layers
            x_orig = images
            x_enc = encoded
            loss_vals = 0.0
            for layer_idx in [0, 5, 10, 17, 24]:
                x_orig = vgg.features[:layer_idx+1](x_orig)
                x_enc = vgg.features[:layer_idx+1](x_enc)
                loss_vals += F.mse_loss(x_enc, x_orig, reduction='mean')
            loss_lpips = loss_vals
        # Steganography discriminator loss (for generator) - WGAN adversarial: want disc_image(encoded) ~ disc_image(original)
        disc_img_out_fake = disc_image(encoded)
        # Use Wasserstein adversarial loss: l_SD = -E[disc_fake] (generator tries to maximize disc output for fake)
        loss_steg_adv = - torch.mean(disc_img_out_fake)
        # Encoder total loss
        loss_encoder = lambda_color * loss_color + lambda_perceptual * loss_lpips + lambda_stegDisc * loss_steg_adv
        # 2. Decoder loss components
        # Cross-entropy loss for message bits (treat as binary classification, use BCE)
        # Use sigmoid on recovered message for probability
        recovered_probs = torch.sigmoid(recovered_msg)
        loss_ce = F.binary_cross_entropy(recovered_probs, target_msg)
        # QS-Attn contrastive loss
        # Project recovered and target messages to feature vectors
        rec_flat = recovered_msg.view(N, -1)
        tgt_flat = target_msg.view(N, -1)
        rec_proj = F.normalize(proj_net(rec_flat), dim=1)
        tgt_proj = F.normalize(proj_net(tgt_flat), dim=1)
        # Compute similarity matrix between rec_proj and tgt_proj
        sim_matrix = rec_proj @ tgt_proj.t() / 0.1  # dot products scaled by tau=0.1
        # Targets are diagonal (each recovered matches its own true message)
        labels = torch.arange(N, device=device)
        loss_qs = F.cross_entropy(sim_matrix, labels)
        # Spectral discriminator loss (for generator) - adversarial on message frequency
        disc_spec_out_fake = disc_spectral(recovered_msg)
        loss_spec_adv = - torch.mean(disc_spec_out_fake)
        # Decoder total loss
        loss_decoder = lambda_ce * loss_ce + lambda_qs * loss_qs + lambda_specDisc * loss_spec_adv
        # Combined generator (encoder+decoder) loss
        loss_gen_total = loss_encoder + loss_decoder
        # Backprop for generator
        optimizer_gen.zero_grad()
        loss_gen_total.backward()
        optimizer_gen.step()
        # Train discriminators
        # Steganography discriminator: maximize D(real) - D(fake) -> minimize negative
        disc_image_real = disc_image(images.detach())
        disc_image_fake = disc_image(encoded.detach())
        loss_disc_img = -(torch.mean(disc_image_real) - torch.mean(disc_image_fake))
        optimizer_disc_image.zero_grad()
        loss_disc_img.backward()
        optimizer_disc_image.step()
        # Spectral discriminator: similar approach on message images
        disc_spec_real = disc_spectral(target_msg.detach())
        disc_spec_fake = disc_spectral(recovered_msg.detach())
        loss_disc_spec = -(torch.mean(disc_spec_real) - torch.mean(disc_spec_fake))
        optimizer_disc_spectral.zero_grad()
        loss_disc_spec.backward()
        optimizer_disc_spectral.step()
        # Update running bit accuracy
        # Compute predicted bits from recovered_msg (threshold at 0.5)
        pred_bits = (recovered_probs.detach() >= 0.5).float()
        correct = torch.sum(pred_bits == target_msg).item()
        running_correct_bits += correct
        total_bits += target_msg.numel()
    # Epoch done, compute bit accuracy
    bit_accuracy = running_correct_bits / total_bits
    print(f"Epoch {epoch+1}/{num_epochs} - Bit Accuracy: {bit_accuracy*100:.2f}%")
    # Save best model
    if bit_accuracy > best_bit_acc:
        best_bit_acc = bit_accuracy
        save_path = "stampone_best.pth"
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict()
        }, save_path)
        print(f"Saved best model (epoch {epoch+1}, bit accuracy {bit_accuracy*100:.2f}%).")
