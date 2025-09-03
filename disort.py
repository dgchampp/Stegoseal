import torch
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

# Add current directory to path for model imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import robust components
from models.encoder import RobustEncoder as Encoder
from models.decoder import RobustDecoder as Decoder
from models.preprocessing import wavelet_transform, gradient_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_pil = transforms.ToPILImage()

# ===== VISUALIZATION FUNCTION =====
def visualize_comparison(original, encoded, decoded_bits, original_bits_unflat, output_path):
    """Enhanced visualization with frequency-domain analysis"""
    decoded_bits = torch.tensor(decoded_bits) if not isinstance(decoded_bits, torch.Tensor) else decoded_bits
    original_bits_flat = original_bits_unflat.flatten()

    min_length = min(len(decoded_bits), len(original_bits_flat))
    decoded_bits = decoded_bits[:min_length]
    original_bits_flat = original_bits_flat[:min_length]
    error_map = (decoded_bits != original_bits_flat).reshape(32, 32)

    fig = plt.figure(figsize=(18, 12))

    # Original and stego images
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=1)
    ax1.imshow(original)
    ax1.set_title("Original Cover")
    ax1.axis("off")

    ax2 = plt.subplot2grid((3, 4), (0, 1), colspan=1)
    ax2.imshow(encoded)
    ax2.set_title("Encoded Stego")
    ax2.axis("off")

    # Difference visualization
    ax3 = plt.subplot2grid((3, 4), (0, 2), colspan=1)
    diff = np.abs(np.array(encoded) - np.array(original))
    diff_vis = np.clip(diff * 10, 0, 1)  # Amplify differences
    ax3.imshow(diff_vis, cmap='viridis')
    ax3.set_title("Embedding Difference (10x amplified)")
    ax3.axis("off")

    # Frequency domain visualization
    ax4 = plt.subplot2grid((3, 4), (0, 3), colspan=1)
    stego_tensor = transforms.ToTensor()(encoded).unsqueeze(0).to(device)
    stego_fft = torch.fft.fftshift(torch.fft.fft2(stego_tensor, dim=(2, 3)))
    mag = torch.log(1 + torch.abs(stego_fft))
    mag = mag.squeeze(0).permute(1, 2, 0).mean(dim=2).cpu().numpy()  # Convert to grayscale
    ax4.imshow(mag, cmap='inferno')
    ax4.set_title("Stego Frequency Spectrum")
    ax4.axis("off")

    # Message visualization
    ax5 = plt.subplot2grid((3, 4), (1, 0), colspan=2)
    ax5.imshow(original_bits_unflat.squeeze().cpu().numpy().reshape(32, 32), cmap='gray')
    ax5.set_title("Original Message (32x32)")
    ax5.axis("off")

    ax6 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
    ax6.imshow(decoded_bits.reshape(32, 32).cpu().numpy(), cmap='gray')
    ax6.set_title("Decoded Message")
    ax6.axis("off")

    # Error map
    ax7 = plt.subplot2grid((3, 4), (2, 0), colspan=2)
    ax7.imshow(error_map.cpu().numpy(), cmap='hot', interpolation='nearest')
    ax7.set_title(f"Bit Error Map (Errors: {error_map.sum().item()}/{32*32})")
    ax7.axis("off")

    # Wavelet band visualization
    ax8 = plt.subplot2grid((3, 4), (2, 2), colspan=2)
    cover_tensor = transforms.ToTensor()(original).unsqueeze(0).to(device)
    wavelet_coeffs = wavelet_transform(cover_tensor)
    # Visualize LL band (low frequency)
    ll_band = wavelet_coeffs[0, :3].permute(1, 2, 0).cpu().numpy()
    # Normalize for display
    ll_band = (ll_band - ll_band.min()) / (ll_band.max() - ll_band.min())
    ax8.imshow(ll_band)
    ax8.set_title("Cover LL Wavelet Band")
    ax8.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ Saved full comparison to: {output_path}")

# ===== MAIN WORKFLOW =====
if __name__ == "__main__":
    original_cover_path = "/home/nitk/Desktop/day3/deep2/examples/00049.jpg"
    base_dir = os.path.dirname(original_cover_path) if os.path.isabs(original_cover_path) else os.path.dirname(os.path.abspath(original_cover_path))
    file_name = os.path.splitext(os.path.basename(original_cover_path))[0]

    stego_path = os.path.join(base_dir, "robust_stego.png")
    comparison_output = os.path.join(base_dir, "robust_comparison.png")
    message_path = os.path.join(base_dir, "robust_message.pt")

    encoder_weights = "best_encoder.pth"
    decoder_weights = "best_decoder.pth"

    # Verify paths
    if not os.path.exists(original_cover_path):
        print(f"❌ Original cover image not found: {original_cover_path}")
        exit(1)

    for weight in [encoder_weights, decoder_weights]:
        if not os.path.exists(weight):
            print(f"❌ Model weights not found: {weight}")
            exit(1)

    # Fixed transforms with proper interpolation and antialiasing
    transform = transforms.Compose([
        transforms.Resize((256, 256), 
                      interpolation=InterpolationMode.BILINEAR,
                      antialias=True),
        transforms.ToTensor()
    ])

    try:
        # Load and process cover image
        cover_img = Image.open(original_cover_path).convert("RGB")
        print(f"✅ Loaded cover image: {original_cover_path}")
        
        # Apply transformations
        cover_tensor = transform(cover_img).to(device)
        cover_pil = to_pil(cover_tensor.cpu())

        # Generate 32x32 message (1024 bits)
        message_bits = torch.randint(0, 2, (1, 1, 32, 32), dtype=torch.float32).to(device)
        torch.save(message_bits.cpu(), message_path)
        original_bits_unflat = message_bits
        print("✅ Generated random 32x32 message")

        # Load robust encoder
        encoder = Encoder().to(device)
        encoder.load_state_dict(torch.load(encoder_weights, map_location=device,weights_only=True))
        encoder.eval()
        print(f"✅ Loaded robust encoder weights: {encoder_weights}")

        # Encode message
        with torch.no_grad():
            stego_tensor = encoder(cover_tensor.unsqueeze(0), message_bits).clamp(0, 1).squeeze(0)
        stego_pil = to_pil(stego_tensor.cpu())
        stego_pil.save(stego_path)
        print(f"✅ Saved robust stego image to: {stego_path}")

        # Load robust decoder
        decoder = Decoder().to(device)
        decoder.load_state_dict(torch.load(decoder_weights, map_location=device,weights_only=True))
        decoder.eval()
        print(f"✅ Loaded robust decoder weights: {decoder_weights}")

        # Decode message
        with torch.no_grad():
            decoded_logits = decoder(stego_tensor.unsqueeze(0))
            decoded_probs = torch.sigmoid(decoded_logits)
            decoded_bits = (decoded_probs >= 0.5).float().cpu().flatten()
        print("✅ Decoded message from robust stego image")

        # Calculate accuracy
        original_bits = message_bits.flatten().cpu()
        min_length = min(len(decoded_bits), len(original_bits))
        decoded_bits = decoded_bits[:min_length]
        original_bits = original_bits[:min_length]

        correct = (decoded_bits == original_bits).sum().item()
        total = min_length
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"[✓] Bit Accuracy: {accuracy:.2f}% ({correct}/{total} correct bits)")

        # Calculate PSNR and SSIM
        try:
            psnr_val = peak_signal_noise_ratio(
                stego_tensor.unsqueeze(0).cpu(), 
                cover_tensor.unsqueeze(0).cpu()
            )
            
            ssim_val = structural_similarity_index_measure(
                stego_tensor.unsqueeze(0).cpu(), 
                cover_tensor.unsqueeze(0).cpu()
            )
            
            print(f"[✓] Quality Metrics - PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
        except Exception as metric_error:
            print(f"⚠️ Could not calculate quality metrics: {metric_error}")

        # Enhanced visualization
        visualize_comparison(
            cover_pil,
            stego_pil,
            decoded_bits,
            original_bits_unflat.cpu(),
            comparison_output
        )

        print("✅ Robust pipeline execution complete!")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()