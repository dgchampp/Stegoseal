import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
import os


# Add models directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import robust components
from models.decoder import RobustDecoder as Decoder
from models.preprocessing import wavelet_transform, gradient_transform

# === CONFIGURATION ===
decoder_weights = "best_decoder.pth"
stego_path = "/home/rio/Desktop/day3 (Working)/deep2/examples/stego.png"
message_path = "original_message_32x32.pt"

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.ToTensor(),
    # Ensure image size matches training (256x256)
    transforms.Resize((256, 256), antialias=True)
])

# === LOAD STEGO IMAGE ===
stego_img = Image.open(stego_path).convert("RGB")
stego_tensor = transform(stego_img).unsqueeze(0).to(device)

# === LOAD ROBUST DECODER ===
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(decoder_weights, map_location=device,weights_only=True))
decoder.eval()

# === DECODE ===
with torch.no_grad():
    decoded_logits = decoder(stego_tensor)
    decoded_probs = torch.sigmoid(decoded_logits)
    decoded_bits = (decoded_probs >= 0.5).float().cpu().flatten()

# === DISPLAY DECODED MESSAGE ===
decoded_bit_string = "".join(str(int(b)) for b in decoded_bits)
decoded_hex_string = "%0*X" % ((len(decoded_bit_string) + 3) // 4, int(decoded_bit_string, 2))
print(f"[+] Decoded Message (hex): {decoded_hex_string}")
print(f"[+] Message Length: {len(decoded_bit_string)} bits")

# === LOAD ORIGINAL MESSAGE ===
try:
    original_bits = torch.load(message_path,weights_only=True).flatten()
    correct = (decoded_bits == original_bits).sum().item()
    total = original_bits.numel()
    accuracy = correct / total * 100
    print(f"[\u2713] Bit Accuracy: {accuracy:.2f}%")
except FileNotFoundError:
    print("[!] Original message file not found. Skipping accuracy check.")
    original_bits = None

# === VISUALIZE MESSAGE ===
plt.figure(figsize=(10, 5))

# Decoded message
plt.subplot(1, 2, 1)
plt.imshow(decoded_bits.reshape(32, 32).numpy(), cmap='gray')
plt.title('Decoded Message')

# Original message if available
if original_bits is not None:
    plt.subplot(1, 2, 2)
    plt.imshow(original_bits.reshape(32, 32).numpy(), cmap='gray')
    plt.title('Original Message')

plt.tight_layout()
plt.savefig('message_comparison.png')
print("[+] Saved message comparison to 'message_comparison.png'")

# Print stego image metrics
if original_bits is not None:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    
    # Load original cover image for comparison
    cover_img = Image.open("/home/rio/Desktop/day3 (Working)/deep2/examples/test.jpg").convert("RGB")
    cover_tensor = transform(cover_img).unsqueeze(0)
    
    # Calculate metrics
    psnr_val = psnr(stego_tensor.cpu(), cover_tensor)
    ssim_val = ssim(stego_tensor.cpu(), cover_tensor)
    
    print(f"[+] Stego Quality - PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")