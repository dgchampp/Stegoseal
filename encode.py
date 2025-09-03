import torch
from PIL import Image
from torchvision import transforms
import sys
import os

# Add models directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import robust components
from models.encoder import RobustEncoder as Encoder
from models.preprocessing import wavelet_transform

# === CONFIGURATION ===
cover_image_path = "/home/rio/Desktop/day3 (Working)/deep2/examples/test.jpg"
encoder_weights = "best_encoder.pth"
stego_output_path = "/home/rio/Desktop/day3 (Working)/deep2/examples/stego.png"

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.ToTensor(),
    # Ensure image size matches training (256x256)
    transforms.Resize((256, 256), antialias=True)
])
to_pil = transforms.ToPILImage()

# === LOAD COVER IMAGE ===
cover_img = Image.open(cover_image_path).convert('RGB')
cover_tensor = transform(cover_img).unsqueeze(0).to(device)  # (1, 3, 256, 256)

# === GENERATE RANDOM 32x32 MESSAGE ===
message_bits = torch.randint(0, 2, (1, 1, 32, 32), dtype=torch.float32).to(device)
original_bits_flat = message_bits.flatten().cpu()
bit_string = "".join(str(int(b)) for b in original_bits_flat)
hex_string = "%0*X" % ((len(bit_string) + 3) // 4, int(bit_string, 2))
print(f"[+] Embedded Message (hex): {hex_string}")
print(f"[+] Message Length: {len(bit_string)} bits")

# === LOAD ROBUST ENCODER ===
encoder = Encoder().to(device)
encoder.load_state_dict(torch.load(encoder_weights, map_location=device,weights_only=True))
encoder.eval()

# === ENCODE ===
with torch.no_grad():
    stego_tensor = encoder(cover_tensor, message_bits).clamp(0, 1).cpu()
    stego_img = to_pil(stego_tensor.squeeze(0))
    stego_img.save(stego_output_path)
    print(f"[+] Stego image saved to {stego_output_path}")

# === SAVE MESSAGE BITS FOR DECODING ===
torch.save(message_bits.cpu(), "original_message_32x32.pt")

print("[+] Running automatic decoding verification...")
# Automatically run decode.py after encoding
with open('decode.py') as f:
    exec(f.read())