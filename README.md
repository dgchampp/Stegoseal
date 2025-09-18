# 🕵️‍♂️ Stegoseal

**Stegoseal** is a deep learning–based steganography system that hides messages in images in a way that is robust to real-world distortions such as **printing and scanning**.  

---

## 📂 Project Structure

```text
Stegoseal/
├── examples/                      # Sample input/output images
├── losses/                        # Loss functions and plots
├── models/                        # Saved model checkpoints
│   ├── best_encoder.pth
│   ├── best_decoder.pth
├── encode.py                      # Script to embed a message into an image
├── decode.py                      # Script to extract a message from an image
├── train.py / train1.py           # Model training scripts
├── simulate_print_scan.py          # Emulate print–scan distortions
├── curriculum.py                  # Curriculum learning logic
├── model.py                       # Core network architectures
├── brightness_robustness_test.png # Example robustness test (brightness)
├── gaussian_noise_robustness_test_vertical.png
├── message_comparison.png         # Visual comparison (original vs. encoded)
├── original_message_32x32.pt      # Sample test message
└── README.md
```
✨ Features
🔒 Neural steganography – hides information inside images.
🖨️ Printer & scanner-proof – training includes real-world distortion simulation.
📈 Curriculum learning – gradually increases difficulty for stable training.
📊 Robustness visualization – test resistance to noise, brightness, etc.

⚙️ Installation
Clone the repository and install dependencies:

git clone https://github.com/dgchampp/Stegoseal.git
cd Stegoseal
pip install torch torchvision numpy pillow
(Consider creating a virtual environment to keep things clean.)

🚀 Usage
Train the Model
```bash
python train.py
```
# or
```bash
python train1.py
```
Encode a Message:
```bash
python encode.py --input input.png --message secret.bin --output encoded.png
```
Decode a Message:
```bash
python decode.py --input encoded.png --output recovered_message.bin
```
Simulate Print–Scan Distortion
```bash
python simulate_print_scan.py --input encoded.png --output scanned.png
```
🖼️ Visual Examples
```test
these test images are in the repo : 
Brightness Robustness
Gaussian Noise Robustness
Message Comparison```
