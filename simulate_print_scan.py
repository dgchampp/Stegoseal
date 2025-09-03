import torch
import torch.nn.functional as F
from preprocessing import wavelet_transform, inverse_wavelet_transform, get_wavelet_bands

def simulate_print_scan(image, distortion_level=0.8, device='cuda'):
    """Enhanced print-scan simulation with frequency targeting"""
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
    return F.gaussian_blur(reconstructed, kernel_size, sigma=distortion_level)