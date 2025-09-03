import torch
import torch.nn.functional as F

def gradient_transform(image: torch.Tensor) -> torch.Tensor:
    """Compute gradient maps for any number of channels"""
    # Handle input dimensions
    original_dim = image.dim()
    if original_dim == 3:
        image = image.unsqueeze(0)  # add batch dimension
    
    # Get number of channels
    num_channels = image.size(1)
    
    # Define Sobel filters
    sobel_x = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32, device=image.device)
    sobel_y = sobel_x.t().to(image.device)
    
    # Create weights for all channels
    weight = torch.zeros((2 * num_channels, 1, 3, 3), device=image.device)
    for i in range(num_channels):
        weight[2*i, 0, :, :] = sobel_x
        weight[2*i+1, 0, :, :] = sobel_y
    
    # Apply convolution
    grad_maps = F.conv2d(image, weight, stride=1, padding=1, groups=num_channels)
    
    # Compute gradient magnitude for each channel
    N, _, H, W = grad_maps.shape
    grad = torch.empty((N, num_channels, H, W), device=image.device)
    for i in range(num_channels):
        gx = grad_maps[:, 2*i, :, :]
        gy = grad_maps[:, 2*i+1, :, :]
        grad[:, i, :, :] = torch.sqrt(gx * gx + gy * gy + 1e-8)
    
    # Restore original dimensions
    if original_dim == 3:
        return grad.squeeze(0)
    return grad

def wavelet_transform(image: torch.Tensor) -> torch.Tensor:
    """Apply Haar wavelet transform (1-level) with dimension handling"""
    # Handle different input dimensions
    original_dim = image.dim()
    if original_dim == 3:  # (C, H, W)
        image = image.unsqueeze(0)  # add batch dimension
    elif original_dim == 5:  # (B, 1, C, H, W)
        image = image.squeeze(1)  # remove extra dimension
    
    # Check and ensure 4D input
    if image.dim() != 4:
        raise ValueError(f"Input must be 3D, 4D or 5D (with extra singleton dim), got {original_dim}D")
    
    # Sobel horizontal and vertical filters
    low = torch.tensor([0.5, 0.5], dtype=torch.float32, device=image.device)
    high = torch.tensor([0.5, -0.5], dtype=torch.float32, device=image.device)
    
    # Construct 2D filter kernels
    ll = torch.outer(low, low)
    lh = torch.outer(low, high)
    hl = torch.outer(high, low)
    hh = torch.outer(high, high)
    
    # Stack filters for one channel
    filters = torch.stack([ll, lh, hl, hh], dim=0)  # (4, 2, 2)
    
    # Repeat for 3 input channels (groups=3 convolution)
    weight = torch.cat([filters, filters, filters], dim=0)  # (12, 2, 2)
    weight = weight.unsqueeze(1)  # (12, 1, 2, 2)
    
    # Apply convolution with stride=2 (downsampling)
    coeffs = F.conv2d(image, weight.to(image.device), stride=2, groups=3)
    
    # Restore original dimensions
    if original_dim == 3:
        return coeffs.squeeze(0)  # remove batch dimension
    return coeffs

# ... (other functions remain unchanged) ...
def inverse_wavelet_transform(coeffs: torch.Tensor) -> torch.Tensor:
    """Inverse Haar wavelet transform with proper channel handling"""
    if coeffs.dim() == 3:
        coeffs = coeffs.unsqueeze(0)
        
    # Define inverse 1D Haar filters
    low = torch.tensor([1., 1.], dtype=torch.float32, device=coeffs.device)
    high = torch.tensor([1., -1.], dtype=torch.float32, device=coeffs.device)
    
    # Construct 2D filter kernels
    ll = torch.outer(low, low) * 0.25
    lh = torch.outer(low, high) * 0.25
    hl = torch.outer(high, low) * 0.25
    hh = torch.outer(high, high) * 0.25
    
    # Create filter bank for one channel (4 input bands to 1 output channel)
    filters = torch.stack([ll, lh, hl, hh], dim=0)  # (4, 2, 2)
    
    # Prepare weight for 3-channel processing
    weight = torch.cat([filters, filters, filters], dim=0)  # (12, 2, 2)
    weight = weight.unsqueeze(1)  # (12, 1, 2, 2)
    
    # Apply transposed convolution for inverse transform
    return F.conv_transpose2d(coeffs, weight.to(coeffs.device), 
                             stride=2, padding=0, groups=3)

def get_wavelet_bands(wavelet: torch.Tensor) -> tuple:
    """Separate wavelet coefficients into frequency bands"""
    # Bands: [LL, LH, HL, HH] for each channel
    bands = []
    for i in range(3):  # For each color channel
        start = i*4
        bands.append(wavelet[:, start:start+1])    # LL
        bands.append(wavelet[:, start+1:start+2])  # LH
        bands.append(wavelet[:, start+2:start+3])  # HL
        bands.append(wavelet[:, start+3:start+4])  # HH
    return bands