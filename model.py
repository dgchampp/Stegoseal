import torch
import torch.nn as nn
import torch.nn.functional as F

# Snake activation function (periodic activation) as defined in StampOne paper
class Snake(nn.Module):
    def __init__(self, channels):
        super(Snake, self).__init__()
        # Learnable parameter "a" for each channel (or feature)
        # Initialize to 0.5 as recommended
        self.a = nn.Parameter(torch.ones(channels) * 0.5)
    def forward(self, x):
        # Apply per-channel Snake activation: x + (1/a) * sin^2(a * x)
        # Handle both 2D (N,C) and 4D (N,C,H,W) inputs
        # Expand parameter dimensions if needed
        if x.dim() == 4:
            # shape (1,C,1,1) to broadcast over N,H,W
            a = self.a.view(1, -1, 1, 1)
        elif x.dim() == 2:
            # shape (1,C) to broadcast over batch dimension
            a = self.a.view(1, -1)
        else:
            a = self.a
        return x + (1.0 / a) * torch.sin(a * x) ** 2

# Depthwise layer: assign distinct weights to each channel (wavelet sub-band emphasis)
class Depthwise(nn.Module):
    def __init__(self, num_channels):
        super(Depthwise, self).__init__()
        # Depthwise conv with groups = num_channels, so each channel has an independent scalar weight and optional bias
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=1, groups=num_channels, bias=True)
    def forward(self, x):
        return self.conv(x)

# Spatial Transformer Network (STN) for correcting warping/rotation in decoder
class SpatialTransformer(nn.Module):
    def __init__(self, in_channels):
        super(SpatialTransformer, self).__init__()
        # Localization network to predict affine transformation parameters (2x3 matrix) for input feature map
        # Use small conv network followed by fully connected layers
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),  # conv -> reduce spatial size
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * (256 // 16) * (256 // 16), 64),  # 256/16 = 16 (assuming input H=W=256)
            nn.ReLU(inplace=True),
            nn.Linear(64, 6)  # 6 parameters for 2x3 affine matrix
        )
        # Initialize the affine transformation parameters to identity transform
        nn.init.zeros_(self.localization[-1].weight)
        nn.init.zeros_(self.localization[-1].bias)
        # Bias to identity: [1, 0, 0; 0, 1, 0] flattened -> (1,0,0,0,1,0)
        self.localization[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, x):
        # x is feature map of shape (N, C, H, W), with H=W=256 typically
        theta = self.localization(x)              # (N,6) affine parameters
        theta = theta.view(-1, 2, 3)             # (N,2,3)
        # Generate affine grid
        N, C, H, W = x.shape
        grid = F.affine_grid(theta, size=(N, C, H, W), align_corners=True)
        # Sample the input feature map according to grid to apply transformation
        x_transformed = F.grid_sample(x, grid, align_corners=True)
        return x_transformed

# Message Preparation Network (MPN) - variant D with dual branches and self-attention (attention gating)
class MessagePrepNetwork(nn.Module):
    def __init__(self, in_channels=15, message_size=256):
        super(MessagePrepNetwork, self).__init__()
        # Flatten input (16x16x15) to vector of length message_size*in_channels (should be 256*? Actually here message_size=256 bits, in_channels=15 sub-band channels)
        self.flatten = nn.Flatten()
        # Branch 1: Dense branch for global context (outputs gating signal)
        self.dense_gating = nn.Linear(16 * 16 * in_channels, 32)  # outputs 32-dim gating vector
        self.gating_act = Snake(32)  # Snake activation on gating vector
        # Branch 2: Convolutional branch to produce spatial feature map
        # First a linear layer to inflate message into spatial feature map of size 128x128 with smaller channel count (e.g. 16 channels)
        self.linear_to_map = nn.Linear(16 * 16 * in_channels, 128 * 128 * 16)
        # After reshaping to (16,128,128), apply 1x1 conv to increase channels to 64
        self.map_conv = nn.Conv2d(16, 64, kernel_size=1)
        # Activation for branch 2 output
        self.map_act = Snake(64)
        # Attention gating (self-attention block) to combine branch1 and branch2
        # 1x1 conv to reduce branch2 (skip) features to intermediate gating channels (32)
        self.attn_conv_skip = nn.Conv2d(64, 32, kernel_size=1)
        # 1x1 conv (as psi) to produce attention coefficients
        self.attn_psi = nn.Conv2d(32, 1, kernel_size=1)
        # Note: gating signal (branch1 output) will be added after broadcasting, no conv needed for it since it's already 32-dim
        # Upsampling layer to upscale feature map to 256x256 after attention
        # (We use simple interpolation here)
        # No learnable parameters in upsampling, we will use F.interpolate in forward
    def forward(self, message_features):
        """
        message_features: Tensor of shape (N, 15, 16, 16) - wavelet + gradient representation of message
        Returns: Tensor of shape (N, 64, 256, 256) - prepared message features for encoder input
        """
        N = message_features.shape[0]
        # Flatten input
        x_flat = self.flatten(message_features)  # shape (N, 3840) for 15*16*16
        # Branch 1: Dense for global signal
        gating_vec = self.dense_gating(x_flat)   # (N, 32)
        gating_vec = self.gating_act(gating_vec) # apply Snake activation
        # Branch 2: Linear to spatial map
        map_flat = self.linear_to_map(x_flat)    # (N, 128*128*16)
        # Reshape to (N,16,128,128)
        map_feat = map_flat.view(N, 16, 128, 128)
        # 1x1 conv to get 64-channel feature map
        map_feat = self.map_conv(map_feat)       # (N, 64, 128, 128)
        map_feat = self.map_act(map_feat)        # apply Snake on spatial features
        # Self-attention gating: use gating_vec (N,32) to refine map_feat (N,64,128,128)
        # Compute attention coefficients
        # Reduce skip features to 32 channels
        skip_down = self.attn_conv_skip(map_feat)            # (N, 32, 128, 128)
        # Prepare gating signal for broadcasting: (N,32) -> (N,32,1,1) -> broadcast to (N,32,128,128)
        gating = gating_vec.view(N, 32, 1, 1)
        gating_broadcast = gating.expand(-1, -1, skip_down.size(2), skip_down.size(3))  # (N,32,128,128)
        # Add skip and gating
        attn = skip_down + gating_broadcast
        attn = F.relu(attn, inplace=True)
        attn = self.attn_psi(attn)              # (N, 1, 128, 128)
        attn = torch.sigmoid(attn)              # attention mask in [0,1]
        # Apply attention mask to original skip features
        attended_map = map_feat * attn          # (N, 64, 128, 128), refined message features
        # Upsample the attended message feature map to 256x256
        message_out = F.interpolate(attended_map, scale_factor=2.0, mode='nearest')
        # Output is (N, 64, 256, 256)
        return message_out

# U-Shape network with Attention (Attention VNet)
class AttentionVNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super(AttentionVNet, self).__init__()
        # Encoder layers (downsampling path) - using conv stride 2 (no pooling)
        # We will create 4 downsampling layers (from 256 -> 128 -> 64 -> 32 -> 16)
        # Each down layer: Conv -> Conv (with possibly attention gating on skip after decoding)
        # Actually, incorporate skip attention on upsampling.
        # Define number of features at each level
        feat = base_channels
        # Level 0 (no downsample, 256x256)
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels, feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat, feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Level 1 (128x128)
        self.enc1 = nn.Sequential(
            nn.Conv2d(feat, feat*2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat*2, feat*2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Level 2 (64x64)
        self.enc2 = nn.Sequential(
            nn.Conv2d(feat*2, feat*4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat*4, feat*4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Level 3 (32x32)
        self.enc3 = nn.Sequential(
            nn.Conv2d(feat*4, feat*8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat*8, feat*8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Level 4 bottleneck (16x16)
        self.enc4 = nn.Sequential(
            nn.Conv2d(feat*8, feat*8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat*8, feat*8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Decoder (upsampling path) with attention gating on skip connections
        # Up 3 (from bottleneck to 32x32)
        self.up3 = nn.ConvTranspose2d(feat*8, feat*8, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(feat*8 + feat*8, feat*8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat*8, feat*8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.att3 = self._make_attn_gate(skip_channels=feat*8, gating_channels=feat*8)
        # Up 2 (64x64)
        self.up2 = nn.ConvTranspose2d(feat*8, feat*4, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(feat*4 + feat*4, feat*4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat*4, feat*4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.att2 = self._make_attn_gate(skip_channels=feat*4, gating_channels=feat*4)
        # Up 1 (128x128)
        self.up1 = nn.ConvTranspose2d(feat*4, feat*2, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(feat*2 + feat*2, feat*2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat*2, feat*2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.att1 = self._make_attn_gate(skip_channels=feat*2, gating_channels=feat*2)
        # Up 0 (256x256)
        self.up0 = nn.ConvTranspose2d(feat*2, feat, kernel_size=4, stride=2, padding=1)
        self.dec0 = nn.Sequential(
            nn.Conv2d(feat + feat, feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat, feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.att0 = None  # no lower skip to attend (lowest level is enc0 itself)
        # Final output conv
        self.final_conv = nn.Conv2d(feat, out_channels, kernel_size=3, padding=1)
    def _make_attn_gate(self, skip_channels, gating_channels, inter_channels=None):
        # Attention gating module (as in Attention U-Net)
        if inter_channels is None:
            inter_channels = skip_channels // 2
        attn_gate = nn.ModuleDict({
            'W_g': nn.Conv2d(gating_channels, inter_channels, kernel_size=1),
            'W_x': nn.Conv2d(skip_channels, inter_channels, kernel_size=1),
            'psi': nn.Conv2d(inter_channels, 1, kernel_size=1)
        })
        return attn_gate
    def _apply_attention(self, skip_feat, gating_feat, attn_gate):
        # Compute attention coefficients for skip connection
        # skip_feat: skip feature map (N, C_skip, H, W)
        # gating_feat: gating feature map (from deeper layer, N, C_g, H', W')
        # First, upsample gating to skip resolution if needed
        if skip_feat.shape[2:] != gating_feat.shape[2:]:
            gating_feat = F.interpolate(gating_feat, size=skip_feat.shape[2:], mode='nearest')
        # Linear transforms
        theta_x = attn_gate['W_x'](skip_feat)           # (N, inter, H, W)
        phi_g = attn_gate['W_g'](gating_feat)           # (N, inter, H, W)
        # Add and apply activation
        attn = F.relu(theta_x + phi_g, inplace=True)    # (N, inter, H, W)
        attn = attn_gate['psi'](attn)                   # (N, 1, H, W)
        attn = torch.sigmoid(attn)                      # (N, 1, H, W) attention mask
        # Multiply attention mask with skip features
        attended_skip = skip_feat * attn
        return attended_skip

    def forward(self, x):
        # Encoder forward
        e0 = self.enc0(x)  # 256x256 -> feat
        e1 = self.enc1(e0) # 128x128 -> feat*2
        e2 = self.enc2(e1) # 64x64 -> feat*4
        e3 = self.enc3(e2) # 32x32 -> feat*8
        e4 = self.enc4(e3) # 16x16 -> feat*8 (bottleneck)
        # Decoder forward with attention gating on skip connections
        d3 = self.up3(e4)           # upsample bottleneck to 32x32
        # Attention gate for skip e3 using gating from up-sampled d3
        att3 = self._apply_attention(e3, d3, self.att3)
        d3 = torch.cat((att3, d3), dim=1)  # concatenate skip (attended) with upsampled
        d3 = self.dec3(d3)          # process combined features
        d2 = self.up2(d3)           # 64x64
        att2 = self._apply_attention(e2, d2, self.att2)
        d2 = torch.cat((att2, d2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)           # 128x128
        att1 = self._apply_attention(e1, d1, self.att1)
        d1 = torch.cat((att1, d1), dim=1)
        d1 = self.dec1(d1)
        d0 = self.up0(d1)           # 256x256
        # For the lowest level, we can concatenate original e0 skip (optionally with no gating, or apply gating from d0 using att0 if it was defined)
        if self.att0 is not None:
            att0 = self._apply_attention(e0, d0, self.att0)
            d0 = torch.cat((att0, d0), dim=1)
        else:
            d0 = torch.cat((e0, d0), dim=1)
        d0 = self.dec0(d0)
        out = self.final_conv(d0)
        return out

# StampOne Encoder: includes preprocessing (gradient + wavelet + depthwise), MPN, and U-shape network, plus output refinement
class StampOneEncoder(nn.Module):
    def __init__(self):
        super(StampOneEncoder, self).__init__()
        # Depthwise layers for cover image and message
        self.depthwise_img = Depthwise(num_channels=15)
        self.depthwise_msg = Depthwise(num_channels=15)
        # Message Preparation Network
        self.mpn = MessagePrepNetwork(in_channels=15)
        # U-Shape encoder network (AttentionVNet) to generate encoded image, base channels=64, output 3 channels (RGB image)
        # Input channels to U-Net = cover features channels + message features channels (15 + 64 = 79)
        self.unet_enc = AttentionVNet(in_channels=79, out_channels=3, base_channels=64)
        # Output refinement CNN block (3 conv layers: first two LeakyReLU, last Snake)
        self.refine = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            Snake(3)  # final activation
        )
    def forward(self, cover_image, message_image):
        # cover_image: (N,3,256,256) original image
        # message_image: (N,3,16,16) binary message image (3 channels)
        N = cover_image.size(0)
        # Preprocessing: gradient + wavelet for cover and message
        cover_feat = self._gradient_wavelet_transform(cover_image)    # (N,15,256,256)
        msg_feat_small = self._gradient_wavelet_transform(message_image)  # (N,15,16,16)
        # Apply depthwise weighting (learnable emphasis on each sub-band)
        cover_feat = self.depthwise_img(cover_feat)
        msg_feat_small = self.depthwise_msg(msg_feat_small)
        # Prepare message features to full image size via MPN
        msg_feat = self.mpn(msg_feat_small)  # (N,64,256,256)
        # Concatenate cover and message features
        fused_input = torch.cat([cover_feat, msg_feat], dim=1)  # (N, 15+64 = 79, 256, 256)
        # U-Net encoder to generate encoded image (with 3 channels)
        encoded_image = self.unet_enc(fused_input)  # (N,3,256,256)
        # Refine output image with small CNN block for realism
        encoded_image_refined = self.refine(encoded_image)
        return encoded_image_refined

    def _gradient_wavelet_transform(self, img):
        # img: (N,3,H,W). For cover, H=W=256; for message, H=W=16
        # 1. Compute gradient (Sobel edges) for each channel
        # Sobel filters
        sobel_kernel_x = torch.tensor([[1, 0, -1],
                                       [2, 0, -2],
                                       [1, 0, -1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[ 1,  2,  1],
                                       [ 0,  0,  0],
                                       [-1, -2, -1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        # Apply separable Sobel filter to each channel using groups
        # Prepare weight for conv with groups=3
        weight_x = sobel_kernel_x.repeat(3, 1, 1, 1)  # (3,1,3,3)
        weight_y = sobel_kernel_y.repeat(3, 1, 1, 1)
        # Pad image to keep same size
        img_pad = F.pad(img, (1,1,1,1), mode='replicate')
        grad_x = F.conv2d(img_pad, weight_x, bias=None, groups=3)
        grad_y = F.conv2d(img_pad, weight_y, bias=None, groups=3)
        # Gradient magnitude
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # (N,3,H,W)
        # 2. Haar wavelet transform (1-level) on the gradient images
        # Define Haar wavelet filters (2x2) for LL, LH, HL, HH
        LL = torch.tensor([[0.25, 0.25], [0.25, 0.25]], dtype=torch.float32, device=img.device)
        LH = torch.tensor([[0.25, -0.25], [0.25, -0.25]], dtype=torch.float32, device=img.device)
        HL = torch.tensor([[0.25, 0.25], [-0.25, -0.25]], dtype=torch.float32, device=img.device)
        HH = torch.tensor([[0.25, -0.25], [-0.25, 0.25]], dtype=torch.float32, device=img.device)
        filters = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)  # (4,1,2,2)
        filters = filters.repeat(3, 1, 1, 1)  # (12,1,2,2) for 3 groups
        # Convolution with stride=2 for wavelet decomposition
        wavelet_out = F.conv2d(grad, filters, stride=2, groups=3)  # (N,12,H/2, W/2)
        # Upsample each sub-band back to original resolution
        wavelet_up = F.interpolate(wavelet_out, scale_factor=2.0, mode='nearest')  # (N,12,H,W)
        # Combine original gradient (IG) and wavelet sub-bands
        combined = torch.cat((grad, wavelet_up), dim=1)  # (N, 3+12 = 15, H, W)
        return combined

# StampOne Decoder: takes encoded image and decodes message
class StampOneDecoder(nn.Module):
    def __init__(self):
        super(StampOneDecoder, self).__init__()
        # Depthwise layer for encoded image features (15 channels after wavelet)
        self.depthwise_enc = Depthwise(num_channels=15)
        # Spatial Transformer to correct geometric distortions
        self.stn = SpatialTransformer(in_channels=15)
        # U-Shape decoder network (AttentionVNet) - mirrors the encoder architecture (input 15 channels, output 3 channels at full res)
        self.unet_dec = AttentionVNet(in_channels=15, out_channels=3, base_channels=64)
        # Downsampling pipeline to reduce output from 256x256 to 16x16
        # We use conv layers with stride 2 for downsampling
        self.down1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)   # 256 -> 128
        self.down2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 128 -> 64
        self.down3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 64 -> 32
        self.down4 = nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1) # 32 -> 16, output 16x16x64
        # 1D convolution layer to refine decoded bits sequence (to enhance precision)
        # Input: sequence length 256 with 64 channels, output: 64 channels (keeping same channels count for refinement)
        self.conv1d = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        # Final 2D conv to output 3-channel message image (with Snake activation)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.final_act = Snake(3)
    def forward(self, encoded_image):
        # encoded_image: (N,3,256,256) possibly with distortions (noisy input to decoder)
        # Preprocess encoded image: gradient + wavelet
        enc_feat = self._gradient_wavelet_transform(encoded_image)  # (N,15,256,256)
        # Apply depthwise weighting
        enc_feat = self.depthwise_enc(enc_feat)
        # Spatial transform to correct rotation/warp
        enc_feat = self.stn(enc_feat)
        # U-Net decoder to get full-res decoded message (as an image)
        decoded_full = self.unet_dec(enc_feat)  # (N,3,256,256)
        # Downsample stepwise to 16x16
        x = F.leaky_relu(self.down1(decoded_full), negative_slope=0.2, inplace=True)  # -> (N,32,128,128)
        x = F.leaky_relu(self.down2(x), negative_slope=0.2, inplace=True)            # -> (N,64,64,64)
        x = F.leaky_relu(self.down3(x), negative_slope=0.2, inplace=True)            # -> (N,128,32,32)
        x = F.leaky_relu(self.down4(x), negative_slope=0.2, inplace=True)            # -> (N,64,16,16)
        # Flatten spatial dimensions for conv1d: shape (N,64,256)
        N, C, H, W = x.shape  # H=W=16
        seq = x.view(N, C, H*W)  # (N,64,256)
        seq = self.conv1d(seq)   # (N,64,256), refine sequence features
        seq = F.leaky_relu(seq, negative_slope=0.2, inplace=True)
        # Reshape back to spatial 16x16
        x_refined = seq.view(N, C, H, W)  # (N,64,16,16)
        # Final convolution to get 3-channel binary message image
        out_msg = self.final_conv(x_refined)  # (N,3,16,16)
        out_msg = self.final_act(out_msg)     # Snake activation
        return out_msg

    def _gradient_wavelet_transform(self, img):
        # Same implementation as in encoder for gradient + wavelet transform
        sobel_kernel_x = torch.tensor([[1, 0, -1],
                                       [2, 0, -2],
                                       [1, 0, -1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[ 1,  2,  1],
                                       [ 0,  0,  0],
                                       [-1, -2, -1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        weight_x = sobel_kernel_x.repeat(3, 1, 1, 1)
        weight_y = sobel_kernel_y.repeat(3, 1, 1, 1)
        img_pad = F.pad(img, (1,1,1,1), mode='replicate')
        grad_x = F.conv2d(img_pad, weight_x, bias=None, groups=3)
        grad_y = F.conv2d(img_pad, weight_y, bias=None, groups=3)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        LL = torch.tensor([[0.25, 0.25],[0.25, 0.25]], dtype=torch.float32, device=img.device)
        LH = torch.tensor([[0.25, -0.25],[0.25, -0.25]], dtype=torch.float32, device=img.device)
        HL = torch.tensor([[0.25, 0.25],[-0.25, -0.25]], dtype=torch.float32, device=img.device)
        HH = torch.tensor([[0.25, -0.25],[-0.25, 0.25]], dtype=torch.float32, device=img.device)
        filters = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)
        filters = filters.repeat(3, 1, 1, 1)
        wavelet_out = F.conv2d(grad, filters, stride=2, groups=3)
        wavelet_up = F.interpolate(wavelet_out, scale_factor=2.0, mode='nearest')
        combined = torch.cat((grad, wavelet_up), dim=1)
        return combined

# Discriminator for steganography (image discriminator) – aims to distinguish cover vs encoded images
class SteganographyDiscriminator(nn.Module):
    def __init__(self):
        super(SteganographyDiscriminator, self).__init__()
        # PatchGAN-like discriminator (WGAN-based, no sigmoid)
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   # 256 -> 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 128 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),# 32 -> 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),# 16 -> 8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=8, stride=1, padding=0)   # 8 -> 1 (1x1 output)
        )
    def forward(self, img):
        # Output is a single score for the image (N,1,1,1) -> flatten to (N)
        out = self.model(img)
        return out.view(-1)

# Discriminator for spectral domain (spectral discriminator) – distinguishes real vs recovered messages using frequency domain
class SpectralDiscriminator(nn.Module):
    def __init__(self):
        super(SpectralDiscriminator, self).__init__()
        # Input will be 6 channels: 3 from spatial message image, 3 from its FFT magnitude
        self.model = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 8 -> 4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# 4 -> 2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),# 2 -> 1
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)  # output 1x1
        )
    def forward(self, message_img):
        # message_img: (N,3,16,16) with values ~0/1
        # Compute FFT magnitude for each channel
        # Use FFT on each image's channel
        # Get real and imaginary parts or magnitude
        # We'll compute magnitude of FFT (shifted)
        fft = torch.fft.fft2(message_img, norm='ortho')
        # Magnitude
        mag = torch.abs(fft)
        # We can also do fftshift (though not strictly necessary for discriminator performance, but we'll keep DC at center)
        # Simple way: roll half the dimensions
        mag = torch.fft.fftshift(mag, dim=(-2, -1))
        # Now mag is (N,3,16,16) real, combine with spatial input
        x = torch.cat([message_img, mag], dim=1)  # (N,6,16,16)
        out = self.model(x)
        return out.view(-1)
