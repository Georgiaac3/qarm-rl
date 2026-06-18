"""
Pixel-wise grasping policy network.

Inspired by TossingBot and similar vision-based manipulation papers.
Network architecture:
- Input: RGB (64x64x3) + Depth (64x64x1) + Heatmap (64x64x1) → concatenated (64x64x5)
- Backbone: ResNet-based feature extraction
- Heads: 
  - Grasp prediction (64x64x1): logits for grasp probability per pixel
  - Velocity correction (64x64x2): [vx, vy] for throwing velocity
  - Confidence (64x64x1): model confidence for each pixel
"""

import torch
import torch.nn as nn
from typing import Tuple


class ConvBlock(nn.Module):
    """Residual conv block with batch norm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = x + residual
        x = self.relu(x)
        return x


class PixelWiseGraspingPolicy(nn.Module):
    """
    Pixel-wise grasping policy network.
    
    Input:
        rgb: (B, 3, 64, 64)
        depth: (B, 1, 64, 64)
        heatmap: (B, 1, 64, 64)
    
    Output:
        grasp_logits: (B, 1, 64, 64) - probability of successful grasp at each pixel
        velocity: (B, 2, 64, 64) - velocity correction [vx, vy]
        confidence: (B, 1, 64, 64) - model confidence
    """
    
    def __init__(self, in_channels: int = 5, hidden_channels: int = 32):
        super().__init__()
        
        # Input processing
        self.input_conv = ConvBlock(in_channels, hidden_channels, kernel_size=3)
        
        # Feature backbone (encoder)
        self.encoder = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2),
            ResidualBlock(hidden_channels * 2),
            ConvBlock(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2),
            ResidualBlock(hidden_channels * 4),
        )
        
        # Decoder (upsample back to original resolution)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_channels * 2),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_channels),
        )
        
        # Output heads (pixel-wise prediction)
        self.grasp_head = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),  # Binary classification
        )
        
        self.velocity_head = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),  # [vx, vy]
        )
        
        self.confidence_head = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),  # Confidence [0, 1]
        )
    
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, heatmap: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            rgb: (B, 3, 64, 64)
            depth: (B, 1, 64, 64)
            heatmap: (B, 1, 64, 64)
        
        Returns:
            grasp_logits: (B, 1, 64, 64)
            velocity: (B, 2, 64, 64) normalized to [-1, 1]
            confidence: (B, 1, 64, 64) in [0, 1]
        """
        # Concatenate inputs
        x = torch.cat([rgb, depth, heatmap], dim=1)  # (B, 5, 64, 64)
        
        # Process input
        x = self.input_conv(x)
        
        # Encode
        x = self.encoder(x)
        
        # Decode
        x = self.decoder(x)
        
        # Compute outputs
        grasp_logits = self.grasp_head(x)  # (B, 1, 64, 64)
        velocity = torch.tanh(self.velocity_head(x))  # Normalize to [-1, 1]
        confidence = torch.sigmoid(self.confidence_head(x))  # Normalize to [0, 1]
        
        return grasp_logits, velocity, confidence
    
    def predict_action(self, rgb: torch.Tensor, depth: torch.Tensor, heatmap: torch.Tensor) \
            -> Tuple[Tuple[int, int], torch.Tensor, float]:
        """
        Predict best grasp pixel and corresponding velocity correction.
        
        Args:
            rgb: (B, 3, 64, 64) or (3, 64, 64)
            depth: (B, 1, 64, 64) or (1, 64, 64)
            heatmap: (B, 1, 64, 64) or (1, 64, 64)
        
        Returns:
            (pixel_x, pixel_y): Best grasp location
            velocity: (2,) velocity correction
            confidence: Confidence score
        """
        # Add batch dimension if needed
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
            depth = depth.unsqueeze(0)
            heatmap = heatmap.unsqueeze(0)
        
        with torch.no_grad():
            grasp_logits, velocity, confidence = self.forward(rgb, depth, heatmap)
        
        # Find pixel with highest grasp probability
        B, _, H, W = grasp_logits.shape
        
        # Combine grasp probability with confidence
        score = torch.sigmoid(grasp_logits) * confidence
        
        # Find best pixel
        score_flat = score[0, 0, :, :]  # Remove batch and channel dims
        pixel_idx = torch.argmax(score_flat)
        pixel_y = pixel_idx // W
        pixel_x = pixel_idx % W
        
        # Get velocity at best pixel
        best_velocity = velocity[0, :, pixel_y, pixel_x].cpu().numpy()
        best_confidence = confidence[0, 0, pixel_y, pixel_x].item()
        
        return (pixel_x.item(), pixel_y.item()), best_velocity, best_confidence


