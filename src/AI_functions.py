import os
import io
import math
import time
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg19

# --- CompressAI Imports ---
from compressai.layers import GDN, AttentionBlock, ResidualBlock
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import conv, deconv



CustomAttentionBlock = AttentionBlock

NUM_FILTERS_N = 256
NUM_FILTERS_M = 448

device = "cuda" if torch.cuda.is_available() else "cpu"


class SatelliteVBR(nn.Module):
    """
    Full VBR Autoencoder model implementing all features from Chapter 3.
    This architecture is fully convolutional and will adapt to 64x64 inputs.
    g_a(64x64) -> y(4x4)
    h_a(y, 4x4) -> z(1x1)
    """
    def __init__(self, N=NUM_FILTERS_N, M=NUM_FILTERS_M):
        super().__init__()

        # --- Main Encoder (g_a) ---
        # 3x3 kernels, stride 2, 4 layers
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=3, stride=2),
            GDN(N), # Simplified GDN (Sec 3.2.2.3) [cite: 1175-1186]
            conv(N, N, kernel_size=3, stride=2),
            GDN(N),
            CustomAttentionBlock(N), # Attention Module (Sec 3.3.2) [cite: 1278-1286]
            conv(N, N, kernel_size=3, stride=2),
            GDN(N),
            conv(N, M, kernel_size=3, stride=2),
            CustomAttentionBlock(M), # Attention Module (Sec 3.3.2)
        )

        # --- Main Decoder (g_s) ---
        self.g_s = nn.Sequential(
            CustomAttentionBlock(M), # Attention Module
            deconv(M, N, kernel_size=3, stride=2),
            GDN(N, inverse=True),
            CustomAttentionBlock(N), # Attention Module
            deconv(N, N, kernel_size=3, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=3, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=3, stride=2),
        )

        # --- Hyperprior (h_a and h_s) ---
        # *** FIX 1: Using LeakyReLU for better stability than ReLU ***
        self.h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1), # 3x3 kernel
            nn.LeakyReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2), # 3x3 kernel
            nn.LeakyReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2), # 3x3 kernel
        )

        self.h_s = nn.Sequential(
            deconv(N, N, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            deconv(N, N, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            deconv(N, M * 2, kernel_size=3, stride=1), # Output 2*M for mean and log_scale
        )

        # --- Entropy Models (from compressai) ---
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None) # None for variable scales

    def forward(self, x, gain):
        """
        Forward pass with Variable Bit Rate (VBR) Gain Unit. [cite: 1045-1050]
        'gain' is a [B, 1, 1, 1] tensor.
        """
        # --- VBR Gain Unit (Encoder) ---
        y = self.g_a(x)
        y_scaled = y * gain # This is the Gain Unit

        # --- Hyperprior Path ---
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        # Hyperprior decoder predicts mean and log_scale
        mean_scale = self.h_s(z_hat)
        mean, log_scale = mean_scale.chunk(2, 1) # Split M*2 into M (mean) and M (log_scale)

        # --- *** FIX 2: Exponentiate log_scale to get positive scale *** ---
        # This is the critical fix to prevent NaN.
        scale = torch.exp(log_scale)

        # --- Main Bottleneck Path ---
        y_hat_scaled, y_likelihoods = self.gaussian_conditional(y_scaled, scale, mean)

        # --- VBR Inverse Gain Unit (Decoder) ---
        y_hat = y_hat_scaled / gain

        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods,
            },
        }



class VggPerceptualLoss(nn.Module):
    """
    Implements the perceptual loss (P)
    Uses early layers of VGG19.
    """
    def __init__(self):
        super(VggPerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features.to(device)
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # Extract features from layers at "depths 2 and 4"
        # This corresponds to relu_2_2 (layer 9) and relu_4_2 (layer 27)
        self.features = nn.Sequential(
            *list(vgg.children())[:28]
        )
        self.feature_layers = [9, 27] # Indices for relu_2_2 and relu_4_2

        # VGG expects normalized input
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x_true, x_pred):
        x_true_norm = self.normalize(x_true)
        x_pred_norm = self.normalize(x_pred)

        loss = 0.0
        features_true = x_true_norm
        features_pred = x_pred_norm

        for i, layer in enumerate(self.features):
            features_true = layer(features_true)
            features_pred = layer(features_pred)

            if i in self.feature_layers:
                loss += F.mse_loss(features_true, features_pred)

        return loss