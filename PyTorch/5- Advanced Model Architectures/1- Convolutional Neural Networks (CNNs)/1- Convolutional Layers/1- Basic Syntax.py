import torch
import torch.nn as nn

# 2D Convolutional Layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
