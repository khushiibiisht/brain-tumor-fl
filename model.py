# model.py
# This file defines our ResNet-18 neural network
# adapted for 4-class brain tumor classification

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────
# THE RESIDUAL BLOCK
# This is the core building block of ResNet
# ─────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # CONV PATH — learns what needs to change
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SKIP CONNECTION
        # If dimensions change (channels or size),
        # we use a 1x1 conv to match them
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Conv path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))  # no ReLU here yet

        # Add skip connection then apply ReLU
        out += self.shortcut(x)          # F(x) + x
        out = F.relu(out)
        return out


# ─────────────────────────────────────────
# FULL ResNet-18 MODEL
# 8 residual blocks across 4 layer groups
# ─────────────────────────────────────────
class ResNet18(nn.Module):
    def __init__(self, num_classes=4):
        # num_classes=4 because we have:
        # glioma, meningioma, notumor, pituitary
        super().__init__()

        # STEM — first layer, reduces image from 224x224 to 56x56
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 4 LAYER GROUPS — each has 2 residual blocks
        # channels double, spatial size halves at each group
        self.layer1 = self._make_layer(64,  64,  stride=1)
        self.layer2 = self._make_layer(64,  128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        # CLASSIFICATION HEAD
        # converts 512 features → 4 class scores
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # global average pooling
            nn.Flatten(),             # flatten to 1D
            nn.Dropout(0.3),          # prevents overfitting
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            # First block handles the stride/channel change
            ResidualBlock(in_channels, out_channels, stride),
            # Second block, dimensions already match
            ResidualBlock(out_channels, out_channels, stride=1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x


# ─────────────────────────────────────────
# HELPER FUNCTION
# Call this to get a ready-to-use model
# ─────────────────────────────────────────
def get_model(device):
    """
    Creates ResNet-18 with pretrained ImageNet weights
    and replaces the head for 4-class tumor classification.
    device = 'cuda' (GPU) or 'cpu'
    """
    import torchvision.models as models

    # Load pretrained ResNet-18
    # pretrained=True means it already knows edges, textures,
    # shapes from ImageNet — we just fine-tune for MRI
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Freeze early layers — they already know basic features
    # Only train Layer4 + head — saves time on laptop
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # Replace final layer: 512 features → 4 tumor classes
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 4)
    )

    # Move model to GPU if available
    model = model.to(device)
    return model


# ─────────────────────────────────────────
# TEST — run this file directly to verify
# python model.py
# ─────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = get_model(device)

    # Create a fake MRI image batch to test
    # shape: (2 images, 3 channels, 224x224 pixels)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)

    print(f'Input shape  : {dummy_input.shape}')
    print(f'Output shape : {output.shape}')
    print(f'Output       : {output}')
    print()
    print('model.py is working correctly!')