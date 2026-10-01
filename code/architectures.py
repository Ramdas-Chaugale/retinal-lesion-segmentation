"""The four segmentation architectures.

Each is paired with two loss functions in gpu_02, giving eight trained models
per input size. The class bodies are unchanged from the run that produced the
current checkpoints, so parameter counts are directly comparable:

    SimpleUNet                 7,765,409   1 channel
    HybridUNet                 7,765,697   2 channels   (+288 over SimpleUNet)
    OCTFSegmentation           1,977,633   1 channel
    ModifiedOCTFSegmentation   1,978,017   2 channels   (+384 over OCTFSegmentation)

The two OCT-F variants reuse the four residual downsampling blocks of the
fine-tuned reconstruction encoder behind a decoder written for this project.
"""

import copy

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class _UNet(nn.Module):
    """The from-scratch U-Net. Only the first convolution differs by input width."""

    def __init__(self, in_ch):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.bottleneck = ConvBlock(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.final = nn.Conv2d(32, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.final(d1))


class SimpleUNet(_UNet):
    """1-channel input: the image alone."""

    def __init__(self):
        super().__init__(1)


class HybridUNet(_UNet):
    """2-channel input: the image and its reconstruction residual."""

    def __init__(self):
        super().__init__(2)


class OCTFSegmentation(nn.Module):
    """Pretrained OCT-F encoder blocks with a newly initialised decoder."""

    def __init__(self, octf_model):
        super().__init__()
        self.enc1 = copy.deepcopy(octf_model.blocks[0])
        self.enc2 = copy.deepcopy(octf_model.blocks[1])
        self.enc3 = copy.deepcopy(octf_model.blocks[2])
        self.enc4 = copy.deepcopy(octf_model.blocks[3])
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = ConvBlock(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = ConvBlock(16, 16)
        self.final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d4 = self.dec4(torch.cat([self.up4(e4), e3], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], 1))
        d1 = self.dec1(self.up1(d2))
        return torch.sigmoid(self.final(d1))


class ModifiedOCTFSegmentation(OCTFSegmentation):
    """As above, but the first convolution and its shortcut take two channels.

    704 new parameters; every other pretrained weight is kept.
    """

    def __init__(self, octf_model):
        super().__init__(octf_model)
        self.enc1.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.enc1.shortcut[0] = nn.Conv2d(2, 32, 1, stride=2)


def parameter_count(model):
    """Trainable parameters, for the table printed by gpu_02."""
    total = 0
    for p in model.parameters():
        total += p.numel()
    return total
