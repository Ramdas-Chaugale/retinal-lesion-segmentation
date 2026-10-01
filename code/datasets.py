"""Datasets for the one-channel and two-channel models.

The scans are stored as three-channel TIFF but R = G = B on every annotated
image, verified with a maximum inter-channel difference of 0, so a single
channel is EXTRACTED by indexing (`img[:, :, 0]`). It is never averaged across channels,
because averaging would be arithmetic on three copies of the same number.

img_size accepts either an int (square resize) or a (height, width) tuple
(native geometry), so the same class serves both runs.
"""

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _target_size(img_size):
    """(height, width) from either an int or a pair."""
    if isinstance(img_size, int):
        return img_size, img_size
    return img_size[0], img_size[1]


def _load_mask(mask_folder, filename, height, width):
    """The binary mask, nearest-neighbour resized so it stays binary."""
    path = os.path.join(mask_folder, filename.replace('.tif', '_mask.png'))
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


class SingleChannelDataset(Dataset):
    """Image only. Feeds Plain U-Net and OCT-F Backbone."""

    def __init__(self, image_folder, mask_folder, file_list, img_size=256,
                 augment=False):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.file_list = file_list
        self.height, self.width = _target_size(img_size)
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        image = np.array(Image.open(os.path.join(self.image_folder, filename)))

        # single channel extracted by indexing, not averaged
        gray = image[:, :, 0]
        gray = cv2.resize(gray, (self.width, self.height)).astype(np.float32) / 255.0
        mask = _load_mask(self.mask_folder, filename, self.height, self.width)

        # the only augmentation is a horizontal flip, applied to both together
        if self.augment and np.random.random() > 0.5:
            gray = np.fliplr(gray).copy()
            mask = np.fliplr(mask).copy()

        return (torch.from_numpy(gray).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0))


class HybridDataset(Dataset):
    """Image and residual stacked as two channels.

    Residuals are read from the cache written by gpu_01, never recomputed, so
    every two-channel model sees byte-identical inputs and no difference between
    them can be an artefact of recomputation.
    """

    def __init__(self, image_folder, mask_folder, file_list, residual_cache,
                 img_size=256, augment=False):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.file_list = file_list
        self.cache = residual_cache
        self.height, self.width = _target_size(img_size)
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        image = np.array(Image.open(os.path.join(self.image_folder, filename)))

        gray = image[:, :, 0]
        gray = cv2.resize(gray, (self.width, self.height)).astype(np.float32) / 255.0

        # cached at 496x768, resampled here to whatever this run uses
        residual = self.cache[filename]
        residual = cv2.resize(residual, (self.width, self.height)).astype(np.float32)

        mask = _load_mask(self.mask_folder, filename, self.height, self.width)

        if self.augment and np.random.random() > 0.5:
            gray = np.fliplr(gray).copy()
            residual = np.fliplr(residual).copy()
            mask = np.fliplr(mask).copy()

        two_channel = np.stack([gray, residual], axis=0)
        return (torch.from_numpy(two_channel),
                torch.from_numpy(mask).unsqueeze(0))
