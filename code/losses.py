"""The two loss functions.

Background outnumbers lesion by roughly 46 to 1, so plain cross-entropy is
minimised well by predicting background everywhere. Both losses below are
therefore Dice-based.

Each returns (loss, dice_score) so the training loop can report an
interpretable number alongside the quantity it is minimising.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Half Dice, half binary cross-entropy. The baseline loss."""

    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()

    def forward(self, prediction, target):
        smooth = 1.0
        p = prediction.view(-1)
        t = target.view(-1)
        dice = 1 - (2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)
        bce = self.bce(prediction, target)
        total = self.dice_weight * dice + self.bce_weight * bce
        return total, 1 - dice.item()


class SoftErosionLoss(nn.Module):
    """Dice, BCE, and a differentiable penalty on fragmented predictions.

    A laser lesion is one connected region, but a connected-component count is
    not differentiable: cv2.connectedComponents returns a plain integer, so
    .backward() cannot be called on it. Soft erosion stands in for it.

    Eroding with min pooling removes small fragments entirely while shrinking a
    large blob only at its edge, so the mean difference between a prediction and
    its eroded form is larger when the prediction is fragmented. That difference
    is a tensor, so the gradient reaches the input.

    kernel_size scales with the input: 5 at 256x256, 9 at native 496x768.
    """

    def __init__(self, dice_weight=0.4, bce_weight=0.4, erosion_weight=0.2,
                 kernel_size=5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.erosion_weight = erosion_weight
        self.kernel_size = kernel_size
        self.bce = nn.BCELoss()

    def soft_erosion(self, prediction):
        """Min pooling, written as negate, max pool, negate."""
        padding = self.kernel_size // 2
        return -F.max_pool2d(-prediction, kernel_size=self.kernel_size,
                             stride=1, padding=padding)

    def forward(self, prediction, target):
        smooth = 1.0
        p = prediction.view(-1)
        t = target.view(-1)
        dice = 1 - (2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)
        bce = self.bce(prediction, target)

        eroded = self.soft_erosion(prediction)
        fragment_penalty = (prediction - eroded).mean()

        total = (self.dice_weight * dice
                 + self.bce_weight * bce
                 + self.erosion_weight * fragment_penalty)
        return total, 1 - dice.item()


def build_loss(name, erosion_kernel):
    """The loss named in the model table, ready to use."""
    if name == 'DiceBCELoss':
        return DiceBCELoss()
    if name == 'SoftErosionLoss':
        return SoftErosionLoss(kernel_size=erosion_kernel)
    raise ValueError('unknown loss: %s' % name)
