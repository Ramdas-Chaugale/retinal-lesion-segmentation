"""Post-processing for predicted lesion masks."""

import cv2
import numpy as np


def improved_post_processing(binary_mask, closing_kernel_size=15):
    """Close small gaps, then keep the largest connected component.

    Closing first is what makes this work: keeping the largest component on its
    own can discard the correct fragment, whereas closing merges nearby
    fragments into one region before the selection. Each lesion is a single
    object, so one component is the expected output.
    closing_kernel_size scales with the image: 15 at 256, 29 at native.
    """
    mask_uint8 = binary_mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    num_labels, labels = cv2.connectedComponents(closed)
    if num_labels <= 1:
        return closed.astype(np.float32)

    largest_label = 0
    largest_size = 0
    for label_id in range(1, num_labels):
        size = (labels == label_id).sum()
        if size > largest_size:
            largest_size = size
            largest_label = label_id

    return (labels == largest_label).astype(np.float32)
