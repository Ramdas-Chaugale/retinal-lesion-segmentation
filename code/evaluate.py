"""Evaluation of segmentation models on the validation set.

evaluate_model runs one model over the validation files and returns per-image
metrics plus the predictions, so notebook 04 can produce every plot without
running inference again.
"""

import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from postprocessing import improved_post_processing

SMOOTH = 1.0


def evaluate_model(model, val_files, orig_folder, mask_folder, residual_cache,
                   device, input_type='single', apply_pp=False, img_size=256,
                   pp_kernel=15):
    """Evaluate one model on the validation files.

    input_type is 'single' for 1-channel models or 'hybrid' for 2-channel models.
    apply_pp applies morphological closing plus largest component to the binary
    prediction. Returns a dict of per-image metrics and predictions.

    img_size is a single integer, so this resizes to a square. That is the
    256 x 256 run, and the defaults above are its settings. The native
    496 x 768 evaluation is not square and is done in notebook 04, which
    keeps the two axes separate.
    """
    model.eval()
    results = {
        'files': [], 'dice': [], 'iou': [], 'precision': [], 'recall': [],
        'cc': [], 'missed': [], 'extra': [], 'time': [], 'gt_size': [],
        'probability': [], 'prediction': [], 'ground_truth': [],
        'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
        'input_type': input_type, 'apply_pp': apply_pp,
        'img_size': img_size, 'pp_kernel': pp_kernel,
    }

    for filename in val_files:
        # single channel extracted, images are 3-channel RGB with R=G=B
        img = np.array(Image.open(os.path.join(orig_folder, filename)))
        gray = cv2.resize(img[:, :, 0], (img_size, img_size)).astype(np.float32) / 255.0

        if input_type == 'hybrid':
            residual = cv2.resize(residual_cache[filename],
                                  (img_size, img_size)).astype(np.float32)
            model_input = torch.from_numpy(np.stack([gray, residual], axis=0))
        else:
            model_input = torch.from_numpy(gray).unsqueeze(0)
        model_input = model_input.unsqueeze(0).to(device)

        mask_path = os.path.join(mask_folder, filename.replace('.tif', '_mask.png'))
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        gt = (gt > 127).astype(np.float32)

        start = time.time()
        with torch.no_grad():
            probability = model(model_input).squeeze().detach().cpu().numpy()
        prediction = (probability > 0.5).astype(np.float32)
        if apply_pp:
            prediction = improved_post_processing(prediction, pp_kernel)
        inference_time = time.time() - start

        tp = ((prediction == 1) & (gt == 1)).sum()
        fp = ((prediction == 1) & (gt == 0)).sum()
        fn = ((prediction == 0) & (gt == 1)).sum()
        tn = ((prediction == 0) & (gt == 0)).sum()

        intersection = (prediction * gt).sum()
        union = prediction.sum() + gt.sum() - intersection
        dice = (2 * intersection + SMOOTH) / (prediction.sum() + gt.sum() + SMOOTH)
        iou = (intersection + SMOOTH) / (union + SMOOTH)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        cc = cv2.connectedComponents(prediction.astype(np.uint8))[0] - 1
        missed = ((gt.sum() - intersection) / gt.sum() * 100) if gt.sum() > 0 else 0
        extra = ((prediction.sum() - intersection) / gt.sum() * 100) if gt.sum() > 0 else 0

        results['files'].append(filename)
        results['dice'].append(dice)
        results['iou'].append(iou)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['cc'].append(cc)
        results['missed'].append(missed)
        results['extra'].append(extra)
        results['time'].append(inference_time)
        results['gt_size'].append(float(gt.mean() * 100))
        results['probability'].append(probability)
        results['prediction'].append(prediction)
        results['ground_truth'].append(gt)
        results['tp'] += int(tp)
        results['fp'] += int(fp)
        results['fn'] += int(fn)
        results['tn'] += int(tn)

    return results


def print_results(results, model_name):
    """Print one summary row for an evaluated model."""
    multi = sum(1 for c in results['cc'] if c > 1)
    total = len(results['cc'])
    print(f"{model_name:<28} "
          f"{np.mean(results['dice']):<8.3f} {np.std(results['dice']):<8.3f} "
          f"{np.mean(results['iou']):<8.3f} "
          f"{np.mean(results['precision']):<8.3f} {np.mean(results['recall']):<8.3f} "
          f"{multi}/{total:<6} "
          f"{np.mean(results['missed']):<8.1f} {np.mean(results['extra']):<8.1f} "
          f"{np.mean(results['time']) * 1000:.0f}ms")


def print_header():
    """Column header matching print_results."""
    print(f"{'Model':<28} {'Dice':<8} {'Std':<8} {'IoU':<8} {'Prec':<8} "
          f"{'Recall':<8} {'CC>1':<8} {'Missed':<8} {'Extra':<8} {'Time'}")
    print("=" * 110)
