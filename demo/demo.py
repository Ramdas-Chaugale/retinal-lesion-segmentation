"""Run the lesion segmentation on your own OCT images.

    python demo.py --image myscan.png
    python demo.py --image myscan.png --truth mymask.png
    python demo.py --folder myscans/ --truth-folder mymasks/ --model hybrid

Point it at a PNG, JPG or TIF and it writes an overlay showing the lesion it
found, plus the lesion area as a percentage of the frame. None of the original
study data is needed: the imaging data is private and is not distributed with
this repository, only the trained weights.

Give it your own ground truth as well, with --truth, and it scores itself.
It prints Dice, IoU, precision and recall, and writes a second picture in
three colours: yellow where it agrees with you, green for lesion it missed,
red for tissue it wrongly called lesion. Green against red is the useful
reading, because it says which way the model errs. Two models can share a
Dice score and fail in opposite directions.

Five models are available. The default, plain, takes the image alone and is the
leading configuration at native resolution. The three marked "2ch" also need the
reconstruction residual, so they load the fine-tuned OCT-F foundation model as
well and take a few seconds longer on the first image.

    plain       Plain U-Net,           1 channel   (default)
    hybrid      Hybrid U-Net,          2 channels
    softerosion Soft Erosion Hybrid,   2 channels
    octf        OCT-F Backbone,        1 channel
    modified    Modified OCT-F 2ch,    2 channels

Everything runs on CPU. No GPU is required.

It reads PNG, JPG, TIFF and BMP. It cannot read what a scanner writes directly,
such as Heidelberg .e2e and .vol, Topcon .fda, Zeiss .img or DICOM, because
those hold a whole volume rather than one picture. Export the B-scan you want
from the scanner's viewer first and it will read that. Pointed at one of those
files it says so, rather than failing without a reason.
"""

import argparse
import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def find_repository_root(start):
    """Walk up from this file until the shared modules are found.

    This script sits in demo/, so the repository root is its parent. Searching
    for a file that must be there is steadier than counting directories: it
    still works if the folder is renamed or nested one deeper.
    """
    folder = start
    for _ in range(4):
        if os.path.exists(os.path.join(folder, 'code', 'architectures.py')):
            return folder
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    raise SystemExit(
        "Could not find code/architectures.py above %s.\n"
        "Run this from inside the repository." % start)


REPO = find_repository_root(HERE)
sys.path.insert(0, os.path.join(REPO, 'code'))

from architectures import (SimpleUNet, HybridUNet, OCTFSegmentation,      # noqa: E402
                           ModifiedOCTFSegmentation)
from postprocessing import improved_post_processing                        # noqa: E402

# the models were trained at the acquisition geometry, so anything fed to them
# is resized to it first
INPUT_HEIGHT = 496
INPUT_WIDTH = 768

# the closing kernel scales with the input; 29 is the native-resolution value
CLOSING_KERNEL = 29

# What a scanner actually writes. None of these is an image file OpenCV can
# open: they are vendor containers holding a whole volume plus its metadata.
# Mapping the extension to a name lets the error say what the file is.
NATIVE_OCT_FORMATS = {
    '.e2e': 'Heidelberg Spectralis E2E',
    '.vol': 'Heidelberg Spectralis VOL',
    '.fda': 'Topcon FDA',
    '.fds': 'Topcon FDS',
    '.img': 'Zeiss IMG',
    '.oct': 'Optovue OCT',
    '.dcm': 'DICOM',
    '.dicom': 'DICOM',
}

# what this script can actually open
READABLE = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

EXPORT_ADVICE = (
    "This is a %s file, which holds a whole scan volume rather than one image,\n"
    "so it has to be exported to a picture first. In the viewer that came with\n"
    "the scanner, open the scan, pick the B-scan you want and export it as PNG\n"
    "or TIFF. Free readers also exist, for example the Python packages\n"
    "oct-converter and eyepy, which will write the B-scans out for you.")


# OpenCV works in BGR. These are the three colours the error figures in the
# paper already use, so a reader who has seen those reads this picture the
# same way without a second explanation.
YELLOW = (0, 255, 255)   # correct: the model and the ground truth agree
GREEN = (0, 255, 0)      # missed: ground truth says lesion, the model does not
RED = (0, 0, 255)        # extra:  the model says lesion, the ground truth does not

# an expert annotation arrives as a yellow outline drawn onto the scan, so the
# same threshold used to build the training masks is reused here
ANNOTATION_R = 200
ANNOTATION_G = 200
ANNOTATION_B = 100

CHECKPOINTS = os.path.join(REPO, 'reconstruction', 'checkpoints')
FINETUNED_OCTF = os.path.join(REPO, 'reconstruction', 'fine_tuned_reconstruction.pth')
OCTF_SOURCE = os.path.join(REPO, 'reconstruction', 'OCT-F_Reconstruction', 'code')

# the name of the model currently running, for the figure heading. A single
# element list because run_one is called after main has chosen the model.
MODEL_NAME = ['plain']

MODELS = {
    'plain':       ('plain_unet.pth',          1),
    'hybrid':      ('hybrid_unet.pth',         2),
    'softerosion': ('hybrid_soft_erosion.pth', 2),
    'octf':        ('octf_backbone.pth',       1),
    'modified':    ('modified_octf_2ch.pth',   2),
}


def load_single_channel(path):
    """Read an image and take one channel.

    The scans are greyscale but are usually stored with three identical
    channels, so I take the first rather than averaging: averaging would be
    arithmetic on three copies of the same number.
    """
    extension = os.path.splitext(path)[1].lower()
    if extension in NATIVE_OCT_FORMATS:
        raise SystemExit("%s\n\n%s" % (
            os.path.basename(path),
            EXPORT_ADVICE % NATIVE_OCT_FORMATS[extension]))

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(
            "Could not read %s\n"
            "Readable formats are: %s" % (path, ', '.join(READABLE)))
    if image.ndim == 3:
        image = image[:, :, 0]
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def build_octf():
    """The fine-tuned OCT-F.

    Two things need it: the three two-channel models, which subtract its
    reconstruction from the scan to make the residual, and the two models whose
    encoder is a copy of its blocks.
    """
    if not os.path.exists(FINETUNED_OCTF):
        raise SystemExit(
            "This model is built on the fine-tuned OCT-F, which is missing:\n"
            "%s\n"
            "Use --model plain, which needs no other model." % FINETUNED_OCTF)
    sys.path.insert(0, OCTF_SOURCE)
    from architectureV2Res_new import ArchitectureV2

    checkpoint = torch.load(FINETUNED_OCTF, map_location='cpu', weights_only=False)
    octf = ArchitectureV2()
    octf.load_state_dict(checkpoint['model_state_dict'])
    octf.eval()
    return octf


def compute_residual(octf, resized):
    """The absolute difference between a scan and its reconstruction.

    The decoder hands the image back transposed, so I transpose it before
    differencing. Getting that wrong compares the scan against its own
    transpose, which looks like a very confident lesion detector and is not one.
    """
    tensor = torch.from_numpy(resized).float()[None, None] / 255.0
    with torch.no_grad():
        reconstruction, _ = octf(tensor)
    original = tensor.squeeze().numpy()
    rebuilt = reconstruction.squeeze().numpy().T
    return np.abs(original - rebuilt)


def build_model(name, octf):
    """Load one segmentation model with its trained weights."""
    filename, channels = MODELS[name]
    path = os.path.join(CHECKPOINTS, filename)
    if not os.path.exists(path):
        raise SystemExit("Missing weights: %s" % path)

    if name == 'plain':
        model = SimpleUNet()
    elif name in ('hybrid', 'softerosion'):
        model = HybridUNet()
    elif name == 'octf':
        model = OCTFSegmentation(octf)
    else:
        model = ModifiedOCTFSegmentation(octf)

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, channels


def segment(model, channels, resized, residual):
    """One forward pass, threshold, then the single-region rule."""
    image = resized.astype(np.float32) / 255.0
    if channels == 2:
        stacked = np.stack([image, residual])
    else:
        stacked = image[None]

    with torch.no_grad():
        probability = model(torch.from_numpy(stacked)[None])
    probability = probability[0, 0].numpy()

    hard = (probability > 0.5).astype(np.uint8)
    # a laser lesion is one connected region, so I close small gaps and then
    # keep the largest blob; doing it the other way round can throw away a
    # genuine part of the lesion
    return improved_post_processing(hard, CLOSING_KERNEL), probability


def draw_overlay(resized, mask):
    """The scan in grey with the predicted lesion outlined and tinted yellow.

    Yellow is the convention this project already uses: the expert annotations
    arrive as yellow outlines drawn onto the scan, and the error figures use
    yellow for lesion found correctly. Keeping it means a reader who has seen
    any other figure here reads this one the same way.
    """
    canvas = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    tint = canvas.copy()
    tint[mask > 0] = YELLOW
    canvas = cv2.addWeighted(canvas, 0.7, tint, 0.3, 0)

    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, YELLOW, 2)
    return canvas


def load_truth_mask(path, from_annotation):
    """Read the ground truth the user supplied and return a binary mask.

    Two forms are accepted, because two forms exist in practice. A binary mask
    is a picture that is already black and white, which is what a labelling
    tool exports. An annotation is the original scan with a yellow outline
    drawn on top of it, which is how the labels for this study were delivered,
    and the outline has to be found and filled before it is a mask.
    """
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit("Could not read the ground truth %s" % path)

    if not from_annotation:
        # a binary mask: anything above mid grey counts as lesion
        if image.ndim == 3:
            image = image[:, :, 0]
        return (image > 127).astype(np.uint8)

    if image.ndim != 3:
        raise SystemExit(
            "--truth-is-annotation looks for a yellow outline, so it needs a "
            "colour image. %s has one channel, which means it is already a "
            "mask: drop that flag." % path)

    # OpenCV loads colour as BGR, so channel 2 is red and channel 0 is blue
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    is_outline = (red > ANNOTATION_R) & (green > ANNOTATION_G) & (blue < ANNOTATION_B)
    outline = is_outline.astype(np.uint8)

    if outline.sum() == 0:
        raise SystemExit(
            "Found no yellow outline in %s. If that file is already a black "
            "and white mask, drop --truth-is-annotation." % path)

    # fill the outline to get the solid region the models are trained against
    contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(outline.shape, dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)

    # if the drawn stroke has a gap the contour is open, filling adds nothing,
    # and the score below would be measured against the stroke itself. That is
    # a real defect that occurred twice in this study, so it is reported rather
    # than silently scored.
    if filled.sum() < 2 * outline.sum():
        print("  warning: the outline in %s does not close, so filling it "
              "added almost nothing. The score is being measured against the "
              "stroke and will read low."
              % os.path.basename(path))
    return filled


def score(prediction, truth):
    """Dice, IoU, precision and recall for one image.

    Written out in full rather than called from a library so the arithmetic is
    visible. Dice and IoU both measure overlap. Precision and recall split that
    overlap into its two failure directions, which is the split that mattered
    most in this project: a model can lose Dice by missing lesion or by adding
    tissue, and those are different faults with different consequences.
    """
    prediction = prediction.astype(bool)
    truth = truth.astype(bool)

    true_positive = float(np.logical_and(prediction, truth).sum())
    false_positive = float(np.logical_and(prediction, ~truth).sum())
    false_negative = float(np.logical_and(~prediction, truth).sum())

    smooth = 1.0
    dice = (2 * true_positive + smooth) / (prediction.sum() + truth.sum() + smooth)
    union = true_positive + false_positive + false_negative
    iou = (true_positive + smooth) / (union + smooth)

    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0.0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0.0

    # the two error directions as a share of the true lesion, reported the way
    # the paper reports them
    if truth.sum() > 0:
        missed = 100.0 * false_negative / truth.sum()
        extra = 100.0 * false_positive / truth.sum()
    else:
        missed = 0.0
        extra = 0.0

    return {
        'dice': dice, 'iou': iou, 'precision': precision, 'recall': recall,
        'missed': missed, 'extra': extra,
    }


def draw_error_map(resized, prediction, truth):
    """The three colour picture: what was right, what was missed, what was added.

    Yellow is agreement. Green is lesion the model failed to find, which is
    under-segmentation. Red is tissue the model called lesion when the ground
    truth does not, which is over-segmentation.
    """
    canvas = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    tint = canvas.copy()

    prediction = prediction.astype(bool)
    truth = truth.astype(bool)

    tint[np.logical_and(prediction, truth)] = YELLOW
    tint[np.logical_and(np.logical_not(prediction), truth)] = GREEN
    tint[np.logical_and(prediction, np.logical_not(truth))] = RED

    canvas = cv2.addWeighted(canvas, 0.55, tint, 0.45, 0)

    # outline the ground truth so its boundary stays readable under the tint
    truth_contours, _ = cv2.findContours(truth.astype(np.uint8),
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, truth_contours, -1, (255, 255, 255), 1)
    return canvas


def draw_legend(canvas, scores):
    """Print the score and the key onto the picture itself.

    A saved image that needs the terminal output beside it to be understood is
    only half a result, so the number goes on the picture.
    """
    panel = np.zeros((96, canvas.shape[1], 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel,
                "Dice %.3f   IoU %.3f   Precision %.3f   Recall %.3f"
                % (scores['dice'], scores['iou'],
                   scores['precision'], scores['recall']),
                (12, 32), font, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    swatches = [(YELLOW, "correct"),
                (GREEN, "missed lesion (under)"),
                (RED, "extra tissue (over)")]
    x = 12
    for colour, label in swatches:
        cv2.rectangle(panel, (x, 54), (x + 22, 74), colour, -1)
        cv2.putText(panel, label, (x + 30, 70), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        x = x + 42 + 11 * len(label)

    return np.vstack([canvas, panel])


def match_truth_files(paths, truth_folder):
    """Pair each image with its ground truth by filename.

    The stem is matched, so `scan_001.tif` finds `scan_001_mask.png`
    or `scan_001.png`. An image with no partner is still segmented, it is
    just not scored, so a part-labelled folder works.
    """
    available = {}
    for filename in sorted(os.listdir(truth_folder)):
        stem = os.path.splitext(filename)[0]
        available[stem] = os.path.join(truth_folder, filename)

    matched = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in available:
            matched.append(available[stem])
        elif stem + '_mask' in available:
            matched.append(available[stem + '_mask'])
        else:
            matched.append(None)
    return matched


def save_panels(out_path, scan, mask, truth, scores, name, model_name):
    """One figure, panels side by side, the way the notebooks present a result.

    Three panels without a ground truth and four with one. Looking at a
    prediction means comparing it against something, so putting the panels in
    one image is the difference between reading a result and assembling it.
    """
    if truth is None:
        titles = ['The scan as it arrives',
                  'Predicted lesion',
                  'Prediction over the scan']
        panels = [scan, (mask * 255).astype(np.uint8),
                  cv2.cvtColor(draw_overlay(scan, mask), cv2.COLOR_BGR2RGB)]
        greys = [True, True, False]
        heading = '%s, %s' % (name, model_name)
    else:
        titles = ['The scan as it arrives',
                  'Ground truth, what the expert drew',
                  'Prediction, Dice %.3f' % scores['dice'],
                  'Yellow agree, green missed, red added']
        panels = [scan, (truth * 255).astype(np.uint8),
                  (mask * 255).astype(np.uint8),
                  cv2.cvtColor(draw_error_map(scan, mask, truth), cv2.COLOR_BGR2RGB)]
        greys = [True, True, True, False]
        heading = ('%s, %s: Dice %.3f, precision %.3f, recall %.3f'
                   % (name, model_name, scores['dice'],
                      scores['precision'], scores['recall']))

    figure, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6))
    for axis, image, title, is_grey in zip(axes, panels, titles, greys):
        if is_grey:
            axis.imshow(image, cmap='gray')
        else:
            axis.imshow(image)
        axis.set_title(title, fontsize=10)
        axis.axis('off')
    figure.suptitle(heading, fontsize=12, fontweight='bold')
    figure.tight_layout()
    figure.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(figure)


def run_one(path, model, channels, octf, out_dir, truth_path, from_annotation):
    original = load_single_channel(path)
    resized = cv2.resize(original, (INPUT_WIDTH, INPUT_HEIGHT))

    residual = None
    if channels == 2:
        residual = compute_residual(octf, resized)

    mask, probability = segment(model, channels, resized, residual)

    lesion_pixels = int(mask.sum())
    frame_pixels = INPUT_HEIGHT * INPUT_WIDTH
    percent = 100.0 * lesion_pixels / frame_pixels

    name = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(out_dir, name + '_overlay.png'),
                draw_overlay(resized, mask))
    cv2.imwrite(os.path.join(out_dir, name + '_mask.png'),
                (mask * 255).astype(np.uint8))

    if truth_path is None:
        save_panels(os.path.join(out_dir, name + '_result.png'),
                    resized, mask, None, None, os.path.basename(path), MODEL_NAME[0])
        confident = float(((probability < 0.1) | (probability > 0.9)).mean())
        print("%-30s %8d px  %6.2f%%  %10.0f%%"
              % (os.path.basename(path)[:30], lesion_pixels, percent,
                 100 * confident))
        return None

    truth = load_truth_mask(truth_path, from_annotation)
    # nearest neighbour keeps the mask binary. A smooth resize would invent
    # grey values along the boundary and change the pixel counts the score is
    # computed from.
    truth = cv2.resize(truth, (INPUT_WIDTH, INPUT_HEIGHT),
                       interpolation=cv2.INTER_NEAREST)

    scores = score(mask, truth)
    error_map = draw_error_map(resized, mask, truth)
    cv2.imwrite(os.path.join(out_dir, name + '_error_map.png'),
                draw_legend(error_map, scores))
    save_panels(os.path.join(out_dir, name + '_result.png'),
                resized, mask, truth, scores, os.path.basename(path), MODEL_NAME[0])

    print("%-30s %6.3f %6.3f %6.3f %6.3f %7.1f%% %7.1f%%"
          % (os.path.basename(path)[:30], scores['dice'], scores['iou'],
             scores['precision'], scores['recall'],
             scores['missed'], scores['extra']))
    return scores


def main():
    # Only force a file-writing backend when this runs as a script. Setting it
    # at import time would switch the backend under demo.ipynb, which imports
    # this module, and its inline plots would silently produce nothing.
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser(
        description="Segment laser-induced lesions in a retinal OCT B-scan.")
    parser.add_argument('--image', help="one image file")
    parser.add_argument('--folder', help="a folder of images")
    parser.add_argument('--truth',
                        help="ground truth mask for --image, to score against")
    parser.add_argument('--truth-folder',
                        help="folder of ground truth masks, matched by filename")
    parser.add_argument('--truth-is-annotation', action='store_true',
                        help="the ground truth files are scans with a yellow "
                             "outline drawn on them, not black and white masks")
    parser.add_argument('--model', default='plain', choices=sorted(MODELS),
                        help="which trained model to use (default: plain)")
    parser.add_argument('--out', default='demo_output',
                        help="where to write the overlays (default: demo_output)")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("give either --image or --folder")

    paths = []
    if args.image:
        # say why before loading a model and printing a header
        extension = os.path.splitext(args.image)[1].lower()
        if extension in NATIVE_OCT_FORMATS:
            raise SystemExit("%s\n\n%s" % (
                os.path.basename(args.image),
                EXPORT_ADVICE % NATIVE_OCT_FORMATS[extension]))
        paths.append(args.image)
    skipped = {}
    if args.folder:
        for filename in sorted(os.listdir(args.folder)):
            extension = os.path.splitext(filename)[1].lower()
            if extension in READABLE:
                paths.append(os.path.join(args.folder, filename))
            elif extension in NATIVE_OCT_FORMATS:
                # count these rather than passing over them without a word
                name = NATIVE_OCT_FORMATS[extension]
                skipped[name] = skipped.get(name, 0) + 1

    if skipped:
        for name in sorted(skipped):
            print("Skipped %d %s file(s)." % (skipped[name], name))
        print(EXPORT_ADVICE % sorted(skipped)[0])
        print()

    if not paths:
        raise SystemExit(
            "No readable images found. Readable formats are: %s"
            % ', '.join(READABLE))

    truth_paths = [None] * len(paths)
    if args.truth:
        if not args.image:
            parser.error("--truth goes with --image; use --truth-folder with --folder")
        truth_paths = [args.truth]
    elif args.truth_folder:
        truth_paths = match_truth_files(paths, args.truth_folder)
        found = sum(1 for t in truth_paths if t is not None)
        print("Ground truth matched for %d of %d image(s)." % (found, len(paths)))

    os.makedirs(args.out, exist_ok=True)

    octf = None
    # Two of the five copy their encoder blocks out of the fine-tuned OCT-F, and
    # three need it again to make the residual, so it is loaded for any model
    # that is either built on it or takes two channels.
    if args.model in ('octf', 'modified') or MODELS[args.model][1] == 2:
        print("Loading the fine-tuned OCT-F foundation model...")
        octf = build_octf()

    MODEL_NAME[0] = args.model
    model, channels = build_model(args.model, octf)
    print("Model: %s, %d input channel(s). Writing to %s/\n"
          % (args.model, channels, args.out))

    scoring = any(truth is not None for truth in truth_paths)
    if scoring:
        print("%-30s %6s %6s %6s %6s %8s %8s"
              % ("image", "Dice", "IoU", "Prec", "Rec", "missed", "extra"))
    else:
        print("%-30s %8s %8s %11s"
              % ("image", "lesion", "size", "confidence"))
    print("-" * 78)

    all_scores = []
    for path, truth_path in zip(paths, truth_paths):
        scores = run_one(path, model, channels, octf, args.out,
                         truth_path, args.truth_is_annotation)
        if scores is not None:
            all_scores.append(scores)

    print()
    if not all_scores:
        print("Overlays written to %s/. Yellow marks the predicted lesion."
              % args.out)
        print("Add --truth and it will score itself against your ground truth")
        print("and draw a picture of where it went wrong.")
    else:
        if len(all_scores) > 1:
            mean_dice = sum(s['dice'] for s in all_scores) / len(all_scores)
            mean_iou = sum(s['iou'] for s in all_scores) / len(all_scores)
            mean_precision = sum(s['precision'] for s in all_scores) / len(all_scores)
            mean_recall = sum(s['recall'] for s in all_scores) / len(all_scores)
            mean_missed = sum(s['missed'] for s in all_scores) / len(all_scores)
            mean_extra = sum(s['extra'] for s in all_scores) / len(all_scores)
            print("%-30s %6.3f %6.3f %6.3f %6.3f %7.1f%% %7.1f%%"
                  % ("mean of %d images" % len(all_scores), mean_dice, mean_iou,
                     mean_precision, mean_recall, mean_missed, mean_extra))
            print()
        else:
            mean_precision = all_scores[0]['precision']
            mean_recall = all_scores[0]['recall']

        print("Error maps written to %s/." % args.out)
        print("  yellow  the model and your ground truth agree")
        print("  green   lesion your ground truth marks and the model missed,")
        print("          which is under-segmentation")
        print("  red     tissue the model called lesion and your ground truth")
        print("          does not, which is over-segmentation")
        print()
        if mean_recall < mean_precision:
            print("More green than red: this model under-segments here.")
        elif mean_precision < mean_recall:
            print("More red than green: this model over-segments here.")
        print("For scale, on the 15 validation images of this study the leading")
        print("configuration scored 0.855 Dice at this input size.")

    print()
    print("Note: these models were trained on rat OCT from one study. On any")
    print("other source, read the output as a demonstration, not a measurement.")


if __name__ == '__main__':
    main()
