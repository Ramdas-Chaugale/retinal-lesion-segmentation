"""Build the three figures used in the README.

    python code/make_figures.py

The first two are diagrams of the method and of the five models. The third is
built from results/per_image_scores.csv, so it always matches the numbers that
verify_numbers.py checks.

No figure shows a scan. The imaging data is private.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "savefig.facecolor": "white",
})

INK = "#1b1b1b"
MUTED = "#6b7280"
EDGE = "#c9ced6"
MODEL_COLOURS = {
    "Plain U-Net": "#1f77b4",
    "Hybrid U-Net": "#d62728",
    "OCT-F Backbone": "#2ca02c",
    "Modified OCT-F 2ch": "#ff7f0e",
    "Soft Erosion Hybrid": "#9467bd",
}
ORDER = list(MODEL_COLOURS)


def box(ax, x, y, w, h, text, face, edge=EDGE, fontsize=10, weight="normal",
        textcolour=INK):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face, edgecolor=edge, linewidth=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolour, weight=weight, linespacing=1.45)


def arrow(ax, start, end, colour="#8a9099"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.5, color=colour, shrinkA=2, shrinkB=2))


# ---------------------------------------------------------------- figure 1
def pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    ax.text(0.1, 5.05, "Method", fontsize=15, weight="bold", color=INK)
    ax.text(0.1, 4.72,
            "Two inputs are prepared, five models are trained on the same split, "
            "and every model is scored per image.",
            fontsize=10.5, color=MUTED)

    data_f, prep_f, model_f, eval_f = "#eef2f7", "#e6f0ea", "#eceaf6", "#fdf0e4"

    # row 1: the residual branch
    box(ax, 0.1, 3.05, 2.5, 1.0,
        "Baseline scans\nhealthy retina", data_f)
    box(ax, 3.0, 3.05, 2.6, 1.0,
        "Fine-tune\nreconstruction model", prep_f)
    box(ax, 6.0, 3.05, 2.5, 1.0,
        "Residual\nscan minus rebuild", prep_f)

    arrow(ax, (2.6, 3.55), (3.0, 3.55))
    arrow(ax, (5.6, 3.55), (6.0, 3.55))

    # row 2: the image branch
    box(ax, 0.1, 1.45, 2.5, 1.0,
        "Endpoint scans\nwith lesions", data_f)
    box(ax, 3.0, 1.45, 2.6, 1.0,
        "Expert outlines\nfilled to masks", prep_f)

    arrow(ax, (2.6, 1.95), (3.0, 1.95))

    # the two branches meet
    box(ax, 6.0, 1.45, 2.5, 1.0, "Image\n1 or 2 channels", prep_f)
    arrow(ax, (5.6, 1.95), (6.0, 1.95))
    arrow(ax, (7.25, 3.05), (7.25, 2.45))

    # training and evaluation
    box(ax, 9.0, 2.25, 2.1, 1.6,
        "Five\nsegmentation\nmodels", model_f, fontsize=10.5, weight="bold")
    arrow(ax, (8.5, 3.55), (9.0, 3.2))
    arrow(ax, (8.5, 1.95), (9.0, 2.6))

    box(ax, 11.5, 2.75, 1.9, 0.85, "Per-image\nscoring", eval_f, fontsize=10)
    box(ax, 11.5, 1.6, 1.9, 0.85, "Largest-region\npost-processing", eval_f,
        fontsize=9.5)
    arrow(ax, (11.1, 3.3), (11.5, 3.18))
    arrow(ax, (11.1, 2.8), (11.5, 2.03))

    ax.text(0.1, 0.78, "Held out: 15 scans never seen in training, "
                       "scored individually and only then averaged",
            fontsize=10, color=MUTED, style="italic")
    ax.text(0.1, 0.35, "No scan, annotation or trained weight is distributed "
                       "with this repository",
            fontsize=9.5, color="#9aa0a8")

    fig.savefig(FIGS / "pipeline.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/pipeline.png")


# ---------------------------------------------------------------- figure 2
def architectures():
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(0.1, 5.85, "The five models", fontsize=15, weight="bold", color=INK)
    ax.text(0.1, 5.5,
            "Same encoder-decoder skeleton throughout. What changes is the input, "
            "where the encoder comes from, and the loss.",
            fontsize=10.5, color=MUTED)

    specs = [
        ("Plain U-Net", "image", "random init", "7.77 M", "Dice + BCE"),
        ("Hybrid U-Net", "image + residual", "random init", "7.77 M", "Dice + BCE"),
        ("OCT-F Backbone", "image", "pretrained", "1.98 M", "Dice + BCE"),
        ("Modified OCT-F 2ch", "image + residual", "pretrained", "1.98 M", "Dice + BCE"),
        ("Soft Erosion Hybrid", "image + residual", "random init", "7.77 M", "+ erosion penalty"),
    ]

    w, gap = 2.45, 0.22
    for i, (name, inp, enc, params, loss) in enumerate(specs):
        x = 0.1 + i * (w + gap)
        colour = MODEL_COLOURS[name]

        ax.add_patch(FancyBboxPatch(
            (x, 1.15), w, 4.05,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor="white", edgecolor=colour, linewidth=1.8))
        ax.add_patch(FancyBboxPatch(
            (x, 4.55), w, 0.65,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor=colour, edgecolor=colour, linewidth=1.8))
        ax.text(x + w / 2, 4.875, name, ha="center", va="center",
                fontsize=10, weight="bold", color="white")

        # input channels, drawn as stacked bars
        n_ch = 2 if "residual" in inp else 1
        start = x + w / 2 - (n_ch * 0.34 + (n_ch - 1) * 0.08) / 2
        for c in range(n_ch):
            ax.add_patch(FancyBboxPatch(
                (start + c * 0.42, 4.0), 0.34, 0.34,
                boxstyle="round,pad=0.006,rounding_size=0.02",
                facecolor=colour if c == 0 else "#ffffff",
                edgecolor=colour, linewidth=1.4))
        ax.text(x + w / 2, 3.78, inp, ha="center", va="center",
                fontsize=8.8, color=MUTED)

        # encoder / decoder blocks, narrowing then widening
        for j, hgt in enumerate([0.46, 0.38, 0.30, 0.22]):
            ax.add_patch(FancyBboxPatch(
                (x + 0.28 + j * 0.2, 3.10 - hgt / 2), 0.15, hgt,
                boxstyle="round,pad=0.004,rounding_size=0.01",
                facecolor=colour if enc == "pretrained" else "#dfe4ea",
                edgecolor="none", alpha=0.95 if enc == "pretrained" else 1))
        for j, hgt in enumerate([0.22, 0.30, 0.38, 0.46]):
            ax.add_patch(FancyBboxPatch(
                (x + 1.28 + j * 0.2, 3.10 - hgt / 2), 0.15, hgt,
                boxstyle="round,pad=0.004,rounding_size=0.01",
                facecolor="#dfe4ea", edgecolor="none"))
        ax.text(x + 0.52, 2.60, "encoder", fontsize=8, color=MUTED)
        ax.text(x + 1.56, 2.60, "decoder", fontsize=8, color=MUTED)

        ax.text(x + w / 2, 2.25,
                "%s encoder" % enc, ha="center", fontsize=9,
                color=colour if enc == "pretrained" else MUTED,
                weight="bold" if enc == "pretrained" else "normal")
        ax.text(x + w / 2, 1.85, params, ha="center", fontsize=12,
                weight="bold", color=INK)
        ax.text(x + w / 2, 1.5, loss, ha="center", fontsize=8.8, color=MUTED)

    ax.text(0.1, 0.62,
            "Pretrained encoders come from a retinal reconstruction model and reach "
            "comparable accuracy with a quarter of the parameters.",
            fontsize=10, color=MUTED)
    ax.text(0.1, 0.25,
            "The residual is the difference between a scan and its reconstruction, "
            "supplied as a second input channel.",
            fontsize=10, color=MUTED)

    fig.savefig(FIGS / "architectures.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/architectures.png")


# ---------------------------------------------------------------- figure 3
def distributions():
    rows = list(csv.DictReader(open(REPO / "results" / "per_image_scores.csv",
                                    encoding="utf-8")))
    native = [r for r in rows if r["resolution"] == "native"]
    raw = defaultdict(list)
    pp = defaultdict(list)
    # the csv labels the fourth model without its "2ch" suffix
    csv_to_display = {"Modified OCT-F": "Modified OCT-F 2ch"}
    for r in native:
        name = csv_to_display.get(r["model"], r["model"])
        raw[name].append(float(r["raw_dice"]))
        pp[name].append(float(r["pp_dice"]))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
    fig.suptitle("Per-image Dice on the 15 held-out scans, full resolution",
                 fontsize=14, weight="bold", color=INK, x=0.012, ha="left", y=1.0)
    fig.text(0.012, 0.935,
             "Every point is one scan. The spread inside each model is far wider "
             "than the gaps between the models, which is why no model here is "
             "called the best.",
             fontsize=10.5, color=MUTED, ha="left")

    for ax, data, title in ((axes[0], raw, "Raw predictions"),
                            (axes[1], pp, "After post-processing")):
        for i, name in enumerate(ORDER):
            vals = data[name]
            colour = MODEL_COLOURS[name]
            parts = ax.boxplot(vals, positions=[i], widths=0.52,
                               patch_artist=True, showfliers=False,
                               medianprops=dict(color=colour, linewidth=2),
                               boxprops=dict(facecolor=colour, alpha=0.16,
                                             edgecolor=colour, linewidth=1.3),
                               whiskerprops=dict(color=colour, linewidth=1.2),
                               capprops=dict(color=colour, linewidth=1.2))
            jitter = [i + (k % 5 - 2) * 0.055 for k in range(len(vals))]
            ax.scatter(jitter, vals, s=26, color=colour, alpha=0.8,
                       edgecolor="white", linewidth=0.6, zorder=3)
            ax.text(i, 1.035, "%.3f" % (sum(vals) / len(vals)), ha="center",
                    fontsize=9.5, weight="bold", color=colour)

        ax.set_title(title, fontsize=12, weight="bold", color=INK, pad=22)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([n.replace(" ", "\n", 1) for n in ORDER], fontsize=9)
        ax.set_ylim(-0.03, 1.09)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("Dice", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGS / "score_distributions.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/score_distributions.png")


pipeline()
architectures()
distributions()
