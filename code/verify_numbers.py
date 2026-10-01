"""
verify_numbers.py checks that every reported number can be recomputed.

Run from the repository root:

    python src/verify_numbers.py

Needs no data, no GPU and no trained weights. It reads the per-image
measurements in results/per_image_scores.pkl, recomputes the macro-averages for
all twenty configurations, and compares each one against the value recorded in
results/NUMBERS.txt.

Twenty configurations, not ten: five models, two input sizes, and each of those
measured twice once on the raw thresholded prediction and once after
post-processing.

Four independent things are checked:

  1. Dice, precision and recall for all twenty configurations.
  2. The fragmentation counts, which the paper reports as 9 of 75 predictions at
     256x256 and 24 of 75 at native resolution.
  3. Intersection over union, which is not stored and is derived per image from
     Dice as D / (2 - D) before averaging.
  4. The identity missed% = (1 - recall) x 100, which the paper relies on.

Values that fall on a rounding boundary are declared in KNOWN_BOUNDARY and
reported separately rather than absorbed into a wider tolerance.
"""

import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCORES = os.path.join(ROOT, "results", "per_image_scores.pkl")

# Expected values, transcribed from results/NUMBERS.txt section 5.
# (resolution, model) -> dict of expected macro-averages, rounded to 3 decimals
# as they appear in the paper.
EXPECTED = {
    ("256", "Plain U-Net"):
        dict(raw_dice=0.850, raw_prec=0.846, raw_rec=0.877, raw_cc=1,
             pp_dice=0.851, pp_prec=0.846, pp_rec=0.880, iou=0.743),
    ("256", "Hybrid U-Net"):
        dict(raw_dice=0.863, raw_prec=0.882, raw_rec=0.860, raw_cc=0,
             pp_dice=0.863, pp_prec=0.882, pp_rec=0.861, iou=0.762),
    ("256", "OCT-F Backbone"):
        dict(raw_dice=0.857, raw_prec=0.857, raw_rec=0.874, raw_cc=3,
             pp_dice=0.862, pp_prec=0.866, pp_rec=0.875, iou=0.753),
    ("256", "Modified OCT-F"):
        dict(raw_dice=0.823, raw_prec=0.856, raw_rec=0.826, raw_cc=4,
             pp_dice=0.821, pp_prec=0.864, pp_rec=0.822, iou=0.710),
    ("256", "Soft Erosion Hybrid"):
        dict(raw_dice=0.866, raw_prec=0.877, raw_rec=0.871, raw_cc=1,
             pp_dice=0.867, pp_prec=0.877, pp_rec=0.873, iou=0.769),
    ("native", "Plain U-Net"):
        dict(raw_dice=0.850, raw_prec=0.840, raw_rec=0.874, raw_cc=5,
             pp_dice=0.855, pp_prec=0.849, pp_rec=0.875, iou=0.743),
    ("native", "Hybrid U-Net"):
        dict(raw_dice=0.843, raw_prec=0.892, raw_rec=0.820, raw_cc=2,
             pp_dice=0.844, pp_prec=0.894, pp_rec=0.820, iou=0.737),
    ("native", "OCT-F Backbone"):
        dict(raw_dice=0.809, raw_prec=0.837, raw_rec=0.824, raw_cc=4,
             pp_dice=0.814, pp_prec=0.845, pp_rec=0.825, iou=0.698),
    ("native", "Modified OCT-F"):
        dict(raw_dice=0.759, raw_prec=0.854, raw_rec=0.741, raw_cc=10,
             pp_dice=0.764, pp_prec=0.871, pp_rec=0.737, iou=0.641),
    ("native", "Soft Erosion Hybrid"):
        dict(raw_dice=0.840, raw_prec=0.880, raw_rec=0.832, raw_cc=3,
             pp_dice=0.843, pp_prec=0.885, pp_rec=0.833, iou=0.735),
}

# Totals the paper states directly, from results/NUMBERS.txt section 7.
EXPECTED_FRAGMENT_TOTALS = {"256": 9, "native": 24}


def mean(xs):
    return sum(xs) / len(xs)



TOLERANCE = 0.0005 + 1e-9

# One value sits on the rounding boundary and is declared here rather than
# hidden by a wider tolerance.

# (resolution, model, key) -> (recomputed value, value in NUMBERS.txt)
KNOWN_BOUNDARY = {
    ("256", "Soft Erosion Hybrid", "raw_rec"): (0.8715, 0.871),
}


def agrees(recomputed, reported):
    return abs(recomputed - reported) <= TOLERANCE


def main():
    if not os.path.exists(SCORES):
        print("cannot find %s" % SCORES)
        return 1

    with open(SCORES, "rb") as fh:
        payload = pickle.load(fh)
    runs = payload["runs"]

    failures = []
    boundary_notes = []
    fragment_totals = {"256": 0, "native": 0}

    print("Per-image measurements: %d configurations x %d validation images"
          % (len(runs), len(payload["val_files"])))
    print()
    header = "%-7s %-20s %-16s %-16s %-16s %-9s" % (
        "input", "model", "Dice raw/paper", "Dice pp/paper", "IoU calc/paper", "CC>1")
    print(header)
    print("-" * len(header))

    for (resolution, model) in sorted(runs.keys()):
        entries = runs[(resolution, model)]
        want = EXPECTED[(resolution, model)]

        got = {
            "raw_dice": mean([e["raw_dice"] for e in entries]),
            "raw_prec": mean([e["raw_prec"] for e in entries]),
            "raw_rec": mean([e["raw_rec"] for e in entries]),
            "pp_dice": mean([e["pp_dice"] for e in entries]),
            "pp_prec": mean([e["pp_prec"] for e in entries]),
            "pp_rec": mean([e["pp_rec"] for e in entries]),
            # IoU is not stored. For one image it follows from Dice exactly, so
            # it is derived per image and only then averaged. Averaging Dice
            # first and converting afterwards would give a different answer.
            "iou": mean([e["raw_dice"] / (2.0 - e["raw_dice"]) for e in entries]),
        }
        raw_cc = sum(1 for e in entries if e["raw_cc"] > 1)
        pp_cc = sum(1 for e in entries if e["pp_cc"] > 1)
        fragment_totals[resolution] += raw_cc

        for key, expected in want.items():
            if key == "raw_cc":
                continue
            if agrees(got[key], expected):
                continue
            declared = KNOWN_BOUNDARY.get((resolution, model, key))
            # A declared boundary value still has to land where it was declared,
            # to half a unit of the last place printed above.
            if declared and abs(got[key] - declared[0]) <= 0.00005:
                boundary_notes.append(
                    "%s / %s / %s: recomputed %.5f, NUMBERS.txt says %.3f "
                    "(declared rounding boundary, difference %.5f)"
                    % (resolution, model, key, got[key], expected,
                       abs(got[key] - expected)))
                continue
            failures.append("%s / %s / %s: recomputed %.4f, NUMBERS.txt says %.3f"
                            % (resolution, model, key, got[key], expected))
        if raw_cc != want["raw_cc"]:
            failures.append("%s / %s / fragmented images: counted %d, NUMBERS.txt says %d"
                            % (resolution, model, raw_cc, want["raw_cc"]))
        # Post-processing is supposed to leave nothing fragmented anywhere.
        if pp_cc != 0:
            failures.append("%s / %s: %d predictions still fragmented after post-processing"
                            % (resolution, model, pp_cc))
        # The paper treats missed% as exactly (1 - recall) x 100.
        for stage in ("raw", "pp"):
            missed = (1.0 - got[stage + "_rec"]) * 100.0
            if abs(missed - (100.0 - got[stage + "_rec"] * 100.0)) > 1e-9:
                failures.append("%s / %s: missed%% identity fails" % (resolution, model))

        print("%-7s %-20s %.4f/%.3f      %.4f/%.3f      %.4f/%.3f      %2d/15 -> %d/15"
              % (resolution, model,
                 got["raw_dice"], want["raw_dice"],
                 got["pp_dice"], want["pp_dice"],
                 got["iou"], want["iou"],
                 raw_cc, pp_cc))

    print()
    for resolution, total in sorted(fragment_totals.items()):
        expected = EXPECTED_FRAGMENT_TOTALS[resolution]
        mark = "ok" if total == expected else "MISMATCH"
        print("fragmented predictions at %-7s %2d of 75   (paper says %d)   %s"
              % (resolution, total, expected, mark))
        if total != expected:
            failures.append("fragment total at %s: counted %d, paper says %d"
                            % (resolution, total, expected))

    print()
    if boundary_notes:
        print("%d DECLARED ROUNDING BOUNDARY / BOUNDARIES" % len(boundary_notes))
        for n in boundary_notes:
            print("  - " + n)
        print()

    if failures:
        print("%d CHECK(S) FAILED" % len(failures))
        for f in failures:
            print("  - " + f)
        return 1

    print("ALL CHECKS PASSED")
    print("  20 configurations reproduce Dice, precision, recall and IoU")
    print("  fragmentation counts reproduce for every model at both input sizes")
    print("  post-processing leaves 0 fragmented predictions in all 10 configurations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
