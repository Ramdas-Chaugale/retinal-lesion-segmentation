# Retinal Lesion Segmentation in OCT

Finding laser-induced lesions in retinal OCT B-scans, automatically.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![Licence](https://img.shields.io/badge/Licence-MIT-green)

Measuring these lesions is done by hand: a scientist outlines each one on screen, scan by scan.
This replaces that with five trained segmentation networks, and answers the question the manual
process cannot: which of them is actually better, and by how much.

**Headline result: 0.850 Dice on held-out scans.** The more interesting result is that the two
ideas expected to improve on that did not, and the repository reports why.

---

## See it work in one second

```
pip install -r requirements.txt
python code/verify_numbers.py
```

No data needed, no GPU, no download. It recomputes Dice, precision, recall and IoU for all twenty
configurations from the stored per-image scores, checks every one against `results/NUMBERS.txt`,
and reproduces the fragmentation counts.

```
ALL CHECKS PASSED
  20 configurations reproduce Dice, precision, recall and IoU
  fragmentation counts reproduce for every model at both input sizes
  post-processing leaves 0 fragmented predictions in all 10 configurations
```

Every number quoted below comes out of that script. None of them is typed in by hand.

## Results

Full resolution, 496 x 768, on 15 held-out scans.

| Model | Inputs | Parameters | Dice | Precision | Recall | Fragmented |
|---|---|---|---:|---:|---:|---:|
| Plain U-Net | image | 7.77 M | **0.850** | 0.840 | 0.874 | 5 / 15 |
| Hybrid U-Net | image + residual | 7.77 M | 0.843 | **0.892** | 0.820 | 2 / 15 |
| Soft Erosion Hybrid | image + residual | 7.77 M | 0.840 | 0.880 | 0.832 | 3 / 15 |
| OCT-F Backbone | image | 1.98 M | 0.809 | 0.837 | 0.824 | 4 / 15 |
| Modified OCT-F 2ch | image + residual | 1.98 M | 0.759 | 0.854 | 0.741 | 10 / 15 |

Post-processing reduces fragmented predictions to 0 / 15 in every configuration, and lifts Plain
U-Net from 0.850 to 0.855 Dice.

Worth noting the third row: the pretrained encoder reaches 0.809 with **a quarter of the
parameters**, 1.98 M against 7.77 M, which matters more than the 0.04 Dice it gives up if this ever
has to run somewhere small.

![Training curves for all five models](figures/03_training_curves_all.png)

## What the results actually say

This is the part most model comparisons leave out.

**No model here is "the best".** There are 15 validation images, so the standard error is about
0.013, and two configurations closer than roughly 0.026 cannot be told apart. Nearly every gap in
the table above is smaller than that. Ranking them would be reading noise.

**One image decides which model appears to win.** On a single held-out scan four of the five models
collapse while the Plain U-Net holds at 0.802. Remove that one image and the leader moves from
first place to third. That is reported here rather than quietly dropped.

**Two results are negative, and that was the point of running them.** Adding a reconstruction
residual as a second input channel does carry a lesion signal, but a weaker one than the image
already carries, so it moves error around instead of removing it. Compare the first two rows of the
table: precision climbs from 0.840 to 0.892 while recall drops from 0.874 to 0.820. Overlap barely
changes, the errors just swap sides. Putting a connectivity penalty inside the loss function does
not work either, while applying the same idea as post-processing after training does.

A negative result that is measured and explained is worth more than a positive one that is not
reproducible.

## How it works

**Face the problem where it is.** A lesion is a local disruption of retinal layers, so the models
are ordinary encoder-decoder segmentation networks rather than anything exotic. Two of the five
reuse a pretrained encoder from a retinal reconstruction model, at a quarter of the parameters.

**The residual idea.** A reconstruction model trained on healthy-baseline scans should reconstruct
lesioned tissue badly. The difference between the scan and its reconstruction is therefore a map of
where the retina deviates from expectation. Feeding that alongside the image gives the network a
second view of the same tissue. It helps precision and costs recall, which the evaluation quantifies
rather than asserts.

**Connectivity.** A laser lesion is one connected region, so a prediction broken into fragments is
wrong by construction. Two approaches were tried: a differentiable erosion penalty inside the loss,
and morphological post-processing afterwards. Only the second works.

**Scoring.** Dice, IoU, precision, recall, connected-component counts, and the split between missed
lesion and wrongly included tissue, per image rather than averaged early.

## Repository layout

```
code/                     the modules everything else imports
  architectures.py          the five networks
  datasets.py               how images and residuals are fed to them
  losses.py                 DiceBCELoss and SoftErosionLoss
  postprocessing.py         close small gaps, then keep the largest region
  train.py                  training loop, early stopping, checkpointing
  evaluate.py               per-image scoring
  verify_numbers.py         recomputes every reported number

notebooks/                the analysis, 02 to 05
gpu_notebooks/            the three training runs
demo/demo.py              runs a trained model on a scan
figures/                  training curves
results/                  NUMBERS.txt and the per-image scores
```

Notebooks 03 and 04 have one switch at the top, `RESOLUTION = 'native'` or `'256'`.

## Running this on your own data

Nothing here is tied to a particular dataset. The loaders take folder paths and a list of
filenames, so any set of images with matching masks will work.

**What the folders need to look like.** One folder of scans and one folder of masks. A mask is
found by swapping the extension, so `anything.tif` is paired with `anything_mask.png` in the mask
folder. Masks are read greyscale and thresholded at 127, so anything binary will do.

```python
from datasets import SingleChannelDataset, HybridDataset

train = SingleChannelDataset(image_folder, mask_folder, file_list,
                             img_size=(496, 768), augment=True)
```

`img_size` takes an int for a square resize or a `(height, width)` pair to keep native geometry.
The two-channel models use `HybridDataset`, which additionally takes a dict of precomputed
reconstruction residuals keyed by filename; without a reconstruction model, use the one-channel
models, which are the leading configuration anyway.

**What to change for a different image size.** The post-processing kernel is the one setting that
does not scale itself. `improved_post_processing` closes gaps with an elliptical kernel sized 15 at
256 x 256 and 29 at 496 x 768, so scale it roughly with the image and keep it odd.

**Three assumptions worth checking against your own scans.** Images are expected to be greyscale
stored as three identical channels, and a single channel is taken by indexing rather than averaging;
if your channels genuinely differ, that line needs changing. The connectivity post-processing keeps
only the largest connected region, which is right for a single lesion and wrong for anything
multi-focal. And the reported numbers come from 15 validation images, so treat them as an
illustration of the method rather than a benchmark.

**Just want to see a model run?** `demo/demo.py` takes any PNG, JPG or TIF and writes an overlay,
and scores itself if you pass a ground-truth mask.

## What is not in this repository, and why

**The imaging data.** The imaging data used in this study is private. No scan, no annotation, and
no model weight trained on them appears here. The notebooks are stored without their outputs for
the same reason, since those outputs contained rendered scans. Scan identifiers in
`results/` are anonymised, and the loaders take a folder path rather than any particular filename,
so the code runs against any similarly organised set of images and masks.

**The reconstruction foundation model.** OCT-F, by Haider et al. at the School of EEECS, Queen's
University Belfast, is withheld until its authors publish. A few notebook cells import it by name
and will only run for someone who already has it. The five segmentation networks in
`code/architectures.py` take the pretrained encoder as a constructor argument rather than importing
it, so that file stands alone.

> A. Haider, D. Wright, R. Hogg, H. Wang, T. Peto and R. Gault, "Combining self-attention and
> convolution to understand OCT images," in *Proc. 27th Irish Machine Vision and Image Processing
> Conference (IMVIP)*, Derry, U.K., 2025, pp. 202–206.
> [Publication record](https://research.tees.ac.uk/en/publications/combining-self-attention-and-convolution-to-understand-oct-images/)

What remains is enough to read the method, audit the arithmetic, and reuse the general parts.

## Scope

These weights were fitted to preclinical OCT from a single study, annotated by one person. Applied
to anything else, a number from this repository is an illustration of the method and not a result.

## Ethics

**No animal procedures were carried out for this project.** The scans predate it. They were
collected by the data provider for their own earlier studies, under the regulatory framework
governing animal research, and this work is a secondary analysis of images that already existed.

The motivation runs the other way as well. Manual lesion measurement carries observer variability,
and that variability is part of the sample size a study needs to reach a given statistical power.
Measuring the same lesions more consistently reduces the number of animals a future study requires.

This is research code. It is not a diagnostic device, it has not been validated on human scans, and
nothing here should be used to inform a clinical decision.

## About this project

Built as the MSc Artificial Intelligence dissertation at Queen's University Belfast, in
collaboration with an industry partner working in preclinical retinal disease, who provided the
imaging data and the expert annotations.

Supervised by **Dr Richard Gault**, School of Electronics, Electrical Engineering and Computer
Science, Queen's University Belfast.

Author: **Ramdas Narayan Chaugale**

## Acknowledgements

The imaging data and the expert lesion annotations were provided by the industry partner, whose
permission to use this unpublished data is gratefully acknowledged.

The OCT-F architecture and its pretrained weights are the work of Haider et al., developed within
the School of EEECS at Queen's University Belfast, and are used here with their permission.

Closest prior work in this group, and a useful companion to the comparison made here:

> A. Baine, A. Haider, R. Hogg and R. Gault, "Deep learning based segmentation of retinal layers
> from OCT imaging," in *Proc. 27th Irish Machine Vision and Image Processing Conference (IMVIP)*,
> Derry, U.K., 2025, pp. 215–218.

Training was run on the Kelvin2 cluster, provided by the Northern Ireland High Performance
Computing service at Queen's University Belfast.

## Licence

MIT, see [LICENSE](LICENSE). This covers the code in this repository. It does not extend to the
private imaging data or to OCT-F, neither of which is distributed here.
