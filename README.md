# agloss: anatomical guidance and confinement of Grad-CAM

Code and result files for the manuscript

> N. Aslam, G. Mustafa, A. N. Qureshi, U. A. Raza. *Guidance versus Confinement: What Anatomical Supervision of Grad-CAM Does and Does Not Change.* Manuscript under review.

Software archive: [doi:10.5281/zenodo.23217035](https://doi.org/10.5281/zenodo.23217035) (v1.0.0; all versions: [doi:10.5281/zenodo.23217034](https://doi.org/10.5281/zenodo.23217034)).

The package trains tuberculosis chest X-ray classifiers with and without spatial supervision of Grad-CAM, evaluates them in-domain and across datasets (Shenzhen ↔ Montgomery), and measures how much each model actually relies on the lungs. Two commands run every experiment in the paper (336 training runs: ResNet-50 and ConvNeXt-Tiny, two datasets, eight seeds). The result CSVs in `results/` rebuild every table and figure without a GPU.

Contents:

```
agloss/            the package (losses, training, metrics, statistics, figures, evaluation suite)
tests/             unit tests (CPU, no data needed)
run_all.py         round 1: ResNet-50 main grid, ablations, masked input, mask perturbation, analysis
run_option2.py     round 2: loss variants, second backbone, evaluation-only analyses
results/           the result files reported in the paper (see "Result files")
paper_tables/      make_tables.py: builds the LaTeX tables of the paper from results/
```

## Reproduce the tables from the included results (no GPU)

```bash
cd paper_tables
python make_tables.py
```

This needs only the Python standard library. It writes the table bodies to `paper_tables/tables/` and the supplementary tables to `paper_tables/supplementary.tex`. The figures of the paper are in `results/figures/`.

## 1. Install

Tested with Python 3, PyTorch 2.11 (CUDA 12.8) on an NVIDIA RTX 5070. Install PyTorch first with the CUDA build that matches your GPU (see pytorch.org); RTX 50-series cards need CUDA 12.8 or later.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Check that the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Run the tests (about a minute, CPU)

```bash
pytest -q
```

The tests cover the losses, metrics, splits, schedules, Grad-CAM, and a tiny end-to-end run on synthetic images.

## 3. Get the data

The datasets are not redistributed here.

| Dataset | Source |
|---|---|
| Shenzhen and Montgomery County chest X-rays | U.S. National Library of Medicine: https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html |
| Shenzhen lung masks (Stirenko et al.) | https://www.kaggle.com/datasets/yoctoman/shcxr-lung-mask (free Kaggle account) |

`run_all.py` downloads and arranges the NLM data itself. The Kaggle masks need an account (or the `kaggle` command-line tool with an API token): if the script stops and asks for them, download the zip from the link it prints, put it in the data folder and run the same command again. Montgomery ships its own masks (`ManualMask/leftMask` and `rightMask`), which the loader merges.

Expected layout:

```
Shenzhen/
  CXR_png/CHNCXR_0001_0.png ...        (_0 = Normal, _1 = TB)
  mask/CHNCXR_0001_0_mask.png ...
Montgomery/
  CXR_png/MCUCXR_0001_0.png ...
  ManualMask/leftMask/*.png  and  ManualMask/rightMask/*.png
```

On first use each dataset is resized to 224×224 and cached in `cache/`. The loader prints the image counts: 566 masked Shenzhen images (279 Normal / 287 TB) and 138 Montgomery images (80 / 58).

## 4. Run the experiments

Both commands resume where they stopped if interrupted.

```bash
python run_all.py --data-root /path/to/data
python run_option2.py --data-root /path/to/data --archs convnext_tiny
```

- `run_all.py`: ResNet-50, the six main configurations, clamp-only ablation, ρ sweep, lung-masked input, mask erosion/dilation; then statistics, figures, the Grad-CAM grid and the occlusion analysis.
- `run_option2.py`: the stricter guidance variants (top-k outside penalty, saliency mass) on ResNet-50; Focal, Focal+Tversky, AG-only and Full on ConvNeXt-Tiny; then the evaluation-only analyses (region ablation, occlusion, deletion-insertion, bootstrap intervals, per-class and clinical metrics) for every backbone.

Use `--quick` (3 epochs, one seed) to check the pipeline end to end first, and `--seeds 42-45` to run fewer seeds. Each run saves a checkpoint of about 100 MB; the evaluation-only analyses need them.

Single runs can be started directly, for example:

```bash
python -m agloss.run --methods full --datasets shenzhen --seeds 42 --shenzhen-root DATA/Shenzhen --montgomery-root DATA/Montgomery
```

## Configurations

| Code name | Paper label | Loss terms |
|---|---|---|
| `vanilla` | Vanilla | cross-entropy |
| `focal` | Focal | focal |
| `focal_tversky` | Focal+Tversky | focal + Tversky (confinement) |
| `anat_only` | AG-only | focal + AG, ρ = 0.75, clamp (guidance) |
| `full` | Full | focal + Tversky + AG, ρ = 0.75, clamp |
| `sym_noclamp` | Sym-AG | focal + Tversky + AG, ρ = 1, no clamp |
| `full_noclamp` | Full (no clamp) | Full without the clamp |
| `anat_only_rho025/050/100` | AG-only ρ = 0.25/0.50/1.00 | AG-only with a different ρ |
| `anat_only_topk` | AG-only (top-k outside) | AG-only, outside term on the top-k outside activations |
| `anat_only_mass` | AG-only (saliency mass) | AG-only on mass-normalised Grad-CAM |
| `masked_input` | Masked input | focal, input multiplied by the lung mask (train and test) |
| `*_erode`, `*_dilate` | eroded / dilated masks | Focal+Tversky or AG-only trained on masks eroded or dilated by 8 px, evaluated against the original masks |

Backbones: `--arch resnet50` (default), `convnext_tiny`. `densenet121` is implemented but was not used in the paper.

## Implementation details

- Loss weights λ_tv = λ_anat = 0.5; focal class weights (0.4, 0.6) for every configuration except Vanilla.
- Adam with L2 weight decay 1e-4; global gradient-norm clipping at 1.0.
- The best checkpoint is chosen by validation AUC among epochs ≥ 25.
- Batch-norm running statistics at evaluation (`model.eval()`).
- torchvision ImageNet weights (`IMAGENET1K_V1` for ResNet-50).
- Augmentation: horizontal flip, ±10 px shift, ±5% scale about the image centre.
- Splits: stratified 80/20 train/test, then one of five stratified folds of the training part as validation. Hyperparameters were chosen on training and validation data only.

## Result files

```
results/summary.csv            mean ± SD over seeds of AUC, Energy Ratio (ER), Pointing Game (PG), CAM-IoU,
                               per condition (in-domain, transfer) and configuration (ResNet-50)
results/paired_tests.csv       seed-paired comparisons between configurations
results/occlusion.csv          occlusion-based Energy Ratio (an explainer not optimised during training)
results/figures/               figures of the paper and tables.tex
results/eval/per_image_<arch>.csv   one row per image, seed, configuration and evaluation set: p_tb, ER, PG, IoU,
                               predictions with lungs or outside removed (zero / blur), occlusion ER at 16/32/64 px
                               patches, deletion/insertion AUCs for Grad-CAM and random orderings
results/eval/reliance_<arch>.csv    per condition and configuration: AUC with lungs or outside removed (zero / blur),
                               occlusion ER, deletion/insertion AUCs and gains
results/eval/bootstrap_<arch>.csv   bootstrap confidence intervals (incl. the post-hoc non-inferiority check)
results/eval/byclass_<arch>.csv     alignment metrics by class
results/eval/clinical_<arch>.csv    AUC, AUPRC, sensitivity at 90% specificity, specificity at 90% sensitivity, Brier score
```

The ConvNeXt-Tiny numbers in the paper are computed from the `results/eval/*_convnext_tiny.csv` files.

Checkpoints and per-seed training logs (`results/<method>/<dataset>/<arch>/per_seed.csv`, `seed_<N>_history.json`, `seed_<N>_best.pt`) are written by the commands above but are not included here because of their size.

## License and citation

MIT License (see `LICENSE`). If you use this code, please cite the software record, https://doi.org/10.5281/zenodo.23217035 (see `CITATION.cff`), and the paper.
