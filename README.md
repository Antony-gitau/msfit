# Multi-Stage Fine-Tuning of Pathology Foundation Models with Head-Diverse Ensembling for White Blood Cell Classification (ISBI 2026)

Official code for our paper: arXiv - https://arxiv.org/abs/2603.20383

```bibtex
@inproceedings{Gitau2026ISBI,
  author    = {Antony Gitau and Martin Paulson and Bjørn-Jostein Singstad and Karl Thomas Hjelmervik and Ola Marius Lysaker and Veralia Gabriela Sanchez},
  title     = {Multi-Stage Fine-Tuning of Pathology Foundation Models with Head-Diverse Ensembling for White Blood Cell Classification},
  booktitle = {2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI)},
  year      = {2026},
  organization = {IEEE}
}
```

## Method Overview

![msfit system architecture](assets/system_architecture_best_method_isbi.png)

During training, separate DINOBloom-based models are obtained by end-to-end fine-tuning the backbone with different classifier heads (linear, cosine, and MLP) across staged optimization. During inference, saved checkpoints are combined using a selective head-diverse ensemble, where an MLP head acts as the primary predictor and is conditionally overridden by agreement between auxiliary heads.

## Quick Start

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Run the published hybrid reproduction:

```bash
RUN_NAME="repro_baseline_v1"
bash reproduce_best_submission.sh \
  --data-root /path/to/wbcbench-2026-data \
  --python python3 \
  --run-name "$RUN_NAME"
```

This writes outputs into `code_reproduction/$RUN_NAME/`, including:

- `submission_best_reproduced.csv`
- `repro_metadata.txt`
- run logs
- intermediate checkpoint folders

## Repository Layout

```text
msfit/
  README.md
  assets/
    system_architecture_best_method_isbi.png
  experiments/
    train_mlp.sh
    train_linear.sh
    train_cosine.sh
    train_decoupled_c1.sh
  label_review/
    train_top3_predictions.csv
    val_top3_predictions.csv
  submissions/
    submission_dinobloom_v4_mlp_s3_tta.csv
    submission_dinobloom_v4_linear_s3_tta.csv
    submission_dec_bestlb_cls_c1_wbcc_tta.csv
    submission_dinobloom_v4_cosine_s3_tta.csv
    submission_mlp_anchor_r2_c1cos_pospair.csv
  reproduce_best_submission.sh
  train.py
  inference.py
  modeling.py
  scripts/
    build_conservative_hybrid_submission.py
  requirements.txt
```

## Published Results

| Component | Role | Training setup | Eval macro-F1 | Leaderboard |
|---|---|---|---:|---:|
| `dinobloom_v4_mlp_s3 + TTA` | Anchor | `dinobloom_base + mlp`, full fine-tuning, S1->S2->S3 | 0.7184 | 0.67584 |
| `dinobloom_v4_linear_s3 + TTA` | Linear reference | `dinobloom_base + linear`, full fine-tuning, S1->S2->S3 | 0.7281 | 0.66870 |
| `dec_bestlb_cls_c1_wbcc + TTA` | Advisor 1 | `dinobloom_base + mlp`, frozen backbone, head-only fine-tuning | 0.7289 | 0.67540 |
| `dinobloom_v4_cosine_s3 + TTA` | Advisor 2 | `dinobloom_base + cosine`, full fine-tuning, S1->S2->S3 | 0.7225 | 0.66085 |
| `submission_mlp_anchor_r2_c1cos_pospair` | Final hybrid | Anchor plus conservative advisor agreement overrides | 0.7217 | **0.67658** |

The submission CSVs for these models are included in `submissions/`.

The primary predictor checkpoint is available on Hugging Face:

- https://huggingface.co/AntonyG/msfit-dinobloom-v4-mlp-s3

## Label Review Release

The repository includes the reviewed top-3 prediction files used for the paper's expert label-review analysis:

- `label_review/train_top3_predictions.csv`
- `label_review/val_top3_predictions.csv`

These files contain image id, assigned label, top-3 model predictions with probabilities, confidence margin, and expert review annotations.

## Reproduce the Published Submission

The default reproduction script rebuilds the published hybrid workflow and refuses to overwrite an existing run folder.

```bash
RUN_NAME="repro_baseline_v1"
bash reproduce_best_submission.sh \
  --data-root /path/to/wbcbench-2026-data \
  --python python3 \
  --run-name "$RUN_NAME"
```

For a more repeatable rerun, pass `--deterministic 1`:

```bash
RUN_NAME="repro_baseline_det_v1"
bash reproduce_best_submission.sh \
  --data-root /path/to/wbcbench-2026-data \
  --python python3 \
  --run-name "$RUN_NAME" \
  --deterministic 1
```

## Single-Model Commands

If you want the main published branches without running the full hybrid reproduction, use the wrappers in [`experiments/`](./experiments):

```bash
bash experiments/train_mlp.sh --data-root /path/to/wbcbench-2026-data
bash experiments/train_linear.sh --data-root /path/to/wbcbench-2026-data
bash experiments/train_cosine.sh --data-root /path/to/wbcbench-2026-data
bash experiments/train_decoupled_c1.sh \
  --data-root /path/to/wbcbench-2026-data \
  --checkpoint /path/to/dinobloom_v4_mlp_s3/best.pth
```

These wrappers create fresh run directories under `code_reproduction/` and keep the published schedules for each branch.

## Environment Notes

Requirements:

- Python 3.10+
- PyTorch 2.x with CUDA
- `timm>=1.0`
- WBCBench 2026 dataset from the challenge organizers
- Internet access or cached Hugging Face weights for DINOBloom and any alternate backbones you use

The framework supports more than the winning recipe. `train.py` and `inference.py` also expose alternate backbones, head types, loss functions, samplers, TTA, and frozen-vs-full fine-tuning configurations.

## Supported Backbones

The framework supports any timm-compatible model, not just DINOBloom:

| Backbone | Type | Params | Pretrained On |
|---|---|---|---|
| `dinobloom_base` | ViT-B/14 | 86M | Hematology (blood + bone marrow) |
| `dinobloom_large` | ViT-L/14 | 307M | Hematology |
| `dinobloom_giant` | ViT-g/14 | 1.1B | Hematology |
| `uni2-h` | ViT-g/14 | 1.5B | General pathology (WSI) |
| `virchow2` | ViT-H/14 | 632M | Tumor pathology (H&E) |
| `maxvit_base_tf_384.in21k_ft_in1k` | MaxViT (hybrid) | 119M | ImageNet-21K |
| `convnextv2_large.fcmae_ft_in22k_in1k_384` | ConvNeXt V2 | 198M | ImageNet-22K |
| Any timm model name | varies | varies | varies |


