# Multi-Stage Fine-Tuning of Pathology Foundation Models with Head-Diverse Ensembling for White Blood Cell Classification


## Method Overview

![msfit system architecture](assets/system_architecture_best_method_isbi.png)

End-to-end fine-tuning and inference simplified visual illustration. During training, separate models are obtained by fine-tuning a pathology foundation model (DINOBloom) with different classifier heads (linear, cosine, and MLP) across staged optimization. During inference, saved checkpoints are combined using a selective head-diverse ensemble, where an MLP head acts as the primary predictor and is conditionally overridden by agreement between auxiliary heads.

## Layout

```text
msfit/
  README.md
  assets/
    system_architecture_best_method.png
  submissions/
    submission_dinobloom_v4_mlp_s3_tta.csv
    submission_dinobloom_v4_linear_s3_tta.csv
    submission_dec_bestlb_cls_c1_wbcc_tta.csv
    submission_dinobloom_v4_cosine_s3_tta.csv
    submission_mlp_anchor_r2_c1cos_pospair.csv
  reproduce_best_submission.sh
  modeling.py
  train.py
  inference.py
  requirements.txt
  scripts/
    build_conservative_hybrid_submission.py
```

## What It Reproduces

- The training setup visualized by the flowchart above, including our best submission of `0.67658` macro-F1 on the WBCBench 2026 leaderboard.

## Results

| Component | Role | Training setup | Eval macro-F1 | Leaderboard |
|---|---|---|---:|---:|
| `dinobloom_v4_mlp_s3 + TTA` | Anchor | `dinobloom_base + mlp`, full fine-tuning, S1->S2->S3 | 0.7184 | 0.67584 |
| `dinobloom_v4_linear_s3 + TTA` | Linear reference | `dinobloom_base + linear`, full fine-tuning, S1->S2->S3 | 0.7281 | 0.66870 |
| `dec_bestlb_cls_c1_wbcc + TTA` | Advisor 1 | `dinobloom_base + mlp`, frozen backbone, head-only fine-tuning | 0.7289 | 0.67540 |
| `dinobloom_v4_cosine_s3 + TTA` | Advisor 2 | `dinobloom_base + cosine`, full fine-tuning, S1->S2->S3 | 0.7225 | 0.66085 |
| `submission_mlp_anchor_r2_c1cos_pospair` | Final hybrid | Anchor plus conservative advisor agreement overrides | 0.7217 | **0.67658** |

Test set submission CSVs for the models above are included in `msfit/submissions/`.

And model checkpoints of the primary predictor (MLP-S3 TTA) can be found on Hugging Face - https://huggingface.co/AntonyG/msfit-dinobloom-v4-mlp-s3

## Install

```bash
pip install -r msfit/requirements.txt
```

Requirements:

- Python 3.10+
- PyTorch 2.x with CUDA
- `timm>=1.0`
- WBCBench 2026 dataset from the challenge organizers
- Internet access or cached Hugging Face weights for DINOBloom (and other models)

## Supported Backbones

The framework supports **any timm-compatible model**, not just DINOBloom:

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

## Reproduce The Submission

This command writes into a fresh run folder under `code_reproduction/` and refuses to overwrite an existing folder:

```bash
RUN_NAME="repro_baseline_nondet_v1"
nohup bash msfit/reproduce_best_submission.sh \
  --data-root /mnt/c/Users/amg/Downloads/wbcc-2026-main/wbc-bench-2026-data \
  --python ./venv/bin/python \
  --run-name "$RUN_NAME" \
  > "code_reproduction/${RUN_NAME}.nohup.log" 2>&1 &
echo "$RUN_NAME"
```

Monitor the run with:

```bash
tail -f "code_reproduction/${RUN_NAME}.nohup.log"
```

## Single-Model Commands

If you want one-command training for the main published branches without running the full hybrid reproduction, use the scripts in [`experiments/`](./experiments):

```bash
bash msfit/experiments/train_mlp.sh --data-root /path/to/wbcbench-2026-data
bash msfit/experiments/train_linear.sh --data-root /path/to/wbcbench-2026-data
bash msfit/experiments/train_cosine.sh --data-root /path/to/wbcbench-2026-data
bash msfit/experiments/train_decoupled_c1.sh \
  --data-root /path/to/wbcbench-2026-data \
  --checkpoint /path/to/dinobloom_v4_mlp_s3/best.pth
```

These wrappers create fresh run directories under `code_reproduction/` and keep the published stage schedules for each branch.

## Method Summary

1. **Backbone in the best setup:** `dinobloom_base`
2. **Anchor model:** `dinobloom_base + mlp head`, full fine-tuning for `11 -> 5 -> 5` epochs
3. **Advisor model 1:** `dinobloom_base + cosine head`, full fine-tuning for `11 -> 5 -> 5` epochs
4. **Advisor model 2 (`c1`):** `dinobloom_base + mlp head`, backbone frozen, head-only fine-tuning for `8` epochs from the final MLP checkpoint
5. **Losses:** focal loss for anchor/cosine, cross-entropy for `c1`
6. **Training details:** label smoothing, rare-class-protected MixUp/CutMix, OneCycleLR, AMP, TTA=8 at inference
7. **Final submission rule:** start from the anchor prediction and override only when both advisors agree on one of four allowed confusion-pair transitions

Outputs:

- `OUT_ROOT/RUN_NAME/submission_best_reproduced.csv`
- `OUT_ROOT/RUN_NAME/repro_metadata.txt`
- `OUT_ROOT/RUN_NAME/*.log`
- `OUT_ROOT/RUN_NAME/*/best.pth`
- `OUT_ROOT/RUN_NAME/preds_*_test/`



## Notes

- `train.py` still supports more than the winning recipe, but the README only documents the published path.
- `inference.py` supports eval/test inference, TTA, logit export, and threshold/bias utilities.
- If you want a slower, more repeatable rerun, pass `--deterministic 1` to the reproduction script.

Like this:

```bash
RUN_NAME="repro_baseline_det_v1"
nohup bash msfit/reproduce_best_submission.sh \
  --data-root /mnt/c/Users/amg/Downloads/wbcc-2026-main/wbc-bench-2026-data \
  --python ./venv/bin/python \
  --run-name "$RUN_NAME" \
  --deterministic 1 \
  > "code_reproduction/${RUN_NAME}.nohup.log" 2>&1 &
```

## Citation

```
@inproceedings{Gitau2026ISBI,
  author    = {Antony Gitau and Martin Paulson and Bjørn-Jostein Singstad and Karl Thomas Hjelmervik and Ola Marius Lysaker and Veralia Gabriela Sanchez},
  title     = {Multi-Stage Fine-Tuning of Pathology Foundation Models with Head-Diverse Ensembling for White Blood Cell Classification},
  booktitle = {2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI)},
  year      = {2026},
  organization = {IEEE}
}
```
