# msfit

`msfit` is the publication-ready code bundle for our ISBI 2026 WBCBench submission:
multi-stage fine-tuning plus a conservative hybrid ensemble.

The public surface is intentionally small. There is one supported shell entrypoint:
[reproduce_best_submission.sh](/home/antonyg/wbcc-2026-main/msfit/reproduce_best_submission.sh).
Everything else is Python code used by that script.

## Layout

```text
msfit/
  README.md
  reproduce_best_submission.sh
  modeling.py
  train.py
  inference.py
  ensemble_smart.py
  requirements.txt
  scripts/
    build_conservative_hybrid_submission.py
```

## What It Reproduces

- Best confirmed leaderboard submission: `0.67658` macro-F1
- Submission name: `submission_mlp_anchor_r2_c1cos_pospair`
- Method: MLP anchor + cosine advisor + decoupled c1 advisor + conservative confusion-pair overrides

## Install

```bash
pip install -r msfit/requirements.txt
```

Requirements:

- Python 3.10+
- PyTorch 2.x with CUDA
- `timm>=1.0`
- WBCBench 2026 dataset from the challenge organizers
- Internet access or cached Hugging Face weights for DINOBloom

## Reproduce The Submission

This command writes into a fresh run folder under `code_reproduction/` and refuses to overwrite an existing folder:

```bash
bash msfit/reproduce_best_submission.sh \
  --data-root /path/to/wbc-bench-2026-data \
  --python /path/to/python \
  --run-name repro_$(date +%Y%m%d_%H%M%S)
```

Recommended stricter verification command:

```bash
bash msfit/reproduce_best_submission.sh \
  --data-root /path/to/wbc-bench-2026-data \
  --python /path/to/python \
  --run-name repro_$(date +%Y%m%d_%H%M%S) \
  --strict-flips 1
```

Outputs:

- `OUT_ROOT/RUN_NAME/submission_best_reproduced.csv`
- `OUT_ROOT/RUN_NAME/repro_metadata.txt`
- `OUT_ROOT/RUN_NAME/*.log`
- `OUT_ROOT/RUN_NAME/*/best.pth`
- `OUT_ROOT/RUN_NAME/preds_*_test/`

## Seed Policy

The winning recipe uses a fixed integer seed, `42`. We keep that as the default because it matches the submitted training recipe.

We do not silently force deterministic CUDA behavior by default, because that would change the original training setup and can reduce throughput. If you want a slower, more repeatable rerun, pass `--deterministic 1` to the reproduction script.

## Notes

- `train.py` still supports more than the winning recipe, but the README only documents the published path.
- `inference.py` supports eval/test inference, TTA, logit export, and threshold/bias utilities.
- `ensemble_smart.py` is retained as a research utility, not as the primary reproduction path.

## Citation

If you use this code, cite the ISBI 2026 challenge paper once the camera-ready citation is available.
