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

```
 RUN_NAME="repro_det_$(date +%Y%m%d_%H%M%S)"
  nohup bash msfit/reproduce_best_submission.sh \
    --data-root /mnt/c/Users/amg/Downloads/wbcc-2026-main/wbc-bench-2026-data \
    --python ./venv/bin/python \
    --run-name "$RUN_NAME" \
    > "code_reproduction/${RUN_NAME}.nohup.log" 2>&1 &
  echo "$RUN_NAME"
```
and you can monitor the run by running this command:

```
tail -f "code_reproduction/${RUN_NAME}.nohup.log"
```


Outputs:

- `OUT_ROOT/RUN_NAME/submission_best_reproduced.csv`
- `OUT_ROOT/RUN_NAME/repro_metadata.txt`
- `OUT_ROOT/RUN_NAME/*.log`
- `OUT_ROOT/RUN_NAME/*/best.pth`
- `OUT_ROOT/RUN_NAME/preds_*_test/`



## Notes

- `train.py` still supports more than the winning recipe, but the README only documents the published path.
- `inference.py` supports eval/test inference, TTA, logit export, and threshold/bias utilities.
- `ensemble_smart.py` is retained as a research utility, not as the primary reproduction path.
- If you want a slower, more repeatable rerun, pass `--deterministic 1` to the reproduction script.

Like this:

```
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