# Experiments

One-command wrappers for the main published training paths.

## Commands

Train the published MLP family:

```bash
bash msfit/experiments/train_mlp.sh \
  --data-root /path/to/wbcbench-2026-data
```

Train the published linear family:

```bash
bash msfit/experiments/train_linear.sh \
  --data-root /path/to/wbcbench-2026-data
```

Train the published cosine family:

```bash
bash msfit/experiments/train_cosine.sh \
  --data-root /path/to/wbcbench-2026-data
```

Train the published decoupled `c1` advisor from an existing base checkpoint:

```bash
bash msfit/experiments/train_decoupled_c1.sh \
  --data-root /path/to/wbcbench-2026-data \
  --checkpoint /path/to/dinobloom_v4_mlp_s3/best.pth
```

All scripts create a fresh run directory under `code_reproduction/` by default and refuse to overwrite an existing run.
