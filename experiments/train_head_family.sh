#!/usr/bin/env bash
# Internal helper for one-command training of a single head family.
# Wrapper scripts in this folder set FAMILY={mlp,linear,cosine} before calling it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${APP_DIR}/.." && pwd)"

FAMILY="${FAMILY:-}"
DATA_ROOT="${DATA_ROOT:-}"
PYTHON="${PYTHON:-python3}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/code_reproduction}"
RUN_NAME="${RUN_NAME:-${FAMILY}_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
DETERMINISTIC="${DETERMINISTIC:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash msfit/experiments/<family>.sh --data-root <path> [options]

Required:
  --data-root PATH          Path to the WBCBench dataset root

Optional:
  --python PATH             Python executable to use (default: python3)
  --out-root PATH           Parent directory for fresh run folders
  --run-name NAME           Run folder name under out-root
  --seed INT                Training seed to record and use (default: 42)
  --deterministic 0|1       Slower, more repeatable CUDA/cuDNN behavior (default: 0)
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="${2:-}"; shift 2 ;;
    --python)
      PYTHON="${2:-}"; shift 2 ;;
    --out-root)
      OUT_ROOT="${2:-}"; shift 2 ;;
    --run-name)
      RUN_NAME="${2:-}"; shift 2 ;;
    --seed)
      SEED="${2:-}"; shift 2 ;;
    --deterministic)
      DETERMINISTIC="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "${FAMILY}" ]]; then
  echo "ERROR: FAMILY env var is required." >&2
  exit 2
fi

if [[ -z "${DATA_ROOT}" ]]; then
  echo "ERROR: --data-root is required." >&2
  usage
  exit 2
fi

if [[ "${DETERMINISTIC}" != "0" && "${DETERMINISTIC}" != "1" ]]; then
  echo "ERROR: --deterministic must be 0 or 1." >&2
  exit 2
fi

case "${FAMILY}" in
  mlp)
    HEAD_ARGS=(--head mlp --mlp-hidden-dim 512 --mlp-dropout 0.20)
    PREFIX_BASE="dinobloom_v4_mlp"
    S1_LR="2.5e-5"; S1_LR_BACKBONE="5e-6"; S1_WARMUP="2"; S1_PATIENCE="15"
    S2_LR="1e-5";   S2_LR_BACKBONE="2e-6"; S2_WARMUP="0"; S2_PATIENCE="5"
    S3_LR="5e-6";   S3_LR_BACKBONE="1e-6"; S3_WARMUP="0"; S3_PATIENCE="5"
    ;;
  linear)
    HEAD_ARGS=(--head linear)
    PREFIX_BASE="dinobloom_v4_linear"
    S1_LR="3e-5";   S1_LR_BACKBONE="5e-6"; S1_WARMUP="2"; S1_PATIENCE="15"
    S2_LR="1e-5";   S2_LR_BACKBONE="2e-6"; S2_WARMUP="1"; S2_PATIENCE="5"
    S3_LR="5e-6";   S3_LR_BACKBONE="1e-6"; S3_WARMUP="1"; S3_PATIENCE="5"
    ;;
  cosine)
    HEAD_ARGS=(--head cosine)
    PREFIX_BASE="dinobloom_v4_cosine"
    S1_LR="3e-5";   S1_LR_BACKBONE="5e-6"; S1_WARMUP="2"; S1_PATIENCE="15"
    S2_LR="1.5e-5"; S2_LR_BACKBONE="2.5e-6"; S2_WARMUP="1"; S2_PATIENCE="15"
    S3_LR="7.5e-6"; S3_LR_BACKBONE="1.25e-6"; S3_WARMUP="1"; S3_PATIENCE="15"
    ;;
  *)
    echo "ERROR: unsupported FAMILY='${FAMILY}'" >&2
    exit 2
    ;;
esac

RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "ERROR: run directory already exists: ${RUN_DIR}" >&2
  echo "Pick a new --run-name to avoid overwriting prior work." >&2
  exit 1
fi
mkdir -p "${RUN_DIR}"

PREFIX="${RUN_DIR}/${PREFIX_BASE}"

COMMON_TRAIN_ARGS=(
  --data-root "${DATA_ROOT}"
  --backbone dinobloom_base
  --no-freeze-backbone
  --loss focal
  --focal-gamma 2.0
  --no-multi-scale
  --no-confusion-loss
  --no-drw
  --sampler none
  --label-smoothing 0.1
  --img-size 384
  --batch-size 16
  --accumulation-steps 4
  --num-workers 6
  --seed "${SEED}"
  --mixup-prob 0.1
  --mixup-alpha 0.8
  --cutmix-prob 0.1
  --cutmix-alpha 1.0
)

if [[ "${DETERMINISTIC}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--deterministic)
fi

run_train() {
  local log_path="$1"
  shift
  "${PYTHON}" -u "${APP_DIR}/train.py" "$@" 2>&1 | tee "${log_path}"
}

{
  echo "timestamp: $(date -Iseconds)"
  echo "repo_root: ${REPO_ROOT}"
  echo "app_dir: ${APP_DIR}"
  echo "data_root: ${DATA_ROOT}"
  echo "python: ${PYTHON}"
  echo "family: ${FAMILY}"
  echo "seed: ${SEED}"
  echo "deterministic: ${DETERMINISTIC}"
  if command -v git >/dev/null 2>&1; then
    (cd "${REPO_ROOT}" && echo "git_commit: $(git rev-parse HEAD)")
  fi
} > "${RUN_DIR}/run_metadata.txt"

echo "========================================================"
echo " msfit: train ${FAMILY} head family (S1 -> S2 -> S3)"
echo "========================================================"
echo "DATA_ROOT      : ${DATA_ROOT}"
echo "PYTHON         : ${PYTHON}"
echo "RUN_DIR        : ${RUN_DIR}"
echo "SEED           : ${SEED}"
echo "DETERMINISTIC  : ${DETERMINISTIC}"
echo ""

echo "===== [1/3] Train ${FAMILY} S1 ====="
run_train "${RUN_DIR}/${PREFIX_BASE}_s1.log" \
  "${COMMON_TRAIN_ARGS[@]}" \
  "${HEAD_ARGS[@]}" \
  --epochs 11 \
  --warmup-epochs "${S1_WARMUP}" \
  --lr "${S1_LR}" \
  --lr-backbone "${S1_LR_BACKBONE}" \
  --patience "${S1_PATIENCE}" \
  --output-dir "${PREFIX}_s1"
echo ""

echo "===== [2/3] Train ${FAMILY} S2 ====="
run_train "${RUN_DIR}/${PREFIX_BASE}_s2.log" \
  "${COMMON_TRAIN_ARGS[@]}" \
  "${HEAD_ARGS[@]}" \
  --checkpoint "${PREFIX}_s1/best.pth" \
  --fresh-optimizer \
  --epochs 5 \
  --warmup-epochs "${S2_WARMUP}" \
  --lr "${S2_LR}" \
  --lr-backbone "${S2_LR_BACKBONE}" \
  --patience "${S2_PATIENCE}" \
  --output-dir "${PREFIX}_s2"
echo ""

echo "===== [3/3] Train ${FAMILY} S3 ====="
run_train "${RUN_DIR}/${PREFIX_BASE}_s3.log" \
  "${COMMON_TRAIN_ARGS[@]}" \
  "${HEAD_ARGS[@]}" \
  --checkpoint "${PREFIX}_s2/best.pth" \
  --fresh-optimizer \
  --epochs 5 \
  --warmup-epochs "${S3_WARMUP}" \
  --lr "${S3_LR}" \
  --lr-backbone "${S3_LR_BACKBONE}" \
  --patience "${S3_PATIENCE}" \
  --output-dir "${PREFIX}_s3"

echo ""
echo "========================================================"
echo "Done."
echo "Run directory : ${RUN_DIR}"
echo "Best checkpoint: ${PREFIX}_s3/best.pth"
echo "========================================================"
