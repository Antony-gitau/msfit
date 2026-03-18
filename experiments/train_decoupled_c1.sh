#!/usr/bin/env bash
# One-command training for the published decoupled c1 advisor.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${APP_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-}"
PYTHON="${PYTHON:-python3}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/code_reproduction}"
RUN_NAME="${RUN_NAME:-dec_c1_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
DETERMINISTIC="${DETERMINISTIC:-0}"
BASE_CKPT="${BASE_CKPT:-}"

usage() {
  cat <<'EOF'
Usage:
  bash msfit/experiments/train_decoupled_c1.sh --data-root <path> --checkpoint <path> [options]

Required:
  --data-root PATH          Path to the WBCBench dataset root
  --checkpoint PATH         Base checkpoint to decouple from (typically mlp_s3/best.pth)

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
    --checkpoint)
      BASE_CKPT="${2:-}"; shift 2 ;;
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

if [[ -z "${DATA_ROOT}" ]]; then
  echo "ERROR: --data-root is required." >&2
  usage
  exit 2
fi

if [[ -z "${BASE_CKPT}" ]]; then
  echo "ERROR: --checkpoint is required." >&2
  usage
  exit 2
fi

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${BASE_CKPT}" >&2
  exit 1
fi

if [[ "${DETERMINISTIC}" != "0" && "${DETERMINISTIC}" != "1" ]]; then
  echo "ERROR: --deterministic must be 0 or 1." >&2
  exit 2
fi

RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "ERROR: run directory already exists: ${RUN_DIR}" >&2
  echo "Pick a new --run-name to avoid overwriting prior work." >&2
  exit 1
fi
mkdir -p "${RUN_DIR}"

C1_TRAIN_ARGS=(
  --data-root "${DATA_ROOT}"
  --checkpoint "${BASE_CKPT}"
  --backbone dinobloom_base
  --backbone-only
  --fresh-optimizer
  --freeze-backbone
  --head mlp
  --mlp-hidden-dim 512
  --mlp-dropout 0.20
  --loss ce
  --no-multi-scale
  --sampler balanced
  --label-smoothing 0.0
  --epochs 8
  --warmup-epochs 1
  --lr 7e-5
  --batch-size 16
  --accumulation-steps 4
  --num-workers 6
  --seed "${SEED}"
  --patience 3
  --grad-clip 0.8
  --mixup-prob 0.0
  --cutmix-prob 0.0
  --ema
  --ema-decay 0.9998
  --output-dir "${RUN_DIR}/dec_bestlb_cls_c1"
)

if [[ "${DETERMINISTIC}" == "1" ]]; then
  C1_TRAIN_ARGS+=(--deterministic)
fi

{
  echo "timestamp: $(date -Iseconds)"
  echo "repo_root: ${REPO_ROOT}"
  echo "app_dir: ${APP_DIR}"
  echo "data_root: ${DATA_ROOT}"
  echo "python: ${PYTHON}"
  echo "seed: ${SEED}"
  echo "deterministic: ${DETERMINISTIC}"
  echo "base_checkpoint: ${BASE_CKPT}"
  if command -v git >/dev/null 2>&1; then
    (cd "${REPO_ROOT}" && echo "git_commit: $(git rev-parse HEAD)")
  fi
} > "${RUN_DIR}/run_metadata.txt"

echo "========================================================"
echo " msfit: train decoupled c1 advisor"
echo "========================================================"
echo "DATA_ROOT      : ${DATA_ROOT}"
echo "PYTHON         : ${PYTHON}"
echo "BASE_CKPT      : ${BASE_CKPT}"
echo "RUN_DIR        : ${RUN_DIR}"
echo "SEED           : ${SEED}"
echo "DETERMINISTIC  : ${DETERMINISTIC}"
echo ""

"${PYTHON}" -u "${APP_DIR}/train.py" "${C1_TRAIN_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/dec_bestlb_cls_c1.log"

echo ""
echo "========================================================"
echo "Done."
echo "Run directory : ${RUN_DIR}"
echo "Best checkpoint: ${RUN_DIR}/dec_bestlb_cls_c1/best.pth"
echo "========================================================"
