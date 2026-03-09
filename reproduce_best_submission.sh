#!/usr/bin/env bash
# One-command reproduction for the published msfit hybrid submission.
# Safe by default: every run writes to OUT_ROOT/RUN_NAME and refuses to reuse
# an existing folder.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_DIR="${SCRIPT_DIR}"

DATA_ROOT="${DATA_ROOT:-}"
PYTHON="${PYTHON:-python3}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/code_reproduction}"
RUN_NAME="${RUN_NAME:-repro_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
STRICT_FLIPS="${STRICT_FLIPS:-0}"
DETERMINISTIC="${DETERMINISTIC:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash msfit/reproduce_best_submission.sh --data-root <path> [options]

Required:
  --data-root PATH          Path to the WBCBench dataset root

Optional:
  --python PATH             Python executable to use (default: python3)
  --out-root PATH           Parent directory for fresh run folders
  --run-name NAME           Run folder name under out-root
  --seed INT                Training seed to record and use (default: 42)
  --deterministic 0|1       Slower, more repeatable CUDA/cuDNN behavior (default: 0)
  --strict-flips 0|1        Fail if the final hybrid does not flip exactly 20 IDs (default: 0)
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
    --strict-flips)
      STRICT_FLIPS="${2:-}"; shift 2 ;;
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

if [[ "${DETERMINISTIC}" != "0" && "${DETERMINISTIC}" != "1" ]]; then
  echo "ERROR: --deterministic must be 0 or 1." >&2
  exit 2
fi

if [[ "${STRICT_FLIPS}" != "0" && "${STRICT_FLIPS}" != "1" ]]; then
  echo "ERROR: --strict-flips must be 0 or 1." >&2
  exit 2
fi

RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "ERROR: run directory already exists: ${RUN_DIR}" >&2
  echo "Pick a new --run-name to avoid overwriting prior work." >&2
  exit 1
fi
mkdir -p "${RUN_DIR}"

MLP_PREFIX="${RUN_DIR}/dinobloom_v4_mlp"
COS_PREFIX="${RUN_DIR}/dinobloom_v4_cosine"
C1_DIR="${RUN_DIR}/dec_bestlb_cls_c1"

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

C1_TRAIN_ARGS=(
  --data-root "${DATA_ROOT}"
  --backbone dinobloom_base
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
)

if [[ "${DETERMINISTIC}" == "1" ]]; then
  C1_TRAIN_ARGS+=(--deterministic)
fi

run_train() {
  local log_path="$1"
  shift
  "${PYTHON}" -u "${APP_DIR}/train.py" "$@" 2>&1 | tee "${log_path}"
}

run_test_inference() {
  local ckpt_path="$1"
  local out_dir="$2"
  local log_path="$3"
  "${PYTHON}" -u "${APP_DIR}/inference.py" \
    --checkpoint "${ckpt_path}" \
    --data-root "${DATA_ROOT}" \
    --backbone dinobloom_base \
    --no-multi-scale \
    --img-size 384 \
    --batch-size 64 \
    --predict-split test \
    --save-logits \
    --tta --tta-views 8 \
    --output-dir "${out_dir}" \
    2>&1 | tee "${log_path}"
}

echo "========================================================"
echo " msfit: reproduce leaderboard submission 0.67658"
echo "========================================================"
echo "DATA_ROOT      : ${DATA_ROOT}"
echo "PYTHON         : ${PYTHON}"
echo "RUN_DIR        : ${RUN_DIR}"
echo "SEED           : ${SEED}"
echo "DETERMINISTIC  : ${DETERMINISTIC}"
echo "STRICT_FLIPS   : ${STRICT_FLIPS}"
echo ""

{
  echo "timestamp: $(date -Iseconds)"
  echo "repo_root: ${REPO_ROOT}"
  echo "app_dir: ${APP_DIR}"
  echo "data_root: ${DATA_ROOT}"
  echo "python: ${PYTHON}"
  echo "seed: ${SEED}"
  echo "deterministic: ${DETERMINISTIC}"
  echo "strict_flips: ${STRICT_FLIPS}"
  if command -v git >/dev/null 2>&1; then
    (cd "${REPO_ROOT}" && echo "git_commit: $(git rev-parse HEAD)")
  fi
} > "${RUN_DIR}/repro_metadata.txt"

echo "===== [1/5] Train MLP anchor (S1 -> S2 -> S3) ====="
run_train "${RUN_DIR}/dinobloom_v4_mlp_s1.log" \
  "${COMMON_TRAIN_ARGS[@]}" \
  --head mlp \
  --mlp-hidden-dim 512 \
  --mlp-dropout 0.20 \
  --epochs 11 \
  --warmup-epochs 2 \
  --lr 2.5e-5 \
  --lr-backbone 5e-6 \
  --patience 15 \
  --output-dir "${MLP_PREFIX}_s1"

run_train "${RUN_DIR}/dinobloom_v4_mlp_s2.log" \
  "${COMMON_TRAIN_ARGS[@]}" \
  --head mlp \
  --mlp-hidden-dim 512 \
  --mlp-dropout 0.20 \
  --checkpoint "${MLP_PREFIX}_s1/best.pth" \
  --fresh-optimizer \
  --epochs 5 \
  --warmup-epochs 0 \
  --lr 1e-5 \
  --lr-backbone 2e-6 \
  --patience 5 \
  --output-dir "${MLP_PREFIX}_s2"

run_train "${RUN_DIR}/dinobloom_v4_mlp_s3.log" \
  "${COMMON_TRAIN_ARGS[@]}" \
  --head mlp \
  --mlp-hidden-dim 512 \
  --mlp-dropout 0.20 \
  --checkpoint "${MLP_PREFIX}_s2/best.pth" \
  --fresh-optimizer \
  --epochs 5 \
  --warmup-epochs 0 \
  --lr 5e-6 \
  --lr-backbone 1e-6 \
  --patience 5 \
  --output-dir "${MLP_PREFIX}_s3"
echo ""

echo "===== [2/5] Train cosine advisor (S1 -> S2 -> S3) ====="
run_train "${RUN_DIR}/dinobloom_v4_cosine_s1.log" \
  "${COMMON_TRAIN_ARGS[@]}" \
  --head cosine \
  --epochs 11 \
  --warmup-epochs 2 \
  --lr 3e-5 \
  --lr-backbone 5e-6 \
  --patience 15 \
  --output-dir "${COS_PREFIX}_s1"

run_train "${RUN_DIR}/dinobloom_v4_cosine_s2.log" \
  "${COMMON_TRAIN_ARGS[@]}" \
  --head cosine \
  --checkpoint "${COS_PREFIX}_s1/best.pth" \
  --fresh-optimizer \
  --epochs 5 \
  --warmup-epochs 1 \
  --lr 1.5e-5 \
  --lr-backbone 2.5e-6 \
  --patience 15 \
  --output-dir "${COS_PREFIX}_s2"

run_train "${RUN_DIR}/dinobloom_v4_cosine_s3.log" \
  "${COMMON_TRAIN_ARGS[@]}" \
  --head cosine \
  --checkpoint "${COS_PREFIX}_s2/best.pth" \
  --fresh-optimizer \
  --epochs 5 \
  --warmup-epochs 1 \
  --lr 7.5e-6 \
  --lr-backbone 1.25e-6 \
  --patience 15 \
  --output-dir "${COS_PREFIX}_s3"
echo ""

echo "===== [3/5] Train decoupled classifier c1 ====="
run_train "${RUN_DIR}/dec_bestlb_cls_c1.log" \
  "${C1_TRAIN_ARGS[@]}" \
  --checkpoint "${MLP_PREFIX}_s3/best.pth" \
  --backbone-only \
  --fresh-optimizer \
  --output-dir "${C1_DIR}"
echo ""

echo "===== [4/5] Run test inference (TTA=8) ====="
run_test_inference \
  "${MLP_PREFIX}_s3/best.pth" \
  "${RUN_DIR}/preds_dinobloom_v4_mlp_s3_test" \
  "${RUN_DIR}/infer_dinobloom_v4_mlp_s3_test.log"
run_test_inference \
  "${COS_PREFIX}_s3/best.pth" \
  "${RUN_DIR}/preds_dinobloom_v4_cosine_s3_test" \
  "${RUN_DIR}/infer_dinobloom_v4_cosine_s3_test.log"
run_test_inference \
  "${C1_DIR}/best.pth" \
  "${RUN_DIR}/preds_dec_bestlb_cls_c1_test" \
  "${RUN_DIR}/infer_dec_bestlb_cls_c1_test.log"
echo ""

echo "===== [5/5] Build conservative hybrid submission ====="
BUILD_ARGS=(
  --anchor "${RUN_DIR}/preds_dinobloom_v4_mlp_s3_test/submission_best_TTA.csv"
  --c1 "${RUN_DIR}/preds_dec_bestlb_cls_c1_test/submission_best_TTA.csv"
  --cos "${RUN_DIR}/preds_dinobloom_v4_cosine_s3_test/submission_best_TTA.csv"
  --output "${RUN_DIR}/submission_best_reproduced.csv"
)
if [[ "${STRICT_FLIPS}" == "1" ]]; then
  BUILD_ARGS+=(--expected-flips 20)
fi

"${PYTHON}" -u "${APP_DIR}/scripts/build_conservative_hybrid_submission.py" \
  "${BUILD_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/build_hybrid.log"

echo ""
echo "========================================================"
echo "Done."
echo "Run directory: ${RUN_DIR}"
echo "Submission   : ${RUN_DIR}/submission_best_reproduced.csv"
echo "========================================================"
