#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY=mlp bash "${SCRIPT_DIR}/train_head_family.sh" "$@"
