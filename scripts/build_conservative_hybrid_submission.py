#!/usr/bin/env python3
"""
Build a conservative hybrid submission from model prediction CSVs.

Default behavior reproduces the LB-best rule:
- Anchor model: MLP S3 (TTA) prediction
- Base override only when:
  1) dec_c1 and cosine_s3 predict the same label, and
  2) (anchor_label -> agreed_label) is one of the allowed base pairs.

Optional extension (for "r2-plus" variants):
- Add extra overrides from 2-of-3 majority across (c1, cos, c2)
- Restrict those extra overrides to specific pairs
- Optionally gate extra overrides by anchor confidence/margin from logits
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional


DEFAULT_ALLOWED_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("BNE", "SNE"),
    ("MO", "VLY"),
    ("MY", "MMY"),
    ("LY", "BL"),
)

ID_COL = "ID"
LABEL_COL_CANDIDATES: Tuple[str, ...] = ("labels", "class", "label", "prediction", "pred")


def parse_pair(raw: str) -> Tuple[str, str]:
    if ":" not in raw:
        raise ValueError(f"Invalid pair '{raw}'. Expected FROM:TO (e.g., BNE:SNE).")
    src, dst = raw.split(":", 1)
    src = src.strip().upper()
    dst = dst.strip().upper()
    if not src or not dst:
        raise ValueError(f"Invalid pair '{raw}'. Empty source/target.")
    return src, dst


def majority_label(labels: Sequence[str]) -> Optional[str]:
    counts = Counter(labels)
    top = counts.most_common()
    if not top:
        return None
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def detect_label_col(fieldnames: Sequence[str]) -> str:
    for candidate in LABEL_COL_CANDIDATES:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        f"Could not find label column in CSV. Tried: {', '.join(LABEL_COL_CANDIDATES)}. "
        f"Available columns: {', '.join(fieldnames)}"
    )


def load_submission_csv(path: Path) -> Tuple[List[str], Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")

    ordered_ids: List[str] = []
    preds: Dict[str, str] = {}

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        if ID_COL not in reader.fieldnames:
            raise ValueError(f"CSV missing '{ID_COL}' column: {path}")
        label_col = detect_label_col(reader.fieldnames)

        for row in reader:
            image_id = row[ID_COL].strip()
            label = row[label_col].strip()
            if not image_id:
                raise ValueError(f"Found empty ID in {path}")
            if image_id in preds:
                raise ValueError(f"Duplicate ID '{image_id}' in {path}")
            ordered_ids.append(image_id)
            preds[image_id] = label

    return ordered_ids, preds


def validate_same_ids(
    anchor_ids: Sequence[str],
    anchor_preds: Dict[str, str],
    other_name: str,
    other_preds: Dict[str, str],
) -> None:
    anchor_set = set(anchor_preds)
    other_set = set(other_preds)
    if anchor_set == other_set:
        return

    missing = sorted(anchor_set - other_set)
    extra = sorted(other_set - anchor_set)
    msg = [f"ID mismatch between anchor and {other_name}."]
    if missing:
        msg.append(f"Missing in {other_name}: {len(missing)} (sample: {missing[:5]})")
    if extra:
        msg.append(f"Extra in {other_name}: {len(extra)} (sample: {extra[:5]})")
    raise ValueError(" ".join(msg))


def write_submission(path: Path, ids: Sequence[str], preds: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([ID_COL, "labels"])
        for image_id in ids:
            writer.writerow([image_id, preds[image_id]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True, help="Anchor submission CSV (e.g., mlp_s3 TTA).")
    parser.add_argument("--c1", required=True, help="dec_c1 submission CSV (TTA).")
    parser.add_argument("--cos", required=True, help="cosine_s3 submission CSV (TTA).")
    parser.add_argument(
        "--c2",
        default=None,
        help="Optional dec_c2 submission CSV (TTA). Required only for --extra-any2-pair rules.",
    )
    parser.add_argument("--output", required=True, help="Output submission CSV path.")
    parser.add_argument(
        "--allowed-pair",
        action="append",
        default=[],
        help=(
            "Allowed override pair in FROM:TO format. "
            "Can be provided multiple times. If omitted, defaults are used."
        ),
    )
    parser.add_argument(
        "--expected-flips",
        type=int,
        default=None,
        help="Optional guardrail: fail if produced flips count does not match this value.",
    )
    parser.add_argument(
        "--extra-any2-pair",
        action="append",
        default=[],
        help=(
            "Optional extra override pair (FROM:TO) applied by 2-of-3 majority over (c1, cos, c2). "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--anchor-logits",
        default=None,
        help="Optional .npy logits for anchor predictions, aligned with anchor CSV order.",
    )
    parser.add_argument(
        "--extra-margin-max",
        type=float,
        default=None,
        help="Optional max anchor top1-top2 softmax margin for extra-any2 overrides.",
    )
    parser.add_argument(
        "--extra-pmax-max",
        type=float,
        default=None,
        help="Optional max anchor top1 softmax probability for extra-any2 overrides.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    anchor_path = Path(args.anchor)
    c1_path = Path(args.c1)
    cos_path = Path(args.cos)
    c2_path = Path(args.c2) if args.c2 else None
    output_path = Path(args.output)

    base_pairs = tuple(parse_pair(p) for p in args.allowed_pair) if args.allowed_pair else DEFAULT_ALLOWED_PAIRS
    base_set = set(base_pairs)
    extra_pairs = tuple(parse_pair(p) for p in args.extra_any2_pair)
    extra_set = set(extra_pairs)

    anchor_ids, anchor_preds = load_submission_csv(anchor_path)
    _, c1_preds = load_submission_csv(c1_path)
    _, cos_preds = load_submission_csv(cos_path)
    c2_preds: Optional[Dict[str, str]] = None
    if extra_set:
        if c2_path is None:
            raise ValueError("--c2 is required when --extra-any2-pair is used.")
        _, c2_preds = load_submission_csv(c2_path)
        validate_same_ids(anchor_ids, anchor_preds, "c2", c2_preds)

    validate_same_ids(anchor_ids, anchor_preds, "c1", c1_preds)
    validate_same_ids(anchor_ids, anchor_preds, "cos", cos_preds)

    pmax_by_id: Dict[str, float] = {}
    margin_by_id: Dict[str, float] = {}
    if args.anchor_logits is not None:
        logits_path = Path(args.anchor_logits)
        if not logits_path.exists():
            raise FileNotFoundError(f"Missing logits: {logits_path}")
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "numpy is required when --anchor-logits is used. "
                "Run with a Python environment that has numpy installed."
            ) from exc

        logits = np.load(logits_path)
        if logits.ndim != 2:
            raise ValueError(f"Expected 2D logits array, got shape {logits.shape} from {logits_path}")
        if logits.shape[0] != len(anchor_ids):
            raise ValueError(
                f"Logits rows ({logits.shape[0]}) do not match anchor rows ({len(anchor_ids)}). "
                "Ensure logits and anchor CSV come from the same run/order."
            )
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        pmax = probs.max(axis=1)
        part = np.partition(probs, -2, axis=1)
        margin = part[:, -1] - part[:, -2]
        for i, image_id in enumerate(anchor_ids):
            pmax_by_id[image_id] = float(pmax[i])
            margin_by_id[image_id] = float(margin[i])

    if (args.extra_margin_max is not None or args.extra_pmax_max is not None) and not pmax_by_id:
        raise ValueError("Confidence gating requested but --anchor-logits was not provided.")

    out_preds = dict(anchor_preds)
    base_flips_by_pair: Counter[Tuple[str, str]] = Counter()
    extra_flips_by_pair: Counter[Tuple[str, str]] = Counter()

    for image_id in anchor_ids:
        anchor_label = anchor_preds[image_id]
        c1_label = c1_preds[image_id]
        cos_label = cos_preds[image_id]

        # Base rule: c1 and cos agree on an allowed pair.
        if c1_label != cos_label:
            pass
        else:
            candidate = (anchor_label, c1_label)
            if candidate in base_set and anchor_label != c1_label:
                out_preds[image_id] = c1_label
                base_flips_by_pair[candidate] += 1
                continue

        # Optional extra rule: 2-of-3 majority (c1, cos, c2) on selected pairs.
        if extra_set and c2_preds is not None:
            maj = majority_label((c1_label, cos_label, c2_preds[image_id]))
            if maj is None or maj == anchor_label:
                continue
            candidate2 = (anchor_label, maj)
            if candidate2 not in extra_set:
                continue

            if args.extra_margin_max is not None and margin_by_id[image_id] > args.extra_margin_max:
                continue
            if args.extra_pmax_max is not None and pmax_by_id[image_id] > args.extra_pmax_max:
                continue

            out_preds[image_id] = maj
            extra_flips_by_pair[candidate2] += 1

    total_flips = sum(base_flips_by_pair.values()) + sum(extra_flips_by_pair.values())
    if args.expected_flips is not None and total_flips != args.expected_flips:
        raise ValueError(
            f"Produced {total_flips} flips, expected {args.expected_flips}. "
            "Inputs or rule set may not match the intended run."
        )

    write_submission(output_path, anchor_ids, out_preds)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(anchor_ids)}")
    print(f"Flips: {total_flips}")
    print("Base pairs:", ", ".join(f"{a}->{b}" for a, b in base_pairs))
    if base_flips_by_pair:
        print("Base flips:")
        for (src, dst), count in base_flips_by_pair.most_common():
            print(f"  {src}->{dst}: {count}")
    if extra_set:
        print("Extra any2 pairs:", ", ".join(f"{a}->{b}" for a, b in extra_pairs))
        if args.extra_margin_max is not None:
            print(f"Extra margin gate: <= {args.extra_margin_max}")
        if args.extra_pmax_max is not None:
            print(f"Extra pmax gate: <= {args.extra_pmax_max}")
        if extra_flips_by_pair:
            print("Extra flips:")
            for (src, dst), count in extra_flips_by_pair.most_common():
                print(f"  {src}->{dst}: {count}")
        else:
            print("Extra flips: none")
    if not base_flips_by_pair and not extra_flips_by_pair:
        print("No overrides applied.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
