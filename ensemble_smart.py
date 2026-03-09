"""
Smart Ensemble for WBCBench 2026
=================================
Three stacked strategies to exploit per-model, per-class differences:

  1. Per-class F1 weighting  — each model gets a different weight per class
                               based on its eval F1 for that class.
  2. Scipy global weights    — one scalar per model, optimised on composite.
  3. Per-class bias tuning   — per-class logit offsets to shift decision boundaries.

Usage (9 models, TTA logits):
    python ensemble_smart.py \
        --test-logits  preds_cosine_s1_test/logits_best_TTA.npy \
               i also have        preds_cosine_s2_test/logits_best_TTA.npy \
                       preds_cosine_s3_test/logits_best_TTA.npy \
                       preds_linear_s1_test/logits_best_TTA.npy \
                       preds_linear_s2_test/logits_best_TTA.npy \
                       preds_linear_s3_test/logits_best_TTA.npy \
                       preds_mlp_s1_test/logits_best_TTA.npy \
                       preds_mlp_s2_test/logits_best_TTA.npy \
                       preds_mlp_s3_test/logits_best_TTA.npy \
        --eval-logits  preds_cosine_s1_eval/logits_best_TTA.npy \
                       preds_cosine_s2_eval/logits_best_TTA.npy \
                       preds_cosine_s3_eval/logits_best_TTA.npy \
                       preds_linear_s1_eval/logits_best_TTA.npy \
                       preds_linear_s2_eval/logits_best_TTA.npy \
                       preds_linear_s3_eval/logits_best_TTA.npy \
                       preds_mlp_s1_eval/logits_best_TTA.npy \
                       preds_mlp_s2_eval/logits_best_TTA.npy \
                       preds_mlp_s3_eval/logits_best_TTA.npy \
        --eval-csv  /mnt/c/Users/amg/Downloads/wbcc-2026-main/wbc-bench-2026-data/phase2_eval.csv \
        --test-csv  /mnt/c/Users/amg/Downloads/wbcc-2026-main/wbc-bench-2026-data/phase2_test.csv \
        --output-dir ./ensemble_smart_out
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import classification_report, f1_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
NUM_CLASSES = len(CLASS_NAMES)
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {i: c for c, i in LABEL_TO_IDX.items()}
TAIL_CLASSES = ['BNE', 'PLY', 'VLY', 'MMY', 'PMY', 'MY', 'PC']


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax. Input: (N, C), output: (N, C)."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def composite(labels, preds) -> float:
    """0.5 * macro-F1 + 0.5 * mean-F1 over tail classes."""
    report = classification_report(
        labels, preds, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    macro = f1_score(labels, preds, average='macro', zero_division=0)
    tail = np.mean([report[c]['f1-score'] for c in TAIL_CLASSES if c in report])
    return 0.5 * macro + 0.5 * tail


def per_class_f1_matrix(probs_list, labels) -> np.ndarray:
    """Return (n_models, n_classes) F1 matrix from a list of prob arrays."""
    n_models = len(probs_list)
    mat = np.zeros((n_models, NUM_CLASSES))
    for i, probs in enumerate(probs_list):
        preds = probs.argmax(axis=1)
        report = classification_report(
            labels, preds, target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        )
        for j, cls in enumerate(CLASS_NAMES):
            mat[i, j] = report.get(cls, {}).get('f1-score', 0.0)
    return mat


# ---------------------------------------------------------------------------
# Ensemble strategies
# ---------------------------------------------------------------------------

def equal_weight_ensemble(probs_list) -> np.ndarray:
    """Simple average of softmax probabilities."""
    return np.mean(np.stack(probs_list, axis=0), axis=0)


def global_weight_ensemble(probs_list, weights) -> np.ndarray:
    """Weighted average where weights are global scalars (one per model)."""
    w = np.array(weights)
    w = w / w.sum()
    stack = np.stack(probs_list, axis=0)          # (M, N, C)
    return (stack * w[:, np.newaxis, np.newaxis]).sum(axis=0)  # (N, C)


def class_weight_ensemble(probs_list, f1_matrix, weight_power=1.0) -> np.ndarray:
    """
    Per-class weighted ensemble with tunable sharpness.

    For each class c:
        score[sample, c] = sum_m( w[m, c] * prob_m[sample, c] )

    where w[m, c] = F1[m, c]^power / sum_m(F1[m, c]^power)

    weight_power controls how hard the routing is:
        power=1  → proportional to F1 (soft, default)
        power=2  → squares the F1 gap — stronger models dominate more
        power=4  → even sharper — only the top 1-2 models matter per class
        power=inf → hard routing: only the single best model per class

    Note: pure hard routing (power→inf) can hurt if models have different
    probability scales. Moderate sharpening (power=2-4) is usually safer.
    """
    # Hard routing: winner takes all per class
    if weight_power == float('inf'):
        stack = np.stack(probs_list, axis=0)      # (M, N, C)
        best_model_per_class = f1_matrix.argmax(axis=0)  # (C,)
        # For each class c, take probabilities from the single best model
        result = np.zeros_like(stack[0])          # (N, C)
        for c, m in enumerate(best_model_per_class):
            result[:, c] = stack[m, :, c]
        return result

    # Soft routing: raise F1 to power, then normalise per class
    powered = np.power(np.maximum(f1_matrix, 1e-8), weight_power)  # (M, C)
    col_sums = powered.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    weight_matrix = powered / col_sums                              # (M, C)

    stack = np.stack(probs_list, axis=0)                            # (M, N, C)
    weighted = stack * weight_matrix[:, np.newaxis, :]              # (M, N, C)
    return weighted.sum(axis=0)                                     # (N, C)


def optimise_global_weights(probs_list, labels, n_restarts=5) -> np.ndarray:
    """
    Find per-model global weights that maximise composite score on eval.
    Uses Nelder-Mead with multiple random restarts.
    Weights are parameterised in log-space so they stay positive and sum to 1.
    """
    n_models = len(probs_list)

    def neg_composite(log_w):
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
        probs = global_weight_ensemble(probs_list, w)
        return -composite(labels, probs.argmax(axis=1))

    best_score = -np.inf
    best_w = np.ones(n_models) / n_models

    for trial in range(n_restarts):
        x0 = np.random.randn(n_models) * 0.5 if trial > 0 else np.zeros(n_models)
        result = minimize(neg_composite, x0, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-5, 'fatol': 1e-5})
        w = np.exp(result.x - result.x.max())
        w /= w.sum()
        score = composite(labels, global_weight_ensemble(probs_list, w).argmax(axis=1))
        if score > best_score:
            best_score = score
            best_w = w

    return best_w


def optimise_class_biases(probs, labels) -> np.ndarray:
    """
    Find per-class additive biases applied to ensemble probabilities before argmax.
    Maximises composite on eval. Effectively shifts decision boundaries per class.

    pred = argmax(probs[sample] + bias)
    """
    def neg_composite(bias):
        shifted = probs + bias[np.newaxis, :]
        return -composite(labels, shifted.argmax(axis=1))

    result = minimize(neg_composite, x0=np.zeros(NUM_CLASSES), method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-5, 'fatol': 1e-5})
    return result.x


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_per_class_f1(f1_matrix, model_names):
    """Pretty-print per-model, per-class F1 table."""
    header = f"{'Class':<6}" + "".join(f"{n:>12}" for n in model_names)
    print(header)
    print("-" * len(header))
    for j, cls in enumerate(CLASS_NAMES):
        vals = "".join(f"{f1_matrix[i, j]:>12.3f}" for i in range(len(model_names)))
        best_i = int(f1_matrix[:, j].argmax())
        marker = " *" if cls in TAIL_CLASSES else ""
        print(f"{cls:<6}{vals}   best={model_names[best_i]}{marker}")


def print_scores(label, labels, probs):
    preds = probs.argmax(axis=1)
    score = composite(labels, preds)
    macro = f1_score(labels, preds, average='macro', zero_division=0)
    report = classification_report(labels, preds, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    tail = np.mean([report[c]['f1-score'] for c in TAIL_CLASSES])
    print(f"  {label:<35} composite={score:.4f}  macro={macro:.4f}  tail={tail:.4f}")
    return score


def save_submission(probs, test_ids, output_path):
    preds = probs.argmax(axis=1)
    labels = [IDX_TO_LABEL[i] for i in preds]
    df = pd.DataFrame({'ID': test_ids, 'labels': labels})
    df.to_csv(output_path, index=False)
    dist = df['labels'].value_counts()
    print(f"\n  Saved: {output_path}")
    for cls in CLASS_NAMES:
        print(f"    {cls}: {dist.get(cls, 0)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--test-logits', nargs='+', required=True)
    p.add_argument('--eval-logits', nargs='+', required=True)
    p.add_argument('--eval-csv',    required=True)
    p.add_argument('--test-csv',    required=True)
    p.add_argument('--model-names', nargs='*', default=None,
                   help='Labels for each model (same order as --test-logits)')
    p.add_argument('--output-dir',  default='./ensemble_smart_out')
    p.add_argument('--weight-power', type=float, default=None,
                   help='Sharpness for per-class F1 weights: 1=soft, 2=sharper, inf=hard routing. '
                        'If not set, all powers [1, 2, 4, inf] are tried and best is used.')
    p.add_argument('--no-bias-tuning', action='store_true',
                   help='Skip per-class bias tuning (faster)')
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_models = len(args.test_logits)
    assert len(args.eval_logits) == n_models, \
        "Number of eval logits must match test logits"

    model_names = args.model_names or [
        Path(p).parent.name.replace('preds_', '').replace('_test', '').replace('_eval', '')
        for p in args.test_logits
    ]
    # Shorten names if auto-generated
    short_names = [n[:10] for n in model_names]

    print(f"\n{'=' * 70}")
    print(f"Smart Ensemble — {n_models} models")
    print(f"{'=' * 70}")

    # ---- Load logits ----
    print("\nLoading eval logits...")
    eval_probs = [softmax(np.load(p)) for p in args.eval_logits]

    print("Loading test logits...")
    test_probs = [softmax(np.load(p)) for p in args.test_logits]

    # ---- Eval labels ----
    df_eval = pd.read_csv(args.eval_csv)
    eval_labels = df_eval['labels'].map(LABEL_TO_IDX).values
    assert not np.isnan(eval_labels).any(), "Some eval labels couldn't be mapped"
    eval_labels = eval_labels.astype(int)

    # ---- Test IDs ----
    df_test = pd.read_csv(args.test_csv)
    test_ids = df_test['ID'].values

    # ---- Per-class F1 matrix ----
    print("\nComputing per-model, per-class F1 on eval...")
    f1_mat = per_class_f1_matrix(eval_probs, eval_labels)

    print(f"\nPer-model per-class F1 (tail classes marked *):\n")
    print_per_class_f1(f1_mat, short_names)

    # ---- Strategy comparison on eval ----
    print(f"\n{'=' * 70}")
    print("Strategy comparison on eval set:")
    print(f"{'=' * 70}")

    # 1. Equal weights baseline
    eq_probs_eval = equal_weight_ensemble(eval_probs)
    s1 = print_scores("1. Equal weights (baseline)", eval_labels, eq_probs_eval)

    # 2. Per-class F1 weights — try multiple sharpness levels
    powers_to_try = [args.weight_power] if args.weight_power is not None \
                    else [1.0, 2.0, 4.0, float('inf')]
    best_cw_score = -1.0
    best_cw_probs = None
    best_power = None
    for pw in powers_to_try:
        label = f"hard routing" if pw == float('inf') else f"power={pw}"
        cw = class_weight_ensemble(eval_probs, f1_mat, weight_power=pw)
        sc = print_scores(f"2. Per-class F1 weights ({label})", eval_labels, cw)
        if sc > best_cw_score:
            best_cw_score = sc
            best_cw_probs = cw
            best_power = pw
    cw_probs_eval = best_cw_probs
    s2 = best_cw_score
    print(f"     -> Best sharpness: {'hard routing' if best_power == float('inf') else f'power={best_power}'}")

    # 3. Scipy global weights
    print("\n  Optimising global weights (Nelder-Mead, 5 restarts)...")
    opt_w = optimise_global_weights(eval_probs, eval_labels, n_restarts=5)
    gw_probs_eval = global_weight_ensemble(eval_probs, opt_w)
    s3 = print_scores("3. Scipy global weights", eval_labels, gw_probs_eval)
    print(f"     Weights: {dict(zip(short_names, [f'{w:.3f}' for w in opt_w]))}")

    # 4. Class weights + scipy (combine both)
    # Use class-weighted probs as "single model", then apply bias tuning
    best_eval_probs = cw_probs_eval if s2 >= s3 else gw_probs_eval
    best_label = "per-class F1" if s2 >= s3 else "scipy global"

    # 5. Per-class bias tuning on the best strategy so far
    if not args.no_bias_tuning:
        print(f"\n  Tuning per-class biases on top of {best_label} ensemble...")
        biases = optimise_class_biases(best_eval_probs, eval_labels)
        biased_probs_eval = best_eval_probs + biases[np.newaxis, :]
        s4 = print_scores(f"4. {best_label} + bias tuning", eval_labels, biased_probs_eval)
        print(f"     Biases: { {c: f'{biases[i]:+.3f}' for i, c in enumerate(CLASS_NAMES)} }")
    else:
        biases = None
        s4 = -1.0

    # ---- Per-class F1 breakdown for best strategy ----
    best_score = max(s1, s2, s3, s4)
    if s4 == best_score and not args.no_bias_tuning:
        final_eval = biased_probs_eval
        final_label = f"{best_label} + bias tuning"
    elif s3 == best_score:
        final_eval = gw_probs_eval
        final_label = "scipy global weights"
    elif s2 == best_score:
        final_eval = cw_probs_eval
        final_label = "per-class F1 weights"
    else:
        final_eval = eq_probs_eval
        final_label = "equal weights"

    print(f"\n{'=' * 70}")
    print(f"Best strategy: {final_label}  (composite={best_score:.4f})")
    print(f"{'=' * 70}")
    report = classification_report(eval_labels, final_eval.argmax(axis=1),
                                   target_names=CLASS_NAMES, zero_division=0)
    print(report)

    # ---- Apply best strategy to test set ----
    print(f"\nGenerating test submission using: {final_label}")

    eq_probs_test  = equal_weight_ensemble(test_probs)
    cw_probs_test  = class_weight_ensemble(test_probs, f1_mat)
    gw_probs_test  = global_weight_ensemble(test_probs, opt_w)

    if s4 == best_score and biases is not None:
        base_test = cw_probs_test if s2 >= s3 else gw_probs_test
        final_test = base_test + biases[np.newaxis, :]
    elif s3 == best_score:
        final_test = gw_probs_test
    elif s2 == best_score:
        final_test = cw_probs_test
    else:
        final_test = eq_probs_test

    save_submission(final_test, test_ids, out / 'submission_smart.csv')

    # Also save individual strategy outputs for comparison
    save_submission(eq_probs_test,  test_ids, out / 'submission_equal.csv')
    save_submission(cw_probs_test,  test_ids, out / 'submission_classweighted.csv')
    save_submission(gw_probs_test,  test_ids, out / 'submission_globalopt.csv')

    # Save biases for reproducibility
    if biases is not None:
        np.save(out / 'class_biases.npy', biases)
        pd.DataFrame({'class': CLASS_NAMES, 'bias': biases}).to_csv(
            out / 'class_biases.csv', index=False)

    print(f"\nAll outputs written to: {out}")


if __name__ == '__main__':
    np.random.seed(42)
    main()
