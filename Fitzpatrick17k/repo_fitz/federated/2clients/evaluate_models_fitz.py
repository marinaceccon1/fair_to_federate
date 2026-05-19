"""
evaluate_models_fitz.py  (v4 — optimal F1 threshold from validation set)
-----------------------------------------------------------------
Each client is described by 3 independent features:
    portion, fitz14_frac, flip_frac
(fitz56_frac = 1 - fitz14_frac is redundant and omitted everywhere.)

Threshold selection
-------------------
  For each model, the threshold that maximises F1 on the validation set
  is found via precision_recall_curve and applied to the test set.
  This affects bal_acc, hard_tpr_*, and hard_tpr_gap/min_hard_tpr.
  AUC and soft_tpr are threshold-free and unaffected.

Metrics computed per model
--------------------------
  Global
    auc          ROC-AUC over the full test set
    bal_acc      Balanced accuracy (threshold = best val F1)

  Per-group AUC  (fitz_14 vs fitz_56)
    auc_14       ROC-AUC restricted to fitz_14 samples
    auc_56       ROC-AUC restricted to fitz_56 samples
    auc_gap      |auc_14 - auc_56|
    min_auc      min(auc_14, auc_56)

  Per-group soft TPR  (mean predicted prob among true positives)
    soft_tpr_14
    soft_tpr_56
    soft_tpr_gap
    min_soft_tpr

  Per-group hard TPR  (recall at best val F1 threshold)
    hard_tpr_14
    hard_tpr_56
    hard_tpr_gap
    min_hard_tpr

Outputs
-------
  <output-dir>/standalone_metrics.csv
  <output-dir>/federated_metrics.csv
  <output-dir>/comparison_dataset.csv   <- 6 features + 42 metric columns

Usage
-----
    python evaluate_models_fitz.py [--model-dir best_models_fitz]
                                   [--seed 42] [--output-dir evaluation_results_fitz]
                                   [--batch-size 32] [--device cuda]
"""

import argparse, os, re, sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root anchored on __file__  (file lives in federated/)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, precision_recall_curve,
)
from src.param_grid_fitz import generate_param_grid

# ---------------------------------------------------------------------------
# Build valid config sets from param_grid for fast lookup
# ---------------------------------------------------------------------------
def _config_key(cfg):
    """Normalise a per-client config dict to a hashable key."""
    fitz14 = cfg["composition"].get("fitz_14", 0.0)
    return (round(cfg["portion"], 4),
            round(fitz14, 4),
            round(cfg["flip_frac"], 4))

_VALID_SINGLE = {_config_key(c[0]) for c in generate_param_grid(1)}
_VALID_PAIR   = {(_config_key(c[0]), _config_key(c[1]))
                 for c in generate_param_grid(2)}

def _parsed_key(d):
    """Convert a parsed filename dict to the same key format."""
    return (round(d["portion"], 4),
            round(d["fitz14_frac"], 4),
            round(d["flip_frac"], 4))

def is_valid_standalone(cfg):
    return _parsed_key(cfg) in _VALID_SINGLE

def is_valid_federated(c0, c1):
    return (_parsed_key(c0), _parsed_key(c1)) in _VALID_PAIR

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model-dir",  default=os.environ.get("MODEL_DIR",  str(REPO_ROOT / "best_models_fitz")))
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", str(REPO_ROOT / "evaluation_results_fitz")))
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--device",     default=os.environ.get("EVAL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
args = parser.parse_args()

MODEL_DIR  = Path(args.model_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED, DEVICE, BATCH_SIZE = args.seed, torch.device(args.device), args.batch_size

print(f"\n{'='*70}\nFITZPATRICK MODEL EVALUATOR")
print(f"  model-dir={MODEL_DIR}  seed={SEED}  device={DEVICE}  out={OUTPUT_DIR}")
print(f"{'='*70}\n")

# ---------------------------------------------------------------------------
# Bootstrap CONFIG_PATH so data_setup_fitz imports cleanly
# ---------------------------------------------------------------------------
_DUMMY = OUTPUT_DIR / "_dummy_config.json"
_DUMMY.write_text('[{"portion":1.0,"composition":{"fitz_14":0.5,"fitz_56":0.5},"flip_frac":0.0}]')
os.environ["CONFIG_PATH"] = str(_DUMMY)

from src.data_setup_fitz import (
    create_model, SkinDataset, val_transform,
    _load_and_preprocess, carve_val_test_per_group,
    LABEL, FITZPATRICK_GROUPS, IMAGE_DIR,
)

# ===========================================================================
# 1.  FIXED TEST + VAL SETS
# ===========================================================================
print(f"Building val/test sets (seed={SEED})...")
group_dfs, N, n_val, n_test = _load_and_preprocess()
val_sets, test_sets, _ = carve_val_test_per_group(
    group_dfs, LABEL, N, n_val, n_test, seed=SEED
)
val_df  = pd.concat(list(val_sets.values()),  ignore_index=True)
test_df = pd.concat(list(test_sets.values()), ignore_index=True)
print(f"  Val  set: {len(val_df)} samples")
print(f"  Test set: {len(test_df)} samples\n")

val_loader = DataLoader(
    SkinDataset(val_df,  root_dir=IMAGE_DIR, transform=val_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
)
test_loader = DataLoader(
    SkinDataset(test_df, root_dir=IMAGE_DIR, transform=val_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
)

# ===========================================================================
# 2.  INFERENCE
# ===========================================================================
def run_inference(model, loader):
    """Return (probs, labels, fitz) arrays for the given loader."""
    model.eval()
    probs_l, labels_l, fitz_l = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(DEVICE).float()).cpu().squeeze(1).numpy()
            probs_l.append(1.0 / (1.0 + np.exp(-logits)))
            labels_l.append(batch[LABEL].numpy())
            fitz_l.append(batch["fitzpatrick"].numpy())
    return (
        np.concatenate(probs_l),
        np.concatenate(labels_l),
        np.concatenate(fitz_l),
    )


# ===========================================================================
# 3.  THRESHOLD SELECTION
# ===========================================================================
def best_f1_threshold(val_labels, val_probs):
    """Return the threshold on val_probs that maximises F1."""
    precision, recall, thresholds = precision_recall_curve(val_labels, val_probs)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1
    return float(thresholds[np.argmax(f1[:-1])])


# ===========================================================================
# 4.  METRICS
# ===========================================================================
def compute_metrics(probs, labels, fitz, threshold):
    """Compute all metrics using the supplied threshold for hard decisions."""
    preds = (probs >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Global metrics
    # ------------------------------------------------------------------
    auc     = roc_auc_score(labels, probs)
    bal_acc = balanced_accuracy_score(labels, preds)

    # ------------------------------------------------------------------
    # Group masks
    # ------------------------------------------------------------------
    mask14 = np.isin(fitz, FITZPATRICK_GROUPS["fitz_14"])
    mask56 = np.isin(fitz, FITZPATRICK_GROUPS["fitz_56"])

    # ------------------------------------------------------------------
    # Per-group AUC  (threshold-free)
    # ------------------------------------------------------------------
    def group_auc(mask):
        y, p = labels[mask], probs[mask]
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))

    auc_14 = group_auc(mask14)
    auc_56 = group_auc(mask56)
    auc_nan = np.isnan(auc_14) or np.isnan(auc_56)

    # ------------------------------------------------------------------
    # Per-group soft TPR  (mean predicted prob among true positives)
    # ------------------------------------------------------------------
    def soft_tpr(mask):
        y_true  = labels[mask]
        y_probs = probs[mask]
        pos = y_true == 1
        return float(y_probs[pos].mean()) if pos.sum() > 0 else float("nan")

    soft_tpr_14 = soft_tpr(mask14)
    soft_tpr_56 = soft_tpr(mask56)
    stpr_nan = np.isnan(soft_tpr_14) or np.isnan(soft_tpr_56)

    # ------------------------------------------------------------------
    # Per-group hard TPR  (recall at best-val-F1 threshold)
    # ------------------------------------------------------------------
    def hard_tpr(mask):
        y_true  = labels[mask]
        y_preds = preds[mask]          # already thresholded above
        pos = y_true == 1
        if pos.sum() == 0:
            return float("nan")
        return float(y_preds[pos].mean())   # TP / (TP + FN)

    hard_tpr_14 = hard_tpr(mask14)
    hard_tpr_56 = hard_tpr(mask56)
    htpr_nan = np.isnan(hard_tpr_14) or np.isnan(hard_tpr_56)

    return {
        # global
        "auc":           auc,
        "bal_acc":       bal_acc,
        # per-group AUC
        "auc_14":        auc_14,
        "auc_56":        auc_56,
        "auc_gap":       float("nan") if auc_nan  else abs(auc_14 - auc_56),
        "min_auc":       float("nan") if auc_nan  else min(auc_14, auc_56),
        # per-group soft TPR
        "soft_tpr_14":   soft_tpr_14,
        "soft_tpr_56":   soft_tpr_56,
        "soft_tpr_gap":  float("nan") if stpr_nan else abs(soft_tpr_14 - soft_tpr_56),
        "min_soft_tpr":  float("nan") if stpr_nan else min(soft_tpr_14, soft_tpr_56),
        # per-group hard TPR
        "hard_tpr_14":   hard_tpr_14,
        "hard_tpr_56":   hard_tpr_56,
        "hard_tpr_gap":  float("nan") if htpr_nan else abs(hard_tpr_14 - hard_tpr_56),
        "min_hard_tpr":  float("nan") if htpr_nan else min(hard_tpr_14, hard_tpr_56),
    }


def load_model(path):
    m = create_model().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE), strict=True)
    return m

# ===========================================================================
# 5.  FILENAME PARSERS
#   standalone: best_standalone_model_fitz__C0_p<P>_14x<P14>_56x<P56>_flip<FF>.pt
#   federated:  best_global_model_fitz__C0_...__C1_....pt
# ===========================================================================
_TOK = re.compile(r"C(\d+)_p([\d.]+)_14x(\d+)_56x(\d+)_flip([\d.]+)")

def _parse_token(tok):
    m = _TOK.match(tok.strip())
    if not m:
        raise ValueError(f"Bad token: {tok!r}")
    return {
        "portion":     float(m.group(2)),
        "fitz14_frac": int(m.group(3)) / 100.0,
        "flip_frac":   float(m.group(5)),
    }

def parse_standalone(stem):
    pfx = "best_standalone_model_fitz__"
    if not stem.startswith(pfx): return None
    toks = stem[len(pfx):].split("__")
    return _parse_token(toks[0]) if len(toks) == 1 else None

def parse_federated(stem):
    pfx = "best_global_model_fitz__"
    if not stem.startswith(pfx): return None
    toks = stem[len(pfx):].split("__")
    if len(toks) != 2: return None
    try:
        return _parse_token(toks[0]), _parse_token(toks[1])
    except ValueError:
        return None

# ===========================================================================
# 6.  SCAN + EVALUATE
# ===========================================================================
pt_files = sorted(MODEL_DIR.glob("*.pt"))
print(f"Found {len(pt_files)} .pt files in {MODEL_DIR}\n")

sa_rows, fl_rows = [], []

MCOLS = [
    "auc", "bal_acc",
    "auc_14", "auc_56", "auc_gap", "min_auc",
    "soft_tpr_14", "soft_tpr_56", "soft_tpr_gap", "min_soft_tpr",
    "hard_tpr_14", "hard_tpr_56", "hard_tpr_gap", "min_hard_tpr",
]

def _fmt(m, threshold):
    return (
        f"threshold={threshold:.4f}  "
        f"auc={m['auc']:.4f}  bal={m['bal_acc']:.4f}  "
        f"auc14={m['auc_14']:.4f}  auc56={m['auc_56']:.4f}  auc_gap={m['auc_gap']:.4f}  "
        f"stpr14={m['soft_tpr_14']:.4f}  stpr56={m['soft_tpr_56']:.4f}  "
        f"htpr14={m['hard_tpr_14']:.4f}  htpr56={m['hard_tpr_56']:.4f}"
    )

for i, pt in enumerate(pt_files):
    tag = f"[{i+1}/{len(pt_files)}]"

    cfg = parse_standalone(pt.stem)
    if cfg is not None:
        if not is_valid_standalone(cfg):
            print(f"{tag} SKIP (not in param_grid): {pt.name}")
            continue
        print(f"{tag} STANDALONE  {pt.name}")
        try:
            model = load_model(pt)
            # Find optimal threshold on val set
            val_probs, val_labels, _ = run_inference(model, val_loader)
            threshold = best_f1_threshold(val_labels, val_probs)
            # Evaluate on test set with that threshold
            test_probs, test_labels, test_fitz = run_inference(model, test_loader)
            m = compute_metrics(test_probs, test_labels, test_fitz, threshold)
            sa_rows.append({**cfg, **m, "threshold": threshold, "filename": pt.name})
            print(f"      {_fmt(m, threshold)}")
        except Exception as e:
            print(f"  ERROR: {e}")
        continue

    res = parse_federated(pt.stem)
    if res is not None:
        c0, c1 = res
        if not is_valid_federated(c0, c1):
            print(f"{tag} SKIP (not in param_grid): {pt.name}")
            continue
        print(f"{tag} FEDERATED   {pt.name}")
        try:
            model = load_model(pt)
            # Find optimal threshold on val set
            val_probs, val_labels, _ = run_inference(model, val_loader)
            threshold = best_f1_threshold(val_labels, val_probs)
            # Evaluate on test set with that threshold
            test_probs, test_labels, test_fitz = run_inference(model, test_loader)
            m = compute_metrics(test_probs, test_labels, test_fitz, threshold)
            fl_rows.append({
                "c0_portion":     c0["portion"],
                "c0_fitz14_frac": c0["fitz14_frac"],
                "c0_flip_frac":   c0["flip_frac"],
                "c1_portion":     c1["portion"],
                "c1_fitz14_frac": c1["fitz14_frac"],
                "c1_flip_frac":   c1["flip_frac"],
                **m, "threshold": threshold, "filename": pt.name,
            })
            print(f"      {_fmt(m, threshold)}")
        except Exception as e:
            print(f"  ERROR: {e}")
        continue

    print(f"{tag} SKIP: {pt.name}")

# ===========================================================================
# 7.  SAVE INDIVIDUAL TABLES
# ===========================================================================
sa_df = pd.DataFrame(sa_rows)
fl_df = pd.DataFrame(fl_rows)
sa_path = OUTPUT_DIR / "standalone_metrics.csv"
fl_path = OUTPUT_DIR / "federated_metrics.csv"
sa_df.to_csv(sa_path, index=False)
fl_df.to_csv(fl_path, index=False)
print(f"\nStandalone metrics -> {sa_path}  ({len(sa_df)} models)")
print(f"Federated  metrics -> {fl_path}  ({len(fl_df)} models)")

# ===========================================================================
# 8.  COMPARISON DATASET
# ===========================================================================
print("\nBuilding comparison dataset...")
sa_idx = sa_df.set_index(["portion", "fitz14_frac", "flip_frac"])
comp_rows, n_miss = [], 0

for _, fl in fl_df.iterrows():
    key = (fl["c0_portion"], fl["c0_fitz14_frac"], fl["c0_flip_frac"])
    if key not in sa_idx.index:
        n_miss += 1
        continue
    sa = sa_idx.loc[key]
    if isinstance(sa, pd.DataFrame):
        sa = sa.iloc[0]
    comp_rows.append({
        # client-0 features (3)
        "c0_portion":     fl["c0_portion"],
        "c0_fitz14_frac": fl["c0_fitz14_frac"],
        "c0_flip_frac":   fl["c0_flip_frac"],
        # client-1 features (3)
        "c1_portion":     fl["c1_portion"],
        "c1_fitz14_frac": fl["c1_fitz14_frac"],
        "c1_flip_frac":   fl["c1_flip_frac"],
        # standalone metrics  (14 cols)
        **{f"sa_{mc}": float(sa[mc]) for mc in MCOLS},
        # federated metrics   (14 cols)
        **{f"fl_{mc}": float(fl[mc]) for mc in MCOLS},
        # deltas FL - standalone (14 cols)
        **{f"delta_{mc}": float(fl[mc]) - float(sa[mc]) for mc in MCOLS},
    })

comp_df = pd.DataFrame(comp_rows)
comp_path = OUTPUT_DIR / "comparison_dataset.csv"
comp_df.to_csv(comp_path, index=False)
print(f"Comparison dataset -> {comp_path}  ({len(comp_df)} rows, "
      f"{6 + 3*len(MCOLS)} columns)")
if n_miss:
    print(f"WARNING: {n_miss} FL rows skipped (no matching standalone model)")

# ===========================================================================
# 9.  SUMMARY
# ===========================================================================
print(f"\n{'='*70}\nSUMMARY")
print(f"  Standalone : {len(sa_df)} models")
print(f"  Federated  : {len(fl_df)} models")
print(f"  Comparison : {len(comp_df)} rows")
print(f"  Files: {sa_path}, {fl_path}, {comp_path}")
print(f"{'='*70}\n")
if len(comp_df):
    print("Delta summary (FL - standalone):")
    print(comp_df[[f"delta_{mc}" for mc in MCOLS]].describe().round(4).to_string())