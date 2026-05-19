"""
evaluate_models_fitz_Nc.py
──────────────────────────
Evaluator for N-client (N > 2) Fitzpatrick FL setups.

Each client is described by 3 independent features:
    portion, fitz14_frac, flip_frac
(fitz56_frac = 1 - fitz14_frac is redundant and omitted everywhere.)

Metrics computed per model
--------------------------
  Global
    auc          ROC-AUC over the full test set
    bal_acc      Balanced accuracy (threshold = best val F1)

  Per-group AUC  (fitz_14 vs fitz_56)
    auc_14, auc_56, auc_gap, min_auc

  Per-group soft TPR  (mean predicted prob among true positives)
    soft_tpr_14, soft_tpr_56, soft_tpr_gap, min_soft_tpr

  Per-group hard TPR  (recall at best val F1 threshold)
    hard_tpr_14, hard_tpr_56, hard_tpr_gap, min_hard_tpr

Filename conventions expected
------------------------------
  Standalone:
    best_standalone_model_fitz__C0_p<P>_14x<P14>_56x<P56>_flip<FF>.pt

  N-client federated:
    best_global_model_fitz__Nc__C0_...__C1_...__...__CN-1_....pt
    (N double-underscore-separated client tokens, prefixed with Nc)

Outputs
-------
  <output-dir>/standalone_metrics.csv
  <output-dir>/federated_{N}c_metrics.csv
  <output-dir>/comparison_dataset_{N}c.csv

Usage
-----
    python evaluate_models_fitz_Nc.py --num-clients 3 \\
        [--model-dir /path/to/models] [--seed 42] \\
        [--output-dir results] [--batch-size 32] [--device cuda:0]
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, precision_recall_curve,
)

# Repo root: two levels up from this file (repo/federated/Nclients/)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from param_grid_fitz_Nc import (
    generate_param_grid, get_client0_strata, _aggregate, _tuple_to_dict,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--num-clients", type=int, required=True,
                    help="Number of FL clients in the federated setup (> 2)")
parser.add_argument("--model-dir",  default=None,
                    help="Directory containing .pt model files "
                         "(default: repo/best_models_fitz_{N}c)")
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--output-dir", default=None,
                    help="Output directory for CSVs "
                         "(default: repo/evaluation_results_fitz_{N}c)")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

NUM_CLIENTS = args.num_clients
if NUM_CLIENTS < 3:
    parser.error("--num-clients must be >= 3")

MODEL_DIR  = Path(args.model_dir) if args.model_dir else \
             _REPO_ROOT / f"best_models_fitz_{NUM_CLIENTS}c"
OUTPUT_DIR = Path(args.output_dir) if args.output_dir else \
             _REPO_ROOT / f"evaluation_results_fitz_{NUM_CLIENTS}c"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED       = args.seed
DEVICE     = torch.device(args.device)
BATCH_SIZE = args.batch_size

print(f"\n{'='*70}")
print(f"FITZPATRICK {NUM_CLIENTS}-CLIENT MODEL EVALUATOR")
print(f"  model-dir  = {MODEL_DIR}")
print(f"  output-dir = {OUTPUT_DIR}")
print(f"  seed={SEED}  device={DEVICE}  batch-size={BATCH_SIZE}")
print(f"{'='*70}\n")

# ---------------------------------------------------------------------------
# Bootstrap CONFIG_PATH so data_setup_fitz imports cleanly
# ---------------------------------------------------------------------------
_DUMMY = OUTPUT_DIR / "_dummy_config.json"
_DUMMY.write_text(
    '[{"portion":1.0,"composition":{"fitz_14":0.5,"fitz_56":0.5},"flip_frac":0.0}]'
)
os.environ["CONFIG_PATH"] = str(_DUMMY)

from src.data_setup_fitz import (
    create_model, SkinDataset, val_transform,
    _load_and_preprocess, carve_val_test_per_group,
    LABEL, FITZPATRICK_GROUPS, IMAGE_DIR,
)

# ---------------------------------------------------------------------------
# Build valid config sets from param_grid for fast lookup
# ---------------------------------------------------------------------------
def _config_key(cfg):
    fitz14 = cfg["composition"].get("fitz_14", 0.0)
    return (round(cfg["portion"], 4), round(fitz14, 4), round(cfg["flip_frac"], 4))

def _internal_to_tuple(d):
    return (round(d["portion"], 4), d["composition"], round(d["flip_frac"], 4))

def _parsed_to_tuple(d):
    fitz14      = round(d["fitz14_frac"], 4)
    composition = {"fitz_14": fitz14, "fitz_56": round(1.0 - fitz14, 4)}
    return (round(d["portion"], 4), composition, round(d["flip_frac"], 4))

# Standalone validity: single-client options (same for any N)
_VALID_SINGLE = {
    (round(p, 4), round(comp["fitz_14"], 4), round(f, 4))
    for c0 in get_client0_strata()
    for p, comp, f in [(c0["portion"], c0["composition"], c0["flip_frac"])]
}

# N-client federated validity: (client0_key, aggregate_key)
_VALID_C0_AGG = {
    (_config_key(cfg[0]),
     _aggregate(*[_internal_to_tuple(cfg[i]) for i in range(1, NUM_CLIENTS)]))
    for cfg in generate_param_grid(NUM_CLIENTS)
}

def _parsed_key(d):
    return (round(d["portion"], 4), round(d["fitz14_frac"], 4), round(d["flip_frac"], 4))

def is_valid_standalone(cfg):
    return _parsed_key(cfg) in _VALID_SINGLE

def is_valid_federated_Nc(*clients):
    """
    clients = (c0_dict, c1_dict, …, cN-1_dict) — parsed-filename dicts.
    Valid if (c0_key, aggregate_key) appears in the budgeted param grid.
    """
    c0      = clients[0]
    rest    = clients[1:]
    c0_key  = _parsed_key(c0)
    agg_key = _aggregate(*[_parsed_to_tuple(c) for c in rest])
    return (c0_key, agg_key) in _VALID_C0_AGG


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
    model.eval()
    probs_l, labels_l, fitz_l = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs, labels, fitz = batch["image"], batch["label"], batch["fitz"]
            probs = torch.sigmoid(model(imgs.to(DEVICE))).cpu().numpy()
            probs_l.append(probs)
            labels_l.append(labels.numpy())
            fitz_l.append(fitz.numpy())
    return (np.concatenate(probs_l).ravel(),
            np.concatenate(labels_l).ravel(),
            np.concatenate(fitz_l).ravel())


# ===========================================================================
# 3.  THRESHOLD SELECTION
# ===========================================================================
def best_f1_threshold(labels, probs):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(thresholds[np.argmax(f1[:-1])])


# ===========================================================================
# 4.  METRICS
# ===========================================================================
def compute_metrics(probs, labels, fitz, threshold):
    preds   = (probs >= threshold).astype(int)
    auc     = float(roc_auc_score(labels, probs))
    bal_acc = float(balanced_accuracy_score(labels, preds))

    groups  = FITZPATRICK_GROUPS

    def _group_auc(grp):
        mask = np.isin(fitz, groups[grp])
        if mask.sum() == 0 or len(np.unique(labels[mask])) < 2:
            return float("nan")
        return float(roc_auc_score(labels[mask], probs[mask]))

    def _soft_tpr(grp):
        mask = np.isin(fitz, groups[grp]) & (labels == 1)
        return float(probs[mask].mean()) if mask.sum() > 0 else float("nan")

    def _hard_tpr(grp):
        mask = np.isin(fitz, groups[grp])
        if mask.sum() == 0:
            return float("nan")
        tp = ((preds[mask] == 1) & (labels[mask] == 1)).sum()
        fn = ((preds[mask] == 0) & (labels[mask] == 1)).sum()
        return float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")

    auc_14 = _group_auc("fitz_14")
    auc_56 = _group_auc("fitz_56")
    st_14  = _soft_tpr("fitz_14")
    st_56  = _soft_tpr("fitz_56")
    ht_14  = _hard_tpr("fitz_14")
    ht_56  = _hard_tpr("fitz_56")

    auc_nan  = np.isnan(auc_14) or np.isnan(auc_56)
    stpr_nan = np.isnan(st_14)  or np.isnan(st_56)
    htpr_nan = np.isnan(ht_14)  or np.isnan(ht_56)

    return {
        "auc":          auc,
        "bal_acc":      bal_acc,
        "auc_14":       auc_14,
        "auc_56":       auc_56,
        "auc_gap":      float("nan") if auc_nan  else abs(auc_14 - auc_56),
        "min_auc":      float("nan") if auc_nan  else min(auc_14, auc_56),
        "soft_tpr_14":  st_14,
        "soft_tpr_56":  st_56,
        "soft_tpr_gap": float("nan") if stpr_nan else abs(st_14 - st_56),
        "min_soft_tpr": float("nan") if stpr_nan else min(st_14, st_56),
        "hard_tpr_14":  ht_14,
        "hard_tpr_56":  ht_56,
        "hard_tpr_gap": float("nan") if htpr_nan else abs(ht_14 - ht_56),
        "min_hard_tpr": float("nan") if htpr_nan else min(ht_14, ht_56),
    }


# ===========================================================================
# 5.  MODEL LOADING
# ===========================================================================
def load_model(pt_path):
    model      = create_model().to(DEVICE)
    state_dict = torch.load(pt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    return model


# ===========================================================================
# 6.  FILENAME PARSERS
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
    if not stem.startswith(pfx):
        return None
    toks = stem[len(pfx):].split("__")
    return _parse_token(toks[0]) if len(toks) == 1 else None

def parse_federated_Nc(stem):
    """
    Parse an N-client federated model filename stem.
    Returns a tuple of N client dicts, or None if it doesn't match.

    Expected format:
        best_global_model_fitz__Nc__C0_...__C1_...__...__CN-1_...
    The leading 'Nc' sentinel is stripped before parsing the N client tokens.
    """
    pfx = "best_global_model_fitz__"
    if not stem.startswith(pfx):
        return None
    toks = stem[len(pfx):].split("__")
    # Strip leading Nc sentinel (e.g. "3c", "5c")
    if toks and re.fullmatch(r"\d+c", toks[0]):
        toks = toks[1:]
    if len(toks) != NUM_CLIENTS:
        return None
    try:
        return tuple(_parse_token(t) for t in toks)
    except ValueError:
        return None


# ===========================================================================
# 7.  SCAN + EVALUATE
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
        f"auc14={m['auc_14']:.4f}  auc56={m['auc_56']:.4f}  "
        f"stpr14={m['soft_tpr_14']:.4f}  stpr56={m['soft_tpr_56']:.4f}  "
        f"htpr14={m['hard_tpr_14']:.4f}  htpr56={m['hard_tpr_56']:.4f}"
    )

for i, pt in enumerate(pt_files):
    tag = f"[{i+1}/{len(pt_files)}]"

    # ── Standalone ────────────────────────────────────────────────────────────
    cfg = parse_standalone(pt.stem)
    if cfg is not None:
        if not is_valid_standalone(cfg):
            print(f"{tag} SKIP (not in param_grid): {pt.name}")
            continue
        print(f"{tag} STANDALONE  {pt.name}")
        try:
            model = load_model(pt)
            val_probs, val_labels, _   = run_inference(model, val_loader)
            threshold                  = best_f1_threshold(val_labels, val_probs)
            test_probs, test_labels, test_fitz = run_inference(model, test_loader)
            m = compute_metrics(test_probs, test_labels, test_fitz, threshold)
            sa_rows.append({**cfg, **m, "threshold": threshold, "filename": pt.name})
            print(f"      {_fmt(m, threshold)}")
        except Exception as e:
            print(f"  ERROR: {e}")
        continue

    # ── N-client federated ────────────────────────────────────────────────────
    res = parse_federated_Nc(pt.stem)
    if res is not None:
        if not is_valid_federated_Nc(*res):
            print(f"{tag} SKIP (not in param_grid): {pt.name}")
            continue
        print(f"{tag} FEDERATED-{NUM_CLIENTS}C  {pt.name}")
        try:
            model = load_model(pt)
            val_probs, val_labels, _         = run_inference(model, val_loader)
            threshold                        = best_f1_threshold(val_labels, val_probs)
            test_probs, test_labels, test_fitz = run_inference(model, test_loader)
            m = compute_metrics(test_probs, test_labels, test_fitz, threshold)
            # Build flat feature columns: cI_portion, cI_fitz14_frac, cI_flip_frac
            client_features = {}
            for ci, cd in enumerate(res):
                client_features[f"c{ci}_portion"]     = cd["portion"]
                client_features[f"c{ci}_fitz14_frac"] = cd["fitz14_frac"]
                client_features[f"c{ci}_flip_frac"]   = cd["flip_frac"]
            fl_rows.append({**client_features, **m,
                            "threshold": threshold, "filename": pt.name})
            print(f"      {_fmt(m, threshold)}")
        except Exception as e:
            print(f"  ERROR: {e}")
        continue

    print(f"{tag} SKIP (unrecognised filename): {pt.name}")


# ===========================================================================
# 8.  SAVE INDIVIDUAL TABLES
# ===========================================================================
sa_df = pd.DataFrame(sa_rows)
fl_df = pd.DataFrame(fl_rows)
sa_path = OUTPUT_DIR / "standalone_metrics.csv"
fl_path = OUTPUT_DIR / f"federated_{NUM_CLIENTS}c_metrics.csv"
sa_df.to_csv(sa_path, index=False)
fl_df.to_csv(fl_path, index=False)
print(f"\nStandalone metrics          -> {sa_path}  ({len(sa_df)} models)")
print(f"Federated-{NUM_CLIENTS}C metrics  -> {fl_path}  ({len(fl_df)} models)")


# ===========================================================================
# 9.  COMPARISON DATASET
# ===========================================================================
print(f"\nBuilding {NUM_CLIENTS}-client comparison dataset...")
sa_idx    = sa_df.set_index(["portion", "fitz14_frac", "flip_frac"])
comp_rows, n_miss = [], 0

for _, fl in fl_df.iterrows():
    key = (fl["c0_portion"], fl["c0_fitz14_frac"], fl["c0_flip_frac"])
    if key not in sa_idx.index:
        n_miss += 1
        continue
    sa = sa_idx.loc[key]
    if isinstance(sa, pd.DataFrame):
        sa = sa.iloc[0]

    # Client features: NUM_CLIENTS × 3 columns
    client_features = {}
    for ci in range(NUM_CLIENTS):
        client_features[f"c{ci}_portion"]     = fl[f"c{ci}_portion"]
        client_features[f"c{ci}_fitz14_frac"] = fl[f"c{ci}_fitz14_frac"]
        client_features[f"c{ci}_flip_frac"]   = fl[f"c{ci}_flip_frac"]

    comp_rows.append({
        **client_features,
        **{f"sa_{mc}":    float(sa[mc]) for mc in MCOLS},
        **{f"fl_{mc}":    float(fl[mc]) for mc in MCOLS},
        **{f"delta_{mc}": float(fl[mc]) - float(sa[mc]) for mc in MCOLS},
    })

comp_df   = pd.DataFrame(comp_rows)
comp_path = OUTPUT_DIR / f"comparison_dataset_{NUM_CLIENTS}c.csv"
comp_df.to_csv(comp_path, index=False)
n_feat_cols = NUM_CLIENTS * 3
print(f"Comparison dataset          -> {comp_path}  "
      f"({len(comp_df)} rows, {n_feat_cols + 3*len(MCOLS)} columns)")
if n_miss:
    print(f"WARNING: {n_miss} FL rows skipped (no matching standalone model for C0)")


# ===========================================================================
# 10.  SUMMARY
# ===========================================================================
print(f"\n{'='*70}\nSUMMARY")
print(f"  Standalone        : {len(sa_df)} models")
print(f"  Federated-{NUM_CLIENTS}C    : {len(fl_df)} models")
print(f"  Comparison        : {len(comp_df)} rows")
print(f"{'='*70}\n")
if len(comp_df):
    print(f"Delta summary (FL - standalone):")
    print(comp_df[[f"delta_{mc}" for mc in MCOLS]].describe().round(4).to_string())
