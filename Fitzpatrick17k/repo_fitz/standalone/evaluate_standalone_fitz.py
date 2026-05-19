"""
evaluate_standalone_fitz.py
────────────────────────────
Full evaluation script for EfficientNet-B2 models.
Computes Soft TPR, Hard TPR, and ROC-AUC per Fitzpatrick group.
Modified to use an optimal threshold (max F1) from the validation set.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score, precision_recall_curve

# ---------------------------------------------------------------------------
# Repo root anchored on __file__
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# =============================================================================
# PARAM GRID & BOOTSTRAP
# =============================================================================
from src.param_grid_fitz import generate_param_grid

_all_grid_configs = generate_param_grid(num_clients=1)
_bootstrap_cfg = _all_grid_configs[0]
_BOOTSTRAP_FILE = Path(tempfile.gettempdir()) / "_eval_bootstrap_config.json"
with open(_BOOTSTRAP_FILE, "w") as _bf:
    json.dump(_bootstrap_cfg, _bf)

os.environ["CONFIG_PATH"] = str(_BOOTSTRAP_FILE)

import src.data_setup_fitz as _ds

create_model = _ds.create_model
LABEL        = _ds.LABEL
FTZ_COL      = "fitzpatrick"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(description="Evaluate standalone Fitzpatrick models")
parser.add_argument("--save-dir",   default=os.environ.get("SAVE_DIR", str(REPO_ROOT / "best_models_fitz")))
parser.add_argument("--out-dir",    default="results")
parser.add_argument("--data-seed",  type=int, default=int(os.environ.get("DATA_SEED", "42")))
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args() 

SAVE_DIR  = Path(args.save_dir)
OUT_DIR   = Path(args.out_dir)
DATA_SEED = args.data_seed
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

# =============================================================================
# HELPERS
# =============================================================================
def format_config_tag(client_configs: list) -> str:
    parts = []
    for i, c in enumerate(client_configs):
        comp = c["composition"]
        comp_str = f"14x{int(comp['fitz_14'] * 100)}_56x{int(comp['fitz_56'] * 100)}"
        parts.append(f"C{i}_p{c['portion']}_{comp_str}_flip{c['flip_frac']}")
    return "__".join(parts)

GROUP_14 = {1, 2, 3, 4}
GROUP_56 = {5, 6}

def compute_group_metrics(labels, probs, preds, mask):
    y_true  = labels[mask]
    y_probs = probs[mask]
    y_preds = preds[mask]
    
    pos_mask = (y_true == 1)
    n_pos = int(pos_mask.sum())
    
    # 1. Soft TPR (Mean Prob of Positives)
    soft_tpr = float(y_probs[pos_mask].mean()) if n_pos > 0 else float("nan")
    
    # 2. Hard TPR (Recall at calculated threshold)
    hard_tpr = float(recall_score(y_true, y_preds, zero_division=0)) if n_pos > 0 else float("nan")
    
    # 3. AUC
    try:
        if len(np.unique(y_true)) > 1:
            auc_val = float(roc_auc_score(y_true, y_probs))
        else:
            auc_val = float("nan")
    except:
        auc_val = float("nan")
        
    return soft_tpr, hard_tpr, auc_val

# =============================================================================
# MODEL DISCOVERY
# =============================================================================
valid_tag_to_config = {format_config_tag(cfg): cfg[0] for cfg in _all_grid_configs}
PREFIX = "best_standalone_model_fitz__"
model_files = sorted(SAVE_DIR.glob(f"{PREFIX}*.pt"))

valid_models = []
for pt_path in model_files:
    tag = pt_path.stem[len(PREFIX):]
    if tag in valid_tag_to_config:
        valid_models.append((pt_path, tag, valid_tag_to_config[tag]))

if not valid_models:
    print(f"❌ No valid models found in {SAVE_DIR} matching the param grid.")
    sys.exit(1)

print(f"🔍 Found {len(valid_models)} models to evaluate.\n")

# =============================================================================
# EVALUATION LOOP
# =============================================================================
rows = []

for idx, (pt_path, tag, cfg) in enumerate(valid_models, 1):
    print(f"[{idx}/{len(valid_models)}] Processing: {tag}")
    
    model = create_model().to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()

    _ds.client_configs = [cfg]
    group_dfs, N, n_val, n_test = _ds._load_and_preprocess()
    
    # Use deterministic seeding for split to match training
    val_sets, test_sets, _ = _ds.carve_val_test_per_group(group_dfs, LABEL, N, n_val, n_test, seed=DATA_SEED)
    
    # --- STEP 1: FIND OPTIMAL THRESHOLD ON VALIDATION SET ---
    val_df = pd.concat(list(val_sets.values()), ignore_index=True)
    val_ds = _ds.SkinDataset(val_df, root_dir=_ds.IMAGE_DIR, transform=_ds.val_transform)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    val_labels, val_probs = [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            val_labels.append(batch[LABEL].cpu().numpy().astype(int).ravel())
            logits = model(imgs.float()).cpu().numpy().ravel()
            val_probs.append(1.0 / (1.0 + np.exp(-logits))) # Manual Sigmoid calculation

    val_labels = np.concatenate(val_labels)
    val_probs = np.concatenate(val_probs)

    # Find threshold maximizing F1 score
    precision, recall, thresholds = precision_recall_curve(val_labels, val_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    # Thresholds array is N-1 compared to precision/recall
    best_threshold = float(thresholds[np.argmax(f1_scores[:-1])])
    
    print(f"  🎯 Optimal Threshold (Val F1): {best_threshold:.4f}")

    # --- STEP 2: EVALUATE ON TEST SET ---
    test_df = pd.concat(list(test_sets.values()), ignore_index=True)
    test_ds = _ds.SkinDataset(test_df, root_dir=_ds.IMAGE_DIR, transform=_ds.val_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_labels, all_logits, all_fitz = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            all_labels.append(batch[LABEL].cpu().numpy().astype(int).ravel())
            all_logits.append(model(imgs.float()).cpu().numpy().ravel())
            all_fitz.append(batch[FTZ_COL].cpu().numpy().astype(int).ravel())

    all_labels = np.concatenate(all_labels)
    all_probs  = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits))) 
    
    # Apply optimal threshold found in Step 1
    all_preds  = (all_probs >= best_threshold).astype(int)
    all_fitz   = np.concatenate(all_fitz)

    m14 = np.isin(all_fitz, list(GROUP_14))
    m56 = np.isin(all_fitz, list(GROUP_56))

    soft14, hard14, auc14 = compute_group_metrics(all_labels, all_probs, all_preds, m14)
    soft56, hard56, auc56 = compute_group_metrics(all_labels, all_probs, all_preds, m56)

    rows.append({
        "config_tag": tag,
        "portion": cfg["portion"],
        "fitz_14_frac": cfg["composition"]["fitz_14"],
        "fitz_56_frac": cfg["composition"]["fitz_56"],
        "flip_frac": cfg["flip_frac"],
        "best_threshold": round(best_threshold, 4),
        "global_auc": round(roc_auc_score(all_labels, all_probs), 4),
        "balanced_acc": round(balanced_accuracy_score(all_labels, all_preds), 4),
        
        "soft_tpr_14": round(soft14, 4), "soft_tpr_56": round(soft56, 4),
        "soft_tpr_gap": round(abs(soft14 - soft56), 4),
        
        "hard_tpr_14": round(hard14, 4), "hard_tpr_56": round(hard56, 4),
        "hard_tpr_gap": round(abs(hard14 - hard56), 4),
        
        "auc_14": round(auc14, 4), "auc_56": round(auc56, 4),
        "auc_gap": round(abs(auc14 - auc56), 4),
        "min_auc": round(min(auc14, auc56), 4) if not (np.isnan(auc14) or np.isnan(auc56)) else np.nan
    })
    torch.cuda.empty_cache()

# =============================================================================
# SAVE
# =============================================================================
results_df = pd.DataFrame(rows)
out_csv = OUT_DIR / "standalone_eval_comprehensive.csv"
results_df.to_csv(out_csv, index=False)

print(f"\n✅ Finished! Results saved to: {out_csv}")