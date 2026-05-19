"""
Model evaluation script for 5-CLIENT federated models.

Analogous to evaluate_federated_5clients.py, but:
  - Targets models with exactly 5 clients
  - Reads from  best_models_5clients/  (filenames contain ``_5clients_``)
  - Saves results to evaluation_results_5clients/
  - Re-evaluates ALL matching models and backs up previous results

Metrics computed:
  - AUC (macro, micro, per-pathology, by sex)      → metrics_auc.csv
  - F1 (macro, per-pathology, by sex)              → metrics_f1.csv
  - Hard TPR (macro, micro, per-pathology, by sex) → metrics_tpr.csv
  - Soft TPR (micro, by sex, gap, min)             → metrics_soft_tpr.csv
  - Balanced Accuracy (macro, micro, by sex)       → metrics_balanced_accuracy.csv
  - Validation F1                                  → metrics_validation.csv
  - Per-pathology tables                           → metrics_pathology_<name>.csv
"""

import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set
import re
from datetime import datetime
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
base_path = str(REPO_ROOT)  # kept for import_nih_dfs / import_cxp_dfs

from torchvision import transforms
from torch.utils.data import DataLoader
from src.data.utils import import_nih_dfs, import_cxp_dfs
from src.federated.utils import preprocess_nih_cxp
from src.data.dataset import CheXpertAndNIH
from src.models.model import create_model

from src.federated.utils import (
    extract_predictions,
    compute_f1_scores,
    compute_tpr_scores,
    find_best_thresholds,
    print_f1_results,
    print_tpr_results,
)

# =============================================================================
# SKLEARN-BASED METRIC FUNCTIONS  (identical to 5-client evaluator)
# =============================================================================

def compute_auc_sklearn(outputs, targets, sex_labels, num_classes):
    outputs = outputs.T
    targets = targets.T
    sex_labels = np.asarray(sex_labels)

    aucs = {
        "per_pathology": [],
        "per_pathology_male": [],
        "per_pathology_female": [],
    }

    for c in range(num_classes):
        valid = ~np.isnan(targets[:, c])
        if valid.sum() == 0 or len(np.unique(targets[valid, c])) < 2:
            aucs["per_pathology"].append(np.nan)
            aucs["per_pathology_male"].append(np.nan)
            aucs["per_pathology_female"].append(np.nan)
            continue

        aucs["per_pathology"].append(
            roc_auc_score(targets[valid, c], outputs[valid, c])
        )

        male   = (sex_labels == 1.0) & valid
        female = (sex_labels == 0.0) & valid

        aucs["per_pathology_male"].append(
            roc_auc_score(targets[male, c], outputs[male, c])
            if male.sum() > 0 and len(np.unique(targets[male, c])) > 1 else np.nan
        )
        aucs["per_pathology_female"].append(
            roc_auc_score(targets[female, c], outputs[female, c])
            if female.sum() > 0 and len(np.unique(targets[female, c])) > 1 else np.nan
        )

    aucs["macro"]        = np.nanmean(aucs["per_pathology"])
    aucs["macro_male"]   = np.nanmean(aucs["per_pathology_male"])
    aucs["macro_female"] = np.nanmean(aucs["per_pathology_female"])

    flat_targets = targets.flatten()
    flat_outputs = outputs.flatten()
    flat_valid   = ~np.isnan(flat_targets)

    aucs["micro"] = roc_auc_score(flat_targets[flat_valid], flat_outputs[flat_valid])

    male_flat   = np.repeat(sex_labels == 1.0, num_classes) & flat_valid
    female_flat = np.repeat(sex_labels == 0.0, num_classes) & flat_valid

    aucs["micro_male"] = (
        roc_auc_score(flat_targets[male_flat], flat_outputs[male_flat])
        if male_flat.sum() > 0 and len(np.unique(flat_targets[male_flat])) > 1
        else np.nan
    )
    aucs["micro_female"] = (
        roc_auc_score(flat_targets[female_flat], flat_outputs[female_flat])
        if female_flat.sum() > 0 and len(np.unique(flat_targets[female_flat])) > 1
        else np.nan
    )

    # Backward-compatible aliases
    aucs["avg"]               = aucs["macro"]
    aucs["avg_male"]          = aucs["macro_male"]
    aucs["avg_female"]        = aucs["macro_female"]
    aucs["cumulative"]        = aucs["micro"]
    aucs["cumulative_male"]   = aucs["micro_male"]
    aucs["cumulative_female"] = aucs["micro_female"]

    return aucs


def compute_balanced_accuracy_sklearn(outputs, targets, sex_labels, thresholds, num_classes):
    outputs    = outputs.T
    targets    = targets.T
    sex_labels = np.asarray(sex_labels)

    preds = np.zeros_like(outputs, dtype=int)
    for c in range(num_classes):
        preds[:, c] = (outputs[:, c] >= thresholds[c]).astype(int)

    results = {
        "per_pathology":        {},
        "per_pathology_male":   {},
        "per_pathology_female": {},
    }

    for c in range(num_classes):
        valid = ~np.isnan(targets[:, c])
        if valid.sum() == 0:
            results["per_pathology"][c]        = np.nan
            results["per_pathology_male"][c]   = np.nan
            results["per_pathology_female"][c] = np.nan
            continue

        y_true = targets[valid, c]
        y_pred = preds[valid, c]
        results["per_pathology"][c] = balanced_accuracy_score(y_true, y_pred)

        male   = (sex_labels == 1) & valid
        female = (sex_labels == 0) & valid

        results["per_pathology_male"][c] = (
            balanced_accuracy_score(targets[male, c], preds[male, c])
            if male.sum() > 0 else np.nan
        )
        results["per_pathology_female"][c] = (
            balanced_accuracy_score(targets[female, c], preds[female, c])
            if female.sum() > 0 else np.nan
        )

    results["macro"]        = np.nanmean(list(results["per_pathology"].values()))
    results["macro_male"]   = np.nanmean(list(results["per_pathology_male"].values()))
    results["macro_female"] = np.nanmean(list(results["per_pathology_female"].values()))

    flat_valid = ~np.isnan(targets.flatten())
    results["micro"] = balanced_accuracy_score(
        targets.flatten()[flat_valid], preds.flatten()[flat_valid]
    )

    male_flat   = np.repeat(sex_labels == 1, num_classes) & flat_valid
    female_flat = np.repeat(sex_labels == 0, num_classes) & flat_valid

    results["micro_male"] = (
        balanced_accuracy_score(targets.flatten()[male_flat], preds.flatten()[male_flat])
        if male_flat.sum() > 0 and len(np.unique(targets.flatten()[male_flat])) > 1
        else np.nan
    )
    results["micro_female"] = (
        balanced_accuracy_score(targets.flatten()[female_flat], preds.flatten()[female_flat])
        if female_flat.sum() > 0 and len(np.unique(targets.flatten()[female_flat])) > 1
        else np.nan
    )

    return results


def print_auc_results(auc_results, num_classes):
    print("\n**AUC SCORES (sklearn)**")
    for p in range(num_classes):
        print(f"  Pathology {p} AUC:        {auc_results['per_pathology'][p]:.3f}")
        print(f"  Pathology {p} Male AUC:   {auc_results['per_pathology_male'][p]:.3f}")
        print(f"  Pathology {p} Female AUC: {auc_results['per_pathology_female'][p]:.3f}")
    print(f"\n  Macro AUC:        {auc_results['macro']:.3f}")
    print(f"  Macro Male AUC:   {auc_results['macro_male']:.3f}")
    print(f"  Macro Female AUC: {auc_results['macro_female']:.3f}")
    print(f"  Micro AUC:        {auc_results['micro']:.3f}")
    print(f"  Micro Male AUC:   {auc_results['micro_male']:.3f}")
    print(f"  Micro Female AUC: {auc_results['micro_female']:.3f}")


def print_balanced_accuracy_results(ba_results, num_classes):
    print("\n**BALANCED ACCURACY SCORES (sklearn)**")
    for p in range(num_classes):
        print(f"  Pathology {p} BA:        {ba_results['per_pathology'][p]:.3f}")
        print(f"  Pathology {p} Male BA:   {ba_results['per_pathology_male'][p]:.3f}")
        print(f"  Pathology {p} Female BA: {ba_results['per_pathology_female'][p]:.3f}")
    print(f"\n  Macro BA:        {ba_results['macro']:.3f}")
    print(f"  Macro Male BA:   {ba_results['macro_male']:.3f}")
    print(f"  Macro Female BA: {ba_results['macro_female']:.3f}")
    print(f"  Micro BA:        {ba_results['micro']:.3f}")
    print(f"  Micro Male BA:   {ba_results['micro_male']:.3f}")
    print(f"  Micro Female BA: {ba_results['micro_female']:.3f}")


# =============================================================================
# SOFT TPR  (threshold-free: mean predicted probability over true positives)
# Aggregation: micro (flattened across all samples and pathologies).
# NaN labels are skipped — only valid positives (target == 1) contribute.
# =============================================================================

def compute_soft_tpr_sklearn(outputs, targets, sex_labels, num_classes):
    """
    Soft TPR = mean(predicted_prob) over all (sample, pathology) pairs
    where the true label is 1 and not NaN.

    Returns a dict with:
        micro          – overall (all sexes flattened)
        micro_male     – males only
        micro_female   – females only
        gap            – micro_male - micro_female
        min            – min(micro_male, micro_female)
    """
    outputs    = outputs.T          # shape: (N, C)
    targets    = targets.T          # shape: (N, C)
    sex_labels = np.asarray(sex_labels)

    flat_targets = targets.flatten()
    flat_outputs = outputs.flatten()
    flat_sex     = np.repeat(sex_labels, num_classes)

    # Valid positive mask: label is 1 and not NaN
    pos_mask        = (flat_targets == 1) & ~np.isnan(flat_targets)
    pos_mask_male   = pos_mask & (flat_sex == 1.0)
    pos_mask_female = pos_mask & (flat_sex == 0.0)

    micro        = float(np.mean(flat_outputs[pos_mask]))        if pos_mask.sum()        > 0 else np.nan
    micro_male   = float(np.mean(flat_outputs[pos_mask_male]))   if pos_mask_male.sum()   > 0 else np.nan
    micro_female = float(np.mean(flat_outputs[pos_mask_female])) if pos_mask_female.sum() > 0 else np.nan

    gap = (micro_male - micro_female) if not (np.isnan(micro_male) or np.isnan(micro_female)) else np.nan
    min_soft_tpr = float(np.nanmin([micro_male, micro_female])) if not (np.isnan(micro_male) and np.isnan(micro_female)) else np.nan

    return {
        "micro":        micro,
        "micro_male":   micro_male,
        "micro_female": micro_female,
        "gap":          gap,
        "min":          min_soft_tpr,
    }


def print_soft_tpr_results(soft_tpr_results):
    print("\n**SOFT TPR SCORES (threshold-free, micro)**")
    print(f"  Micro Soft TPR:        {soft_tpr_results['micro']:.3f}")
    print(f"  Micro Male Soft TPR:   {soft_tpr_results['micro_male']:.3f}")
    print(f"  Micro Female Soft TPR: {soft_tpr_results['micro_female']:.3f}")
    print(f"  Gap (Male - Female):   {soft_tpr_results['gap']:.3f}")
    print(f"  Min(Male, Female):     {soft_tpr_results['min']:.3f}")


# =============================================================================
# PARAM GRID  (7-client models must satisfy these per-client constraints)
# =============================================================================
PORTIONS = [0.12, 0.06, 0.03]
GENDER_PROPORTIONS = [
    {"Male": 1.0, "Female": 0.0},
    {"Male": 0.5, "Female": 0.5},
    {"Male": 0.0, "Female": 1.0},
]
FLIP_FRACTIONS = [0.0, 0.15, 0.30, 0.45]


def is_valid_config(config: List[Dict]) -> bool:
    """Return True only for 7-client models whose configs match param_grid."""
    if len(config) != 5:
        return False
    for client in config:
        if client["portion"] not in PORTIONS:
            return False
        gender = client["gender"]
        if not any(
            abs(gender["Male"]   - g["Male"])   < 0.001 and
            abs(gender["Female"] - g["Female"]) < 0.001
            for g in GENDER_PROPORTIONS
        ):
            return False
        if client["flip_frac"] not in FLIP_FRACTIONS:
            return False
        # All-male → flip must be 0
        if abs(gender["Male"] - 1.0) < 0.001 and abs(client["flip_frac"]) > 0.001:
            return False
    return True


# =============================================================================
# CONFIGURATION
# =============================================================================
device     = torch.device(os.environ.get("GPU_DEVICE", "cuda") if torch.cuda.is_available() else "cpu")
path_image = os.environ.get("NIH_DATA_PATH")
if path_image is None:
    raise EnvironmentError(
        "❌ NIH_DATA_PATH environment variable is not set. "
        "Please set it to the root folder containing the NIH images."
    )

pathologies = [
    "Lung Opacity", "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Effusion", "Enlarged Cardiomediastinum", "Fracture",
    "Lung Lesion", "Pleural Other", "Pneumonia", "Pneumothorax",
]

target_pathologies = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Pneumonia", "Pneumothorax",
]

# ── Output directory — intentionally DIFFERENT from 3/5-client scripts ────────
RESULTS_DIR    = Path("evaluation_results_5clients")
RESULTS_DIR.mkdir(exist_ok=True)

NEW_MODELS_DIR = Path(os.environ.get("MODELS_DIR", "best_models_5clients"))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_existing_results():
    results_file = RESULTS_DIR / "all_results.json"
    if not results_file.exists():
        print("ℹ️  No existing 5-client results found — starting fresh")
        return [], set()
    try:
        with open(results_file) as f:
            existing = json.load(f)
        evaluated = {Path(r["model_path"]).name for r in existing}
        print(f"✅ Loaded {len(existing)} existing 5-client results")
        return existing, evaluated
    except Exception as e:
        print(f"⚠️  Error loading existing results: {e} — starting fresh")
        return [], set()


def get_models_to_evaluate(_existing_models: Set[str]) -> List[Path]:
    """Return all 5-client model files (re-evaluates everything)."""
    return sorted(NEW_MODELS_DIR.glob("best_global_model_nih_5clients__*.pt"))


def parse_model_filename(filename: str) -> List[Dict]:
    """
    Parse a filename of the form produced by server_nih_Nclients.py:
        best_global_model_nih_5clients__C0_p0.12_M100F0_flip0.0__C1_...pt
    """
    # Strip the prefix and suffix added by the 5-client server
    config_str  = filename.replace("best_global_model_nih_5clients__", "").replace(".pt", "")
    client_strs = config_str.split("__")
    configs = []
    for cs in client_strs:
        m = re.match(r"C(\d+)_p([\d.]+)_M(\d+)F(\d+)_flip([\d.]+)", cs)
        if m:
            configs.append({
                "client_id": int(m.group(1)),
                "portion":   float(m.group(2)),
                "gender": {
                    "Male":   int(m.group(3)) / 100,
                    "Female": int(m.group(4)) / 100,
                },
                "flip_frac": float(m.group(5)),
            })
    return configs


# =============================================================================
# DATA SETUP
# =============================================================================

def evaluate_model(
    model_path: str,
    config: List[Dict],
    val_df_nih,
    test_loader,
    female_paths_test: list,
    female_paths_val: list,
    common_transform,
) -> Dict:
    """Evaluate a single 5-client model and return full metrics dict."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {Path(model_path).name}")
    print(f"{'='*70}")

    model = create_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    from src.federated.utils import gender_partition, select_women_to_flip

    client_portions = [c["portion"]   for c in config]
    client_gender   = [c["gender"]    for c in config]
    client_flips    = [c["flip_frac"] for c in config]

    nih_clients_val_dict, _ = gender_partition(
        val_df_nih,
        client_gender_proportions=client_gender,
        portions=client_portions,
        gender_col="Sex",
        seed=42,
    )
    nih_clients_val = [nih_clients_val_dict[i] for i in sorted(nih_clients_val_dict.keys())]

    for i, df in enumerate(nih_clients_val):
        if df.shape[0] == 0:
            raise ValueError(f"Client {i} has an empty validation dataframe")

    women_to_flip = select_women_to_flip(
        nih_clients_val, target_pathologies, fracs=client_flips, seed=42
    )

    val_loaders = []
    for i, df in enumerate(nih_clients_val):
        ds = CheXpertAndNIH(df, path_image=path_image, transform=common_transform)
        val_loaders.append(DataLoader(
            ds, batch_size=32, shuffle=False,
            num_workers=20, pin_memory=True, persistent_workers=True, prefetch_factor=4,
        ))

    # ── Test set predictions ─────────────────────────────────────────────────
    test_out, test_tgt, test_sex = extract_predictions(
        test_loader, model, device, len(target_pathologies), female_paths_test, []
    )

    auc = compute_auc_sklearn(test_out, test_tgt, test_sex, len(target_pathologies))
    print_auc_results(auc, len(target_pathologies))

    # ── Validation predictions (for threshold selection) ─────────────────────
    val_outs, val_tgts = [], []
    for i, loader in enumerate(val_loaders):
        o, t, _ = extract_predictions(
            loader, model, device, len(target_pathologies),
            female_paths_val, women_to_flip.get(i, set())
        )
        val_outs.append(o)
        val_tgts.append(t)

    thresholds, f1_val, _ = find_best_thresholds(val_outs, val_tgts, len(target_pathologies))
    avg_f1_val = float(np.mean([np.mean(v) for v in f1_val.values()]))

    # ── Test metrics ─────────────────────────────────────────────────────────
    f1       = compute_f1_scores(test_out, test_tgt, test_sex, thresholds, len(target_pathologies))
    hard_tpr = compute_tpr_scores(test_out, test_tgt, test_sex, thresholds, len(target_pathologies))
    soft_tpr = compute_soft_tpr_sklearn(test_out, test_tgt, test_sex, len(target_pathologies))
    ba       = compute_balanced_accuracy_sklearn(
        test_out, test_tgt, test_sex, thresholds, len(target_pathologies)
    )

    print_f1_results(f1, len(target_pathologies), "Test NIH")
    print_tpr_results(hard_tpr, len(target_pathologies))
    print_soft_tpr_results(soft_tpr)
    print_balanced_accuracy_results(ba, len(target_pathologies))

    return {
        "model_path": model_path,
        "config":     config,
        "best_thresholds": thresholds,
        "validation": {
            "f1_scores": f1_val,
            "avg_f1":    avg_f1_val,
        },
        "nih_test": {
            "auc":               auc,
            "f1":                f1,
            "hard_tpr":          hard_tpr,
            "soft_tpr":          soft_tpr,
            "balanced_accuracy": ba,
        },
    }


# =============================================================================
# SUMMARY TABLES  — writes to RESULTS_DIR (evaluation_results_5clients/)
# =============================================================================

def add_config_columns(row: dict, model_name: str, config: List[Dict]) -> dict:
    """Populate per-client config columns.  Works for any number of clients."""
    row["model_name"]  = model_name
    row["num_clients"] = len(config)
    for i, c in enumerate(config):
        row[f"client_{i}_portion"]    = c["portion"]
        row[f"client_{i}_male_pct"]   = c["gender"]["Male"]
        row[f"client_{i}_female_pct"] = c["gender"]["Female"]
        row[f"client_{i}_flip_frac"]  = c["flip_frac"]
    return row


def create_summary_tables(all_results: List[Dict]):
    """Write one CSV per metric category, all inside RESULTS_DIR."""

    # ── 1. Validation ────────────────────────────────────────────────────────
    val_rows = []
    for r in all_results:
        row = {}
        add_config_columns(row, Path(r["model_path"]).stem, r["config"])
        row["val_avg_f1"] = r.get("validation", {}).get("avg_f1", np.nan)
        val_rows.append(row)
    df_val = pd.DataFrame(val_rows)
    out = RESULTS_DIR / "metrics_validation.csv"
    df_val.to_csv(out, index=False)
    print(f"✅ Saved validation metrics → {out}")

    # ── 2. AUC ───────────────────────────────────────────────────────────────
    auc_rows = []
    for r in all_results:
        row = {}
        add_config_columns(row, Path(r["model_path"]).stem, r["config"])
        auc = r.get("nih_test", {}).get("auc", {})
        row["nih_auc_macro"]        = auc.get("macro",        np.nan)
        row["nih_auc_macro_male"]   = auc.get("macro_male",   np.nan)
        row["nih_auc_macro_female"] = auc.get("macro_female", np.nan)
        row["nih_auc_micro"]        = auc.get("micro",        np.nan)
        row["nih_auc_micro_male"]   = auc.get("micro_male",   np.nan)
        row["nih_auc_micro_female"] = auc.get("micro_female", np.nan)
        auc_rows.append(row)
    df_auc = pd.DataFrame(auc_rows)
    out = RESULTS_DIR / "metrics_auc.csv"
    df_auc.to_csv(out, index=False)
    print(f"✅ Saved AUC metrics → {out}")

    # ── 3. F1 ────────────────────────────────────────────────────────────────
    f1_rows = []
    for r in all_results:
        row = {}
        add_config_columns(row, Path(r["model_path"]).stem, r["config"])
        f1 = r.get("nih_test", {}).get("f1", {})
        row["nih_f1_macro"]        = f1.get("avg",        np.nan)
        row["nih_f1_macro_male"]   = f1.get("avg_male",   np.nan)
        row["nih_f1_macro_female"] = f1.get("avg_female", np.nan)
        f1_rows.append(row)
    df_f1 = pd.DataFrame(f1_rows)
    out = RESULTS_DIR / "metrics_f1.csv"
    df_f1.to_csv(out, index=False)
    print(f"✅ Saved F1 metrics → {out}")

    # ── 4. Hard TPR ──────────────────────────────────────────────────────────
    hard_tpr_rows = []
    for r in all_results:
        row = {}
        add_config_columns(row, Path(r["model_path"]).stem, r["config"])
        hard_tpr = r.get("nih_test", {}).get("hard_tpr", {})
        per_path        = hard_tpr.get("per_pathology",        {})
        per_path_male   = hard_tpr.get("per_pathology_male",   {})
        per_path_female = hard_tpr.get("per_pathology_female", {})
        row["nih_hard_tpr_macro"]        = np.mean(list(per_path.values()))        if per_path        else np.nan
        row["nih_hard_tpr_macro_male"]   = np.mean(list(per_path_male.values()))   if per_path_male   else np.nan
        row["nih_hard_tpr_macro_female"] = np.mean(list(per_path_female.values())) if per_path_female else np.nan
        row["nih_hard_tpr_micro"]        = hard_tpr.get("cumulative",        np.nan)
        row["nih_hard_tpr_micro_male"]   = hard_tpr.get("cumulative_male",   np.nan)
        row["nih_hard_tpr_micro_female"] = hard_tpr.get("cumulative_female", np.nan)
        hard_tpr_rows.append(row)
    df_hard_tpr = pd.DataFrame(hard_tpr_rows)
    out = RESULTS_DIR / "metrics_hard_tpr.csv"
    df_hard_tpr.to_csv(out, index=False)
    print(f"✅ Saved Hard TPR metrics → {out}")

    # ── 5. Soft TPR ──────────────────────────────────────────────────────────
    soft_tpr_rows = []
    for r in all_results:
        row = {}
        add_config_columns(row, Path(r["model_path"]).stem, r["config"])
        soft_tpr = r.get("nih_test", {}).get("soft_tpr", {})
        row["nih_soft_tpr_micro"]        = soft_tpr.get("micro",        np.nan)
        row["nih_soft_tpr_micro_male"]   = soft_tpr.get("micro_male",   np.nan)
        row["nih_soft_tpr_micro_female"] = soft_tpr.get("micro_female", np.nan)
        row["nih_soft_tpr_gap"]          = soft_tpr.get("gap",          np.nan)
        row["nih_soft_tpr_min"]          = soft_tpr.get("min",          np.nan)
        soft_tpr_rows.append(row)
    df_soft_tpr = pd.DataFrame(soft_tpr_rows)
    out = RESULTS_DIR / "metrics_soft_tpr.csv"
    df_soft_tpr.to_csv(out, index=False)
    print(f"✅ Saved Soft TPR metrics → {out}")

    # ── 6. Balanced Accuracy ─────────────────────────────────────────────────
    ba_rows = []
    for r in all_results:
        row = {}
        add_config_columns(row, Path(r["model_path"]).stem, r["config"])
        ba = r.get("nih_test", {}).get("balanced_accuracy", {})
        row["nih_ba_macro"]        = ba.get("macro",        np.nan)
        row["nih_ba_macro_male"]   = ba.get("macro_male",   np.nan)
        row["nih_ba_macro_female"] = ba.get("macro_female", np.nan)
        row["nih_ba_micro"]        = ba.get("micro",        np.nan)
        row["nih_ba_micro_male"]   = ba.get("micro_male",   np.nan)
        row["nih_ba_micro_female"] = ba.get("micro_female", np.nan)
        ba_rows.append(row)
    df_ba = pd.DataFrame(ba_rows)
    out = RESULTS_DIR / "metrics_balanced_accuracy.csv"
    df_ba.to_csv(out, index=False)
    print(f"✅ Saved Balanced Accuracy metrics → {out}")

    # ── 7. Summary statistics ────────────────────────────────────────────────
    print("\n📊 Summary Statistics:")
    print(f"   Total models evaluated : {len(all_results)}")
    for label, df, cols in [
        ("AUC",               df_auc,      ["nih_auc_macro",      "nih_auc_macro_male",      "nih_auc_macro_female"]),
        ("F1",                df_f1,       ["nih_f1_macro",       "nih_f1_macro_male",       "nih_f1_macro_female"]),
        ("Hard TPR",          df_hard_tpr, ["nih_hard_tpr_macro", "nih_hard_tpr_macro_male", "nih_hard_tpr_macro_female"]),
        ("Soft TPR",          df_soft_tpr, ["nih_soft_tpr_micro", "nih_soft_tpr_micro_male", "nih_soft_tpr_micro_female",
                                            "nih_soft_tpr_gap",   "nih_soft_tpr_min"]),
        ("Balanced Accuracy", df_ba,       ["nih_ba_macro",       "nih_ba_macro_male",       "nih_ba_macro_female"]),
    ]:
        print(f"\n   {label}:")
        for col in cols:
            non_nan = df[col].notna().sum()
            print(f"      {col}: {non_nan}/{len(df)}")

    create_per_pathology_tables(all_results)


def create_per_pathology_tables(all_results: List[Dict]):
    for pathology_idx, pathology in enumerate(target_pathologies):
        rows = []
        for r in all_results:
            row = {}
            add_config_columns(row, Path(r["model_path"]).stem, r["config"])

            auc      = r.get("nih_test", {}).get("auc",      {})
            f1       = r.get("nih_test", {}).get("f1",       {})
            hard_tpr = r.get("nih_test", {}).get("hard_tpr", {})
            soft_tpr = r.get("nih_test", {}).get("soft_tpr", {})
            ba       = r.get("nih_test", {}).get("balanced_accuracy", {})

            for key, src in [
                ("nih_auc",        auc.get("per_pathology",        [])),
                ("nih_auc_male",   auc.get("per_pathology_male",   [])),
                ("nih_auc_female", auc.get("per_pathology_female", [])),
            ]:
                row[key] = (
                    float(src[pathology_idx])
                    if isinstance(src, (list, np.ndarray)) and len(src) > pathology_idx
                    else np.nan
                )

            for key, src in [
                ("nih_f1",              f1.get("per_pathology",              {})),
                ("nih_f1_male",         f1.get("per_pathology_male",         {})),
                ("nih_f1_female",       f1.get("per_pathology_female",       {})),
                ("nih_hard_tpr",        hard_tpr.get("per_pathology",        {})),
                ("nih_hard_tpr_male",   hard_tpr.get("per_pathology_male",   {})),
                ("nih_hard_tpr_female", hard_tpr.get("per_pathology_female", {})),
                ("nih_ba",              ba.get("per_pathology",              {})),
                ("nih_ba_male",         ba.get("per_pathology_male",         {})),
                ("nih_ba_female",       ba.get("per_pathology_female",       {})),
            ]:
                row[key] = float(src.get(pathology_idx, np.nan))

            # Soft TPR is micro-only (no per-pathology breakdown)
            row["nih_soft_tpr_micro"]        = soft_tpr.get("micro",        np.nan)
            row["nih_soft_tpr_micro_male"]   = soft_tpr.get("micro_male",   np.nan)
            row["nih_soft_tpr_micro_female"] = soft_tpr.get("micro_female", np.nan)
            row["nih_soft_tpr_gap"]          = soft_tpr.get("gap",          np.nan)
            row["nih_soft_tpr_min"]          = soft_tpr.get("min",          np.nan)

            rows.append(row)

        df  = pd.DataFrame(rows)
        out = RESULTS_DIR / f"metrics_pathology_{pathology.replace(' ', '_').lower()}.csv"
        df.to_csv(out, index=False)
        print(f"✅ Saved {pathology} metrics → {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("📊 SETTING UP DATA FOR 5-CLIENT EVALUATION")
    print("=" * 70)
    
    train_df_nih, val_df_nih, test_df_nih = import_nih_dfs(base_path)
    train_df_cxp, val_df_cxp, test_df_cxp = import_cxp_dfs(base_path)
    
    train_df_nih, _ = preprocess_nih_cxp(train_df_nih, train_df_cxp, pathologies, target_pathologies)
    val_df_nih,   _ = preprocess_nih_cxp(val_df_nih,   val_df_cxp,   pathologies, target_pathologies)
    test_df_nih,  _ = preprocess_nih_cxp(test_df_nih,  test_df_cxp,  pathologies, target_pathologies)
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    common_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_dataset = CheXpertAndNIH(
        test_df_nih, path_image=path_image, transform=common_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=20, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    
    female_paths_test = test_df_nih.loc[test_df_nih["Sex"] == "Female", "Path"].tolist()
    female_paths_val  = val_df_nih.loc[val_df_nih["Sex"] == "Female",   "Path"].tolist()
    
    print("✅ Data setup complete")
    
    
    # =============================================================================
    # MODEL EVALUATION
    # =============================================================================

    print("\n" + "=" * 70)
    print("🔍 5-CLIENT MODEL EVALUATION")
    print(f"   Output directory : {RESULTS_DIR}   (other-N CSVs are NOT touched)")
    print("=" * 70)

    existing_results, evaluated_models = load_existing_results()
    all_models = get_models_to_evaluate(evaluated_models)

    if not all_models:
        print(f"\n⚠️  No model files found in {NEW_MODELS_DIR}")
        return

    print(f"\n📋 Found {len(all_models)} model files — filtering to 5-client param_grid...")

    valid_models, skipped_models = [], []
    for model_path in all_models:
        try:
            cfg = parse_model_filename(model_path.name)
            if is_valid_config(cfg):          # enforces num_clients == 5
                valid_models.append((model_path, cfg))
            else:
                skipped_models.append(model_path)
        except Exception as e:
            print(f"   ⚠️  Could not parse {model_path.name}: {e}")
            skipped_models.append(model_path)

    print(f"\n📊 Filtering results:")
    print(f"   Total files found          : {len(all_models)}")
    print(f"   Valid 5-client models      : {len(valid_models)}")
    print(f"   Skipped (wrong num/config) : {len(skipped_models)}")

    if not valid_models:
        print("\n✅ No valid 5-client models to evaluate.")
        return

    print(f"\n📋 Models to evaluate:")
    for mp, _ in valid_models:
        print(f"   • {mp.name}")

    new_results, failed = [], []

    for i, (model_path, cfg) in enumerate(valid_models, 1):
        print(f"\n{'='*70}")
        print(f"MODEL {i}/{len(valid_models)}")
        print(f"{'='*70}")
        try:
            result = evaluate_model(
                str(model_path), cfg,
                val_df_nih=val_df_nih,
                test_loader=test_loader,
                female_paths_test=female_paths_test,
                female_paths_val=female_paths_val,
                common_transform=common_transform,
            )
            new_results.append(result)
            print(f"✅ {model_path.name}")
        except Exception as e:
            import traceback
            print(f"❌ Failed: {model_path.name}\n   {e}")
            traceback.print_exc()
            failed.append({"model": str(model_path), "error": str(e)})

    # ── Save results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("💾 SAVING RESULTS")
    print("=" * 70)

    # Backup old results if present
    if existing_results:
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = RESULTS_DIR / f"all_results_backup_{timestamp}.json"
        with open(backup_file, "w") as f:
            json.dump(existing_results, f, indent=2, default=str)
        print(f"   💾 Old results backed up → {backup_file.name}")

    results_file = RESULTS_DIR / "all_results.json"
    with open(results_file, "w") as f:
        json.dump(new_results, f, indent=2, default=str)
    print(f"✅ Saved {len(new_results)} results → {results_file}")

    if failed:
        fail_file = RESULTS_DIR / "failed_evaluations.json"
        with open(fail_file, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"⚠️  {len(failed)} failures saved → {fail_file}")

    if skipped_models:
        skip_file = RESULTS_DIR / "skipped_models.json"
        with open(skip_file, "w") as f:
            json.dump(
                [{"model": str(m), "reason": "Not a valid 5-client param_grid config"}
                 for m in skipped_models],
                f, indent=2,
            )
        print(f"ℹ️  {len(skipped_models)} skipped models logged → {skip_file}")

    # ── Summary tables ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 GENERATING SUMMARY TABLES")
    print("=" * 70)
    create_summary_tables(new_results)

    print("\n" + "=" * 70)
    print("🏁 5-CLIENT EVALUATION COMPLETE")
    print(f"   Successfully evaluated : {len(new_results)}")
    print(f"   Skipped                : {len(skipped_models)}")
    print(f"   Failed                 : {len(failed)}")
    print(f"   Results directory      : {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
