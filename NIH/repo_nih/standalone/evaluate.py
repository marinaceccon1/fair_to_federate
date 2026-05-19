"""
Incremental evaluation script for standalone models that:
1. Evaluates only new models not yet in results
2. Merges metrics with previously saved results
3. Prevents duplicate evaluations
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
from contextlib import redirect_stdout, redirect_stderr

# ---------------------------------------------------------------------------
# Repo root: three levels up from this file (NIH/repo/standalone/evaluate.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from torchvision import transforms
from torch.utils.data import DataLoader
from src.data.utils import import_nih_dfs, import_cxp_dfs
from src.federated.utils import (
    preprocess_nih_cxp,
    gender_partition,
    select_women_to_flip,
    extract_predictions,
    compute_f1_scores,
    compute_tpr_scores,
    find_best_thresholds,
    print_auc_results,
    print_f1_results,
    print_tpr_results,
    compute_auc_sklearn,
    compute_balanced_accuracy_sklearn
)
from src.data.dataset import CheXpertAndNIH
from src.models.model import create_model

# =============================================================================
# CONFIGURATION
# =============================================================================
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path_image = os.environ.get("NIH_DATA_PATH")
if path_image is None:
    raise EnvironmentError(
        "❌ NIH_DATA_PATH environment variable is not set. "
        "Please set it to the root folder containing the NIH ChestX-ray14 images, e.g.:\n"
        "  export NIH_DATA_PATH=/path/to/your/images"
    )

pathologies = [
    'Lung Opacity', 'Atelectasis', 'Cardiomegaly', 'Consolidation',
    'Edema', 'Effusion', 'Enlarged Cardiomediastinum', 'Fracture',
    'Lung Lesion', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
]

target_pathologies = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Pneumonia', 'Pneumothorax',
]

MODELS_DIR   = REPO_ROOT / "repo_nih" / "single_models"
RESULTS_DIR  = REPO_ROOT / "repo_nih" / "evaluation_results_standalone"
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# SOFT TPR
# =============================================================================
def compute_soft_tpr_sklearn(outputs, targets, sex_labels, num_classes):
    """
    Soft TPR = mean predicted probability over all (sample, pathology) pairs
    where the true label is 1 and not NaN.

    Returns a dict with keys: micro, micro_male, micro_female, gap, min.
    """
    outputs    = outputs.T
    targets    = targets.T
    sex_labels = np.asarray(sex_labels)

    flat_targets = targets.flatten()
    flat_outputs = outputs.flatten()
    flat_sex     = np.repeat(sex_labels, num_classes)

    pos_mask        = (flat_targets == 1) & ~np.isnan(flat_targets)
    pos_mask_male   = pos_mask & (flat_sex == 1.0)
    pos_mask_female = pos_mask & (flat_sex == 0.0)

    micro        = float(np.mean(flat_outputs[pos_mask]))        if pos_mask.sum()        > 0 else np.nan
    micro_male   = float(np.mean(flat_outputs[pos_mask_male]))   if pos_mask_male.sum()   > 0 else np.nan
    micro_female = float(np.mean(flat_outputs[pos_mask_female])) if pos_mask_female.sum() > 0 else np.nan
    gap          = (micro_male - micro_female) if not (np.isnan(micro_male) or np.isnan(micro_female)) else np.nan
    min_soft_tpr = float(np.nanmin([micro_male, micro_female])) \
                   if not (np.isnan(micro_male) and np.isnan(micro_female)) else np.nan

    return {
        "micro":        micro,
        "micro_male":   micro_male,
        "micro_female": micro_female,
        "gap":          gap,
        "min":          min_soft_tpr,
    }


def print_soft_tpr_results(r):
    print("\n**SOFT TPR SCORES (threshold-free, micro)**")
    print(f"  Micro Soft TPR:        {r['micro']:.3f}")
    print(f"  Micro Male Soft TPR:   {r['micro_male']:.3f}")
    print(f"  Micro Female Soft TPR: {r['micro_female']:.3f}")
    print(f"  Gap (Male - Female):   {r['gap']:.3f}")
    print(f"  Min(Male, Female):     {r['min']:.3f}")


# =============================================================================
# HELPERS
# =============================================================================
def load_existing_results() -> tuple[List[Dict], Set[str]]:
    results_file = RESULTS_DIR / "full_results.json"
    if not results_file.exists():
        print("ℹ️  No existing results found - starting fresh")
        return [], set()
    try:
        with open(results_file) as f:
            existing = json.load(f)
        evaluated = {Path(r["model_path"]).name for r in existing}
        print(f"✅ Loaded {len(existing)} existing results ({len(evaluated)} models)")
        return existing, evaluated
    except Exception as e:
        print(f"⚠️  Error loading existing results: {e} — starting fresh")
        return [], set()


def get_new_models(existing_models: Set[str]) -> List[Path]:
    all_models = list(MODELS_DIR.glob("best_model__*.pt"))
    return sorted(m for m in all_models if m.name not in existing_models)


def parse_filename(filename: str) -> Dict:
    """Parse a standalone model filename to extract its configuration."""
    config_str = filename.replace("best_model__", "").replace(".pt", "")
    match = re.match(r"p([\d.]+)_M(\d+)F(\d+)_flip([\d.]+)", config_str)
    if not match:
        raise ValueError(f"Could not parse filename: {filename}")

    return {
        "portion":   float(match.group(1)),
        "gender":    {"Male": int(match.group(2)) / 100, "Female": int(match.group(3)) / 100},
        "flip_frac": float(match.group(4)),
    }


# =============================================================================
# DATA SETUP
# =============================================================================
def setup_data():
    print("\n" + "=" * 70)
    print("📊 SETTING UP DATA FOR EVALUATION")
    print("=" * 70)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize
    ])

    train_df_nih, val_df_nih, test_df_nih = import_nih_dfs(str(REPO_ROOT))
    train_df_cxp, val_df_cxp, test_df_cxp = import_cxp_dfs(str(REPO_ROOT))

    train_df_nih_mod, _ = preprocess_nih_cxp(train_df_nih, train_df_cxp, pathologies, target_pathologies)
    val_df_nih_mod,   _ = preprocess_nih_cxp(val_df_nih,   val_df_cxp,   pathologies, target_pathologies)
    test_df_nih_mod,  _ = preprocess_nih_cxp(test_df_nih,  test_df_cxp,  pathologies, target_pathologies)

    test_dataloader = DataLoader(
        CheXpertAndNIH(test_df_nih_mod, path_image=path_image, transform=transform),
        batch_size=32, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=4
    )

    female_paths_test = test_df_nih_mod.loc[test_df_nih_mod["Sex"] == "Female", "Path"].tolist()
    female_paths_val  = val_df_nih_mod.loc[val_df_nih_mod["Sex"] == "Female",   "Path"].tolist()

    print("✅ Data setup complete")
    return test_dataloader, female_paths_test, val_df_nih_mod, female_paths_val, transform


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_model(model_path: str, config: Dict,
                   test_dataloader, female_paths_test,
                   val_df_nih_mod, female_paths_val, val_transform) -> Dict:
    print(f"\n{'=' * 70}")
    print(f"Evaluating: {Path(model_path).name}")
    print(f"{'=' * 70}")
    print(f"Config — Portion: {config['portion']}, "
          f"Gender: F={config['gender']['Female']:.2f}, "
          f"Flip: {config['flip_frac']}")

    model = create_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = {"model_path": model_path, "config": config}

    dummy_config = {"portion": 0.03, "gender": {"Male": 0.5, "Female": 0.5}, "flip_frac": 0.0}
    client_configs = [config, dummy_config]

    nih_clients_val, _ = gender_partition(
        val_df_nih_mod,
        [c["gender"]  for c in client_configs],
        [c["portion"] for c in client_configs],
        gender_col="Sex", seed=42
    )

    women_to_flip_val = select_women_to_flip(
        nih_clients_val, target_pathologies,
        fracs=[config["flip_frac"], dummy_config["flip_frac"]], seed=42
    )

    val_dataloader = DataLoader(
        CheXpertAndNIH(nih_clients_val[0], path_image=path_image, transform=val_transform),
        batch_size=32, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=4
    )

    # ── Test set ──────────────────────────────────────────────────────────────
    print("\n📊 Evaluating on NIH test set...")
    test_outputs, test_targets, sex_labels_test = extract_predictions(
        test_dataloader, model, device, len(target_pathologies),
        female_paths_test, set(), split_name="Test NIH", print_stats=True
    )

    auc_results = compute_auc_sklearn(test_outputs, test_targets, sex_labels_test, len(target_pathologies))
    print_auc_results(auc_results, len(target_pathologies))

    # ── Validation set (for threshold selection) ──────────────────────────────
    print("\n📊 Finding best thresholds on validation set...")
    val_outputs, val_targets, _ = extract_predictions(
        val_dataloader, model, device, len(target_pathologies),
        female_paths_val, women_to_flip_val.get(0, set()),
        split_name="Val NIH", print_stats=True
    )
    best_thresholds, best_f1_scores_val, _ = find_best_thresholds(
        [val_outputs], [val_targets], len(target_pathologies)
    )
    avg_f1_val = np.mean([np.mean(v) for v in best_f1_scores_val.values()])
    print(f"Average F1 score (validation): {avg_f1_val:.4f}")

    # ── Test metrics ──────────────────────────────────────────────────────────
    print("\n📊 Computing F1, TPR, Soft TPR, Balanced Accuracy on NIH test set...")
    test_f1  = compute_f1_scores(test_outputs, test_targets, sex_labels_test, best_thresholds, len(target_pathologies))
    test_tpr = compute_tpr_scores(test_outputs, test_targets, sex_labels_test, best_thresholds, len(target_pathologies))
    soft_tpr = compute_soft_tpr_sklearn(test_outputs, test_targets, sex_labels_test, len(target_pathologies))
    ba       = compute_balanced_accuracy_sklearn(test_outputs, test_targets, sex_labels_test, best_thresholds, len(target_pathologies))

    print_f1_results(test_f1, len(target_pathologies), "Test NIH")
    print_tpr_results(test_tpr, len(target_pathologies))
    print_soft_tpr_results(soft_tpr)
    print(f"Balanced Accuracy | macro={ba['macro']:.4f}, micro={ba['micro']:.4f}, "
          f"male={ba['micro_male']:.4f}, female={ba['micro_female']:.4f}")

    results.update({
        "best_thresholds": best_thresholds,
        "validation": {"f1_scores": best_f1_scores_val, "avg_f1": float(avg_f1_val)},
        "nih_test": {
            "auc":               auc_results,
            "f1":                test_f1,
            "hard_tpr":          test_tpr,
            "soft_tpr":          soft_tpr,
            "balanced_accuracy": ba,
        }
    })
    return results


# =============================================================================
# SUMMARY CSV
# =============================================================================
def create_summary_csv(all_results: List[Dict]):
    print("\n💾 Creating summary CSV...")
    rows = []
    for r in all_results:
        c = r['config']
        rows.append({
            'model_path':            r['model_path'],
            'portion':               c['portion'],
            'male_pct':              c['gender']['Male'],
            'female_pct':            c['gender']['Female'],
            'flip_frac':             c['flip_frac'],
            'val_avg_f1':            r['validation']['avg_f1'],
            'nih_auc_macro':         r['nih_test']['auc']['avg'],
            'nih_auc_micro':         r['nih_test']['auc']['cumulative'],
            'nih_auc_micro_male':    r['nih_test']['auc']['cumulative_male'],
            'nih_auc_micro_female':  r['nih_test']['auc']['cumulative_female'],
            'nih_f1_macro':          r['nih_test']['f1']['avg'],
            'nih_f1_micro_male':     r['nih_test']['f1']['avg_male'],
            'nih_f1_micro_female':   r['nih_test']['f1']['avg_female'],
            'nih_tpr_micro':         r['nih_test']['hard_tpr']['cumulative'],
            'nih_tpr_micro_male':    r['nih_test']['hard_tpr']['cumulative_male'],
            'nih_tpr_micro_female':  r['nih_test']['hard_tpr']['cumulative_female'],
            'nih_soft_tpr_micro':        r['nih_test']['soft_tpr']['micro'],
            'nih_soft_tpr_micro_male':   r['nih_test']['soft_tpr']['micro_male'],
            'nih_soft_tpr_micro_female': r['nih_test']['soft_tpr']['micro_female'],
            'nih_soft_tpr_gap':          r['nih_test']['soft_tpr']['gap'],
            'nih_soft_tpr_min':          r['nih_test']['soft_tpr']['min'],
            'nih_bacc_macro':        r['nih_test']['balanced_accuracy']['macro'],
            'nih_bacc_micro':        r['nih_test']['balanced_accuracy']['micro'],
            'nih_bacc_micro_male':   r['nih_test']['balanced_accuracy']['micro_male'],
            'nih_bacc_micro_female': r['nih_test']['balanced_accuracy']['micro_female'],
        })

    csv_path = RESULTS_DIR / "summary_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"✅ Saved CSV to: {csv_path}")


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, dict):        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):        return [convert_to_serializable(i) for i in obj]
    return obj


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("🔍 INCREMENTAL STANDALONE MODEL EVALUATION")
    print("=" * 70)

    existing_results, evaluated_models = load_existing_results()
    new_models = get_new_models(evaluated_models)

    if not new_models:
        print(f"\n✅ No new models found in {MODELS_DIR}")
        if existing_results:
            create_summary_csv(existing_results)
        return

    print(f"\n📋 Found {len(new_models)} new models to evaluate:")
    for m in new_models:
        print(f"   • {m.name}")

    test_dataloader, female_paths_test, val_df_nih_mod, female_paths_val, val_transform = setup_data()

    new_results   = []
    failed_models = []

    for i, model_path in enumerate(new_models, 1):
        print(f"\n{'=' * 70}")
        print(f"MODEL {i}/{len(new_models)}")
        print(f"{'=' * 70}")
        try:
            config  = parse_filename(model_path.name)
            results = evaluate_model(
                str(model_path), config,
                test_dataloader, female_paths_test,
                val_df_nih_mod, female_paths_val, val_transform
            )
            new_results.append(results)
            print(f"✅ Successfully evaluated {model_path.name}")
        except Exception as e:
            import traceback
            print(f"❌ Failed: {model_path.name}\n   Error: {e}")
            traceback.print_exc()
            failed_models.append({"model": str(model_path), "error": str(e)})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"new_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(convert_to_serializable(new_results), f, indent=2)
    print(f"\n✅ Saved new results to: {json_path}")

    create_summary_csv(existing_results + new_results)


if __name__ == "__main__":
    log_path = RESULTS_DIR / "evaluation_log.txt"
    with open(log_path, "w") as f, redirect_stdout(f), redirect_stderr(f):
        main()
