"""
train_meta_regressors.py
=========================
Builds a meta-dataset from the federation-advantage CSVs produced by the
comparison pipeline, engineers aggregated features, and trains regression
models that predict the benefit (or harm) of federating for client 0.

Dataset framing
---------------
Each row represents one federated experiment seen from client 0's perspective.
Peer clients (C1 … CN-1) are collapsed into three aggregate scalars so the
feature space stays fixed regardless of N.

Targets  (all sign-flipped so positive = improvement)
-------
  delta_auc             AUC increment (federated − standalone)
  delta_bal_acc         Balanced-accuracy increment
  delta_soft_tpr_gap    Soft TPR-gap decrease  (↑ = fairer)
  delta_min_soft_tpr    Soft min-TPR increment
  delta_hard_tpr_gap    Hard TPR-gap decrease  (↑ = fairer)
  delta_min_hard_tpr    Hard min-TPR increment

Features (10)
-------------
  c0_fitz14_frac_weighted    portion × fitz14_frac  (client 0 composition)
  c0_flip_frac_weighted      portion × (1−fitz14) × flip_frac  (client 0 bias)
  agg_fitz14_frac_weighted   Σ portion_i × fitz14_i  (peer composition)
  agg_flip_frac_weighted     Σ portion_i × (1−fitz14_i) × flip_i  (peer bias)
  fitz14_frac_weighted_diff  peer − client-0 composition
  flip_frac_weighted_diff    peer − client-0 bias
  log_size_ratio             log(agg_portion / c0_portion)
  num_clients                total clients in the federation
  c0_portion                 client-0 data fraction
  agg_portion                total peer data fraction

Usage
-----
    python meta_learning/train_meta_regressors.py

Environment variables
---------------------
    RESULTS_2C   path to results/comparison_dataset.csv
                 (default: <repo_root>/results/comparison_dataset.csv)
    RESULTS_3C   path to results/comparison_dataset_3c.csv
    RESULTS_5C   path to results/comparison_dataset_5c.csv
    PLOT_DIR     directory where figures and CSVs are saved
                 (default: <repo_root>/meta_learning/plots/)
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available — skipping. Install with: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available — skipping. Install with: pip install shap")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent

RESULTS_2C = Path(os.environ.get("RESULTS_2C", str(REPO_ROOT / "results" / "comparison_dataset.csv")))
RESULTS_3C = Path(os.environ.get("RESULTS_3C", str(REPO_ROOT / "results" / "comparison_dataset_3c.csv")))
RESULTS_5C = Path(os.environ.get("RESULTS_5C", str(REPO_ROOT / "results" / "comparison_dataset_5c.csv")))
PLOT_DIR   = Path(os.environ.get("PLOT_DIR",   str(_SCRIPT_DIR / "plots")))
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"]  = 100
plt.rcParams["savefig.dpi"] = 300


# =============================================================================
# 1. LOAD DATA
# =============================================================================

def _load_csv(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"⚠️  {label} not found: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  {label}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


print("\n" + "=" * 70)
print("📂 LOADING DATA")
print("=" * 70)

df_2c = _load_csv(RESULTS_2C, "2-client")
df_3c = _load_csv(RESULTS_3C, "3-client")
df_5c = _load_csv(RESULTS_5C, "5-client")

if all(df is None for df in (df_2c, df_3c, df_5c)):
    sys.exit("❌  No input CSVs found. Check RESULTS_2C / RESULTS_3C / RESULTS_5C.")


# =============================================================================
# 2. AGGREGATE PEER FEATURES
#    Collapse peer clients (C1 … CN-1) into three weighted scalars so the
#    feature space is the same for all federation sizes.
# =============================================================================

def _aggregate(df: pd.DataFrame, peer_ids: list[int], n_clients: int) -> pd.DataFrame:
    """
    Compute aggregate peer features for any list of peer client indices.

      agg_portion         = Σ portion_i
      agg_fitz14_weighted = Σ portion_i × fitz14_frac_i
      agg_flip_weighted   = Σ portion_i × (1 − fitz14_frac_i) × flip_frac_i
    """
    out = df.copy()
    portions = [out[f"c{i}_portion"]    for i in peer_ids]
    fitz14   = [out[f"c{i}_fitz14_frac"] for i in peer_ids]
    flips    = [out[f"c{i}_flip_frac"]   for i in peer_ids]

    out["agg_portion"]         = sum(portions)
    out["agg_fitz14_weighted"] = sum(portions[i] * fitz14[i] for i in range(len(peer_ids)))
    out["agg_flip_weighted"]   = sum(
        portions[i] * (1.0 - fitz14[i]) * flips[i] for i in range(len(peer_ids))
    )
    out["num_clients"] = n_clients
    return out


print("\n" + "=" * 70)
print("⚙️  COMPUTING AGGREGATE PEER FEATURES")
print("=" * 70)

if df_2c is not None:
    df_2c = _aggregate(df_2c, peer_ids=[1],       n_clients=2)
    print(f"  2-client agg_portion range: "
          f"[{df_2c['agg_portion'].min():.3f}, {df_2c['agg_portion'].max():.3f}]")

if df_3c is not None:
    df_3c = _aggregate(df_3c, peer_ids=[1, 2],    n_clients=3)
    print(f"  3-client agg_portion range: "
          f"[{df_3c['agg_portion'].min():.3f}, {df_3c['agg_portion'].max():.3f}]")

if df_5c is not None:
    df_5c = _aggregate(df_5c, peer_ids=[1, 2, 3, 4], n_clients=5)
    print(f"  5-client agg_portion range: "
          f"[{df_5c['agg_portion'].min():.3f}, {df_5c['agg_portion'].max():.3f}]")


# =============================================================================
# 3. DATA AUGMENTATION
#    The param-grid enforces flip_frac=0 when fitz14_frac==1.0 (no light-skin
#    patients to flip).  We synthetically add rows with non-zero flip fracs to
#    prevent the model from learning a spurious floor effect.
# =============================================================================

_EXTRA_FLIPS = [0.15, 0.30, 0.45]


def _augment_fitz14_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate rows where c0_fitz14_frac == 1.0 and c0_flip_frac == 0.0
    for each non-zero flip value.
    """
    mask = (df["c0_fitz14_frac"] == 1.0) & (df["c0_flip_frac"] == 0.0)
    base = df[mask].copy()
    extra = []
    for flip_val in _EXTRA_FLIPS:
        rows = base.copy()
        rows["c0_flip_frac"] = flip_val
        extra.append(rows)
    if not extra:
        print("  ℹ️  No fitz14=1 / flip=0 rows found — nothing to augment.")
        return df
    return pd.concat([df] + extra, ignore_index=True)


print("\n" + "=" * 70)
print("🔧 DATA AUGMENTATION")
print("=" * 70)

_sizes_before = {}
for label, df_n in (("2-client", df_2c), ("3-client", df_3c), ("5-client", df_5c)):
    if df_n is not None:
        _sizes_before[label] = len(df_n)

if df_2c is not None:
    df_2c = _augment_fitz14_rows(df_2c)
    print(f"  2-client: {_sizes_before['2-client']} → {len(df_2c)} rows  "
          f"(+{len(df_2c) - _sizes_before['2-client']} synthetic)")

if df_3c is not None:
    df_3c = _augment_fitz14_rows(df_3c)
    print(f"  3-client: {_sizes_before['3-client']} → {len(df_3c)} rows  "
          f"(+{len(df_3c) - _sizes_before['3-client']} synthetic)")

if df_5c is not None:
    df_5c = _augment_fitz14_rows(df_5c)
    print(f"  5-client: {_sizes_before['5-client']} → {len(df_5c)} rows  "
          f"(+{len(df_5c) - _sizes_before['5-client']} synthetic)")


# =============================================================================
# 4. MERGE + SIGN-FLIP GAP TARGETS
#    Gap targets are originally negative-better; we flip them so that
#    positive = improvement across all targets.
# =============================================================================

TARGET_COLS = [
    "delta_auc",
    "delta_bal_acc",
    "delta_soft_tpr_gap",
    "delta_min_soft_tpr",
    "delta_hard_tpr_gap",
    "delta_min_hard_tpr",
]
GAP_TARGETS = ["delta_auc_gap", "delta_soft_tpr_gap", "delta_hard_tpr_gap"]

C0_COLS  = ["c0_portion", "c0_fitz14_frac", "c0_flip_frac"]
AGG_COLS = ["agg_portion", "agg_fitz14_weighted", "agg_flip_weighted"]
META_COLS = ["num_clients"]

KEEP_COLS = C0_COLS + AGG_COLS + META_COLS + TARGET_COLS

print("\n" + "=" * 70)
print("🔗 MERGING AND SIGN-FLIPPING GAP TARGETS")
print("=" * 70)

parts = []
for label, df_n in (("2-client", df_2c), ("3-client", df_3c), ("5-client", df_5c)):
    if df_n is None:
        continue
    # Sign-flip gap targets so positive = improvement
    df_n = df_n.copy()
    for col in GAP_TARGETS:
        if col in df_n.columns:
            df_n[col] = -df_n[col]
    # Only keep columns that exist in this dataset
    existing = [c for c in KEEP_COLS if c in df_n.columns]
    parts.append(df_n[existing])

df_merged = pd.concat(parts, ignore_index=True, sort=False)

# Only keep targets that exist in the merged data
TARGET_COLS = [c for c in TARGET_COLS if c in df_merged.columns]

print(f"  Merged: {len(df_merged)} rows  "
      f"(2c={len(df_2c) if df_2c is not None else 0}, "
      f"3c={len(df_3c) if df_3c is not None else 0}, "
      f"5c={len(df_5c) if df_5c is not None else 0})")
print(f"\n  Client distribution:")
print(df_merged["num_clients"].value_counts().sort_index().to_string())


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 70)
print("🛠️  FEATURE ENGINEERING")
print("=" * 70)

_src = df_merged[C0_COLS + AGG_COLS + META_COLS].copy()

X = pd.DataFrame(index=df_merged.index)

# Client-0 weighted features
X["c0_fitz14_frac_weighted"] = _src["c0_portion"] * _src["c0_fitz14_frac"]
X["c0_flip_frac_weighted"]   = (
    _src["c0_portion"] * (1.0 - _src["c0_fitz14_frac"]) * _src["c0_flip_frac"]
)

# Aggregate peer weighted features
X["agg_fitz14_frac_weighted"] = _src["agg_fitz14_weighted"]
X["agg_flip_frac_weighted"]   = _src["agg_flip_weighted"]

# Difference features (aggregate − client-0)
X["fitz14_frac_weighted_diff"] = X["agg_fitz14_frac_weighted"] - X["c0_fitz14_frac_weighted"]
X["flip_frac_weighted_diff"]   = X["agg_flip_frac_weighted"]   - X["c0_flip_frac_weighted"]

# Size features
X["log_size_ratio"] = np.log(_src["agg_portion"] / _src["c0_portion"])
X["num_clients"]    = _src["num_clients"].astype(float)
X["c0_portion"]     = _src["c0_portion"]
X["agg_portion"]    = _src["agg_portion"]

FEATURE_COLS = list(X.columns)

FEATURE_DISPLAY_NAMES = {
    "c0_fitz14_frac_weighted":   r"$\tilde{\mathrm{comp}}_c$",
    "c0_flip_frac_weighted":     r"$\tilde{\mathrm{bias}}_c$",
    "agg_fitz14_frac_weighted":  r"$\mathrm{comp}_f$",
    "agg_flip_frac_weighted":    r"$\mathrm{bias}_f$",
    "fitz14_frac_weighted_diff": r"$\delta_{\mathrm{comp}}$",
    "flip_frac_weighted_diff":   r"$\delta_{\mathrm{bias}}$",
    "log_size_ratio":            r"$r_{\mathrm{size}}$",
    "num_clients":               r"$n_{\mathcal{F}}$",
    "c0_portion":                r"$\mathrm{size}_c$",
    "agg_portion":               r"$\mathrm{size}_f$",
}

# Validity mask
valid_mask = ~(X.isna().any(axis=1) | np.isinf(X).any(axis=1))
X_valid  = X[valid_mask].reset_index(drop=True)
y_dict   = {col: df_merged.loc[valid_mask, col].values for col in TARGET_COLS}

print(f"  Feature matrix: {X_valid.shape[0]} valid rows × {len(FEATURE_COLS)} features")
for f in FEATURE_COLS:
    print(f"    - {f}  ({FEATURE_DISPLAY_NAMES.get(f, f)})")
print(f"\n  Targets ({len(TARGET_COLS)}):")
for t in TARGET_COLS:
    print(f"    - {t}")


# =============================================================================
# 6. MODEL DEFINITIONS
# =============================================================================

CV      = KFold(n_splits=5, shuffle=True, random_state=42)
N_ITER  = 30
SCORING = "r2"


def build_search_spaces() -> dict:
    spaces = {
        "Ridge": (
            Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": np.logspace(-3, 3, 100).tolist()},
        ),
        "Lasso": (
            Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=10_000))]),
            {"model__alpha": np.logspace(-4, 1, 100).tolist()},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators":      [100, 200, 300, 500],
                "max_depth":         [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf":  [1, 2, 4],
                "max_features":      ["sqrt", "log2", 0.5, 0.8],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators":      [100, 200, 300],
                "learning_rate":     np.logspace(-2, 0, 30).tolist(),
                "max_depth":         [2, 3, 4, 5],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf":  [1, 2, 4],
                "subsample":         [0.7, 0.8, 0.9, 1.0],
            },
        ),
    }
    if XGBOOST_AVAILABLE:
        spaces["XGBoost"] = (
            XGBRegressor(
                random_state=42, n_jobs=-1,
                eval_metric="rmse", verbosity=0, device="cpu",
            ),
            {
                "n_estimators":     [100, 200, 300, 500],
                "learning_rate":    np.logspace(-2, 0, 30).tolist(),
                "max_depth":        [2, 3, 4, 5, 6],
                "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha":        np.logspace(-3, 1, 30).tolist(),
                "reg_lambda":       np.logspace(-3, 1, 30).tolist(),
            },
        )
    return spaces


# =============================================================================
# 7. TRAINING
# =============================================================================

print("\n" + "=" * 70)
print("🏋️  TRAINING")
print("=" * 70)

results: dict[str, dict] = {}

for target_name in TARGET_COLS:
    print(f"\n{'='*70}\n  Target: {target_name}  ({len(y_dict[target_name])} samples)\n{'='*70}")
    y = y_dict[target_name]
    results[target_name] = {}

    for model_name, (estimator, param_dist) in build_search_spaces().items():
        search = RandomizedSearchCV(
            estimator,
            param_distributions=param_dist,
            n_iter=N_ITER,
            scoring=SCORING,
            cv=CV,
            random_state=42,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_valid.values, y)
        best_est = search.best_estimator_

        cv_r2   = cross_val_score(best_est, X_valid.values, y, cv=CV, scoring="r2")
        cv_rmse = cross_val_score(best_est, X_valid.values, y, cv=CV,
                                  scoring="neg_root_mean_squared_error")
        cv_mae  = cross_val_score(best_est, X_valid.values, y, cv=CV,
                                  scoring="neg_mean_absolute_error")

        results[target_name][model_name] = {
            "model":   best_est,
            "y":       y,
            "metrics": {
                "cv_r2_mean":   cv_r2.mean(),
                "cv_r2_std":    cv_r2.std(),
                "cv_rmse_mean": (-cv_rmse).mean(),
                "cv_rmse_std":  (-cv_rmse).std(),
                "cv_mae_mean":  (-cv_mae).mean(),
                "cv_mae_std":   (-cv_mae).std(),
                "best_params":  search.best_params_,
            },
        }
        print(
            f"  {model_name:20s}  "
            f"CV R²: {cv_r2.mean():+.4f} ± {cv_r2.std():.4f}  "
            f"RMSE: {(-cv_rmse).mean():.6f} ± {(-cv_rmse).std():.6f}"
        )

print("\n✅ Training complete!")


# =============================================================================
# 8. HELPERS
# =============================================================================

_NICE_NAMES = {
    "delta_auc":           "AUC Δ",
    "delta_bal_acc":       "Balanced Accuracy Δ",
    "delta_soft_tpr_gap":  "Soft TPR Gap Δ (↑ = fairer)",
    "delta_min_soft_tpr":  "Soft Min-TPR Δ",
    "delta_hard_tpr_gap":  "Hard TPR Gap Δ (↑ = fairer)",
    "delta_min_hard_tpr":  "Hard Min-TPR Δ",
}


def nice_name(col: str) -> str:
    return _NICE_NAMES.get(col, col)


def best_model_name(target_name: str) -> str:
    return max(
        results[target_name],
        key=lambda m: results[target_name][m]["metrics"]["cv_r2_mean"],
    )


# =============================================================================
# 9. PLOTS
# =============================================================================

def _plot_predictions(target_names: list, filename: str) -> None:
    """Predicted vs Actual scatter plots for a list of targets."""
    n_targets = len(target_names)
    if n_targets == 0:
        return
    ncols = 2
    nrows = (n_targets + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 5))
    axes = np.array(axes).flatten()

    for idx, target_name in enumerate(target_names):
        ax       = axes[idx]
        bm       = best_model_name(target_name)
        best_est = results[target_name][bm]["model"]
        y_true   = results[target_name][bm]["y"]

        y_pred = cross_val_predict(best_est, X_valid.values, y_true, cv=CV, n_jobs=-1)
        r2   = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        ax.scatter(y_true, y_pred, alpha=0.6, s=40)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect prediction")

        ax.set_xlabel("Actual", fontsize=14)
        ax.set_ylabel("Predicted", fontsize=14)
        ax.set_title(
            f"{nice_name(target_name)} — {bm}\nR²={r2:.3f}, RMSE={rmse:.4f}",
            fontsize=15, fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=12)

    for ax in axes[n_targets:]:
        ax.set_visible(False)

    plt.tight_layout()
    out = PLOT_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved → {out}")


print("\n" + "=" * 70)
print("📊 GENERATING PLOTS")
print("=" * 70)

_plot_predictions(TARGET_COLS, "predictions_vs_actual.png")


# =============================================================================
# 10. SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)

summary_rows = []
for target_name in TARGET_COLS:
    for model_name, entry in results[target_name].items():
        m = entry["metrics"]
        summary_rows.append({
            "target":       target_name,
            "model":        model_name,
            "cv_r2_mean":   m["cv_r2_mean"],
            "cv_r2_std":    m["cv_r2_std"],
            "cv_rmse_mean": m["cv_rmse_mean"],
            "cv_rmse_std":  m["cv_rmse_std"],
            "cv_mae_mean":  m["cv_mae_mean"],
            "cv_mae_std":   m["cv_mae_std"],
        })

df_summary = pd.DataFrame(summary_rows)
out_csv = PLOT_DIR / "model_summary.csv"
df_summary.to_csv(out_csv, index=False)
print(f"✅ Summary saved → {out_csv}")

print("\nBest model per target:")
for target_name in TARGET_COLS:
    bm = best_model_name(target_name)
    r2 = results[target_name][bm]["metrics"]["cv_r2_mean"]
    print(f"  {target_name:<25s}  {bm:<20s}  CV R²={r2:+.4f}")
