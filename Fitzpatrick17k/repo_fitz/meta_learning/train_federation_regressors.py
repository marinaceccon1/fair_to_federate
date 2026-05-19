"""
train_federation_regressors.py
================================
Trains regression models that predict the *gain* for the existing federation
when a new client (C2) joins.

Dataset framing
---------------
The question answered here is:
  "Given an existing 2-client federation (C0+C1), how much does the
   federation gain when a new client C2 joins?"

Two source datasets are merged into a unified feature space:

  3-client rows  (results/gain_dataset.csv)
      C0 + C1  = existing federation
      C2       = new client joining
      targets  = gain_*

  2-client rows  (results/comparison_dataset.csv)
      C0       = existing federation (singleton)
      C1 → C2  = new client joining
      targets  = delta_* (renamed to gain_* after loading)
      num_clients = 1  (existing federation is a singleton)

Targets  (gap targets sign-flipped so positive = improvement)
-------
  gain_auc            AUC gain for the federation
  gain_bal_acc        Balanced-accuracy gain
  gain_soft_tpr_gap   Soft TPR-gap decrease  (↑ = fairer)
  gain_min_soft_tpr   Soft min-TPR gain
  gain_hard_tpr_gap   Hard TPR-gap decrease  (↑ = fairer)
  gain_min_hard_tpr   Hard min-TPR gain

Features (8)
------------
  agg_portion              total data fraction of existing federation
  c2_portion               new client data fraction
  log_size_ratio           log(agg_portion / c2_portion)
  agg_fitz14_frac_weighted Σ portion_i × fitz14_i  (existing federation)
  c2_fitz14_frac_weighted  c2_portion × c2_fitz14_frac
  agg_flip_frac_weighted   Σ portion_i × (1−fitz14_i) × flip_i
  c2_flip_frac_weighted    c2_portion × (1−fitz14_c2) × flip_c2
  fitz14_frac_weighted_diff  agg − c2 composition
  flip_frac_weighted_diff    agg − c2 bias
  num_clients              size of the existing federation (1 or 2)

Usage
-----
    python meta_learning/train_federation_regressors.py

Environment variables
---------------------
    RESULTS_GAIN   path to results/gain_dataset.csv
                   (default: <repo_root>/results/gain_dataset.csv)
    RESULTS_2C     path to results/comparison_dataset.csv
                   (default: <repo_root>/results/comparison_dataset.csv)
    PLOT_DIR       directory where figures and CSVs are saved
                   (default: <repo_root>/meta_learning/plots_federation/)
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

RESULTS_GAIN = Path(os.environ.get("RESULTS_GAIN", str(REPO_ROOT / "results" / "gain_dataset.csv")))
RESULTS_2C   = Path(os.environ.get("RESULTS_2C",   str(REPO_ROOT / "results" / "comparison_dataset.csv")))
PLOT_DIR     = Path(os.environ.get("PLOT_DIR",     str(_SCRIPT_DIR / "plots_federation")))
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

df_gain = _load_csv(RESULTS_GAIN, "3-client gain dataset")
df_2c   = _load_csv(RESULTS_2C,   "2-client comparison dataset")

if df_gain is None and df_2c is None:
    sys.exit("❌  No input CSVs found. Check RESULTS_GAIN / RESULTS_2C.")


# =============================================================================
# 2. DATA AUGMENTATION
#    The param-grid enforces flip_frac=0 when fitz14_frac==1.0.  We synthesise
#    copies with non-zero flip fracs to prevent a spurious floor effect.
# =============================================================================

_EXTRA_FLIPS = [0.15, 0.30, 0.45]


def _augment_fitz14_rows(df: pd.DataFrame, client_prefixes: list[str]) -> pd.DataFrame:
    """
    For each prefix in `client_prefixes`, duplicate rows where
    {prefix}_fitz14_frac == 1.0 and {prefix}_flip_frac == 0.0
    for each non-zero flip value.
    """
    result = df.copy()
    for prefix in client_prefixes:
        frac_col = f"{prefix}_fitz14_frac"
        flip_col = f"{prefix}_flip_frac"
        if frac_col not in result.columns or flip_col not in result.columns:
            continue
        before = len(result)
        mask   = (result[frac_col] == 1.0) & (result[flip_col] == 0.0)
        base   = result[mask].copy()
        extra  = [base.assign(**{flip_col: v}) for v in _EXTRA_FLIPS]
        result = pd.concat([result] + extra, ignore_index=True)
        added  = len(result) - before
        if added:
            print(f"    {prefix}: +{added} synthetic rows")
    return result


print("\n" + "=" * 70)
print("🔧 DATA AUGMENTATION")
print("=" * 70)

if df_gain is not None:
    n_before = len(df_gain)
    print("  3-client gain dataset:")
    df_gain = _augment_fitz14_rows(df_gain, ["c0", "c1", "c2"])
    print(f"  Total: {n_before} → {len(df_gain)} rows  (+{len(df_gain) - n_before} synthetic)")

if df_2c is not None:
    n_before = len(df_2c)
    print("  2-client comparison dataset:")
    df_2c = _augment_fitz14_rows(df_2c, ["c0", "c1"])
    print(f"  Total: {n_before} → {len(df_2c)} rows  (+{len(df_2c) - n_before} synthetic)")


# =============================================================================
# 3. SIGN-FLIP GAP TARGETS  (positive = improvement throughout)
# =============================================================================

_GAP_COLS_GAIN  = ["gain_soft_tpr_gap", "gain_hard_tpr_gap"]
_GAP_COLS_DELTA = ["delta_soft_tpr_gap", "delta_hard_tpr_gap"]

if df_gain is not None:
    for col in _GAP_COLS_GAIN:
        if col in df_gain.columns:
            df_gain[col] = -df_gain[col]
    print("\n  gain_dataset gap targets negated — positive now means improvement.")

if df_2c is not None:
    for col in _GAP_COLS_DELTA:
        if col in df_2c.columns:
            df_2c[col] = -df_2c[col]
    print("  comparison_dataset gap targets negated — positive now means improvement.")


# =============================================================================
# 4. FEATURE ENGINEERING
#    Both datasets are mapped to the same 10-column feature space:
#
#      agg_*  = weighted aggregate of the existing federation members
#      c2_*   = the new client joining
#
#    For 3-client rows: agg = C0+C1,  c2 = C2,  num_clients = 2
#    For 2-client rows: agg = C0,     c2 = C1,  num_clients = 1
# =============================================================================

def _build_features(
    df: pd.DataFrame,
    agg_prefixes: list[str],
    c2_prefix: str,
    num_clients: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build the 10-column feature matrix and the validity mask.

    Parameters
    ----------
    df            : augmented source DataFrame
    agg_prefixes  : client column prefixes that form the existing federation
    c2_prefix     : column prefix for the new joining client
    num_clients   : size of the existing federation (used as a feature)
    """
    X = pd.DataFrame(index=df.index)

    # Weighted features for each aggregated client
    agg_fitz14 = pd.Series(0.0, index=df.index)
    agg_flip   = pd.Series(0.0, index=df.index)
    agg_por    = pd.Series(0.0, index=df.index)

    for prefix in agg_prefixes:
        p   = df[f"{prefix}_portion"]
        f14 = df[f"{prefix}_fitz14_frac"]
        fl  = df[f"{prefix}_flip_frac"]
        agg_por    = agg_por    + p
        agg_fitz14 = agg_fitz14 + p * f14
        agg_flip   = agg_flip   + p * (1.0 - f14) * fl

    # New-client weighted features
    p_c2   = df[f"{c2_prefix}_portion"]
    f14_c2 = df[f"{c2_prefix}_fitz14_frac"]
    fl_c2  = df[f"{c2_prefix}_flip_frac"]
    c2_fitz14 = p_c2 * f14_c2
    c2_flip   = p_c2 * (1.0 - f14_c2) * fl_c2

    X["agg_portion"]               = agg_por
    X["c2_portion"]                = p_c2
    X["log_size_ratio"]            = np.log(agg_por / p_c2)
    X["agg_fitz14_frac_weighted"]  = agg_fitz14
    X["c2_fitz14_frac_weighted"]   = c2_fitz14
    X["agg_flip_frac_weighted"]    = agg_flip
    X["c2_flip_frac_weighted"]     = c2_flip
    X["fitz14_frac_weighted_diff"] = agg_fitz14 - c2_fitz14
    X["flip_frac_weighted_diff"]   = agg_flip   - c2_flip
    X["num_clients"]               = float(num_clients)

    valid = ~(X.isna().any(axis=1) | np.isinf(X).any(axis=1))
    return X[valid].reset_index(drop=True), valid.values


print("\n" + "=" * 70)
print("⚙️  FEATURE ENGINEERING")
print("=" * 70)

FEATURE_COLS = [
    "agg_portion", "c2_portion", "log_size_ratio",
    "agg_fitz14_frac_weighted", "c2_fitz14_frac_weighted",
    "agg_flip_frac_weighted",   "c2_flip_frac_weighted",
    "fitz14_frac_weighted_diff", "flip_frac_weighted_diff",
    "num_clients",
]

TARGET_COLS = [
    "gain_auc",
    "gain_bal_acc",
    "gain_soft_tpr_gap",
    "gain_min_soft_tpr",
    "gain_hard_tpr_gap",
    "gain_min_hard_tpr",
]

# 3-client gain dataset
X_gain, valid_gain = None, None
y_gain: dict[str, np.ndarray] = {}

if df_gain is not None:
    X_gain, valid_gain = _build_features(df_gain, ["c0", "c1"], "c2", num_clients=2)
    y_gain = {col: df_gain.loc[valid_gain, col].values for col in TARGET_COLS if col in df_gain.columns}
    print(f"  3-client: {X_gain.shape[0]} valid rows × {X_gain.shape[1]} features")

# 2-client comparison dataset  (C0 → agg,  C1 → c2)
_DELTA_TO_GAIN = {
    "delta_auc":          "gain_auc",
    "delta_bal_acc":      "gain_bal_acc",
    "delta_soft_tpr_gap": "gain_soft_tpr_gap",
    "delta_min_soft_tpr": "gain_min_soft_tpr",
    "delta_hard_tpr_gap": "gain_hard_tpr_gap",
    "delta_min_hard_tpr": "gain_min_hard_tpr",
}

X_2c, valid_2c = None, None
y_2c: dict[str, np.ndarray] = {}

if df_2c is not None:
    X_2c, valid_2c = _build_features(df_2c, ["c0"], "c1", num_clients=1)
    y_2c = {
        gain_col: df_2c.loc[valid_2c, delta_col].values
        for delta_col, gain_col in _DELTA_TO_GAIN.items()
        if delta_col in df_2c.columns
    }
    print(f"  2-client: {X_2c.shape[0]} valid rows × {X_2c.shape[1]} features")

# Verify column alignment before concatenating
if X_gain is not None and X_2c is not None:
    assert list(X_gain.columns) == list(X_2c.columns), (
        f"Column mismatch!\nX_gain: {list(X_gain.columns)}\nX_2c:   {list(X_2c.columns)}"
    )

# Concatenate
parts_X = [df for df in (X_gain, X_2c) if df is not None]
X_valid = pd.concat(parts_X, ignore_index=True)

# Only keep targets present in at least one dataset
TARGET_COLS = [
    col for col in TARGET_COLS
    if col in y_gain or col in y_2c
]

y_dict: dict[str, np.ndarray] = {}
for col in TARGET_COLS:
    arrays = []
    if col in y_gain: arrays.append(y_gain[col])
    if col in y_2c:   arrays.append(y_2c[col])
    y_dict[col] = np.concatenate(arrays)

print(f"\n  Combined: {len(X_valid)} rows × {len(FEATURE_COLS)} features")
print(f"  Targets ({len(TARGET_COLS)}):")
for t in TARGET_COLS:
    print(f"    - {t}")


# =============================================================================
# 5. MODEL DEFINITIONS
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
# 6. TRAINING
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
# 7. HELPERS
# =============================================================================

_NICE_NAMES = {
    "gain_auc":          "AUC Gain",
    "gain_bal_acc":      "Balanced Accuracy Gain",
    "gain_soft_tpr_gap": "Soft TPR Gap Δ (↑ = fairer)",
    "gain_min_soft_tpr": "Soft Min-TPR Gain",
    "gain_hard_tpr_gap": "Hard TPR Gap Δ (↑ = fairer)",
    "gain_min_hard_tpr": "Hard Min-TPR Gain",
}


def nice_name(col: str) -> str:
    return _NICE_NAMES.get(col, col)


def best_model_name(target_name: str) -> str:
    return max(
        results[target_name],
        key=lambda m: results[target_name][m]["metrics"]["cv_r2_mean"],
    )


# =============================================================================
# 8. PLOTS
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
# 9. SUMMARY TABLE
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
df_summary.to_csv(out_csv, index=False, float_format="%.6f")
print(f"✅ Summary saved → {out_csv}")

print("\nBest model per target:")
for target_name in TARGET_COLS:
    bm = best_model_name(target_name)
    r2 = results[target_name][bm]["metrics"]["cv_r2_mean"]
    print(f"  {target_name:<25s}  {bm:<20s}  CV R²={r2:+.4f}")
