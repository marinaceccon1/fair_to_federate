"""
train_federation_regressors.py
================================
Trains regression models that predict the *incremental* advantage of joining
a federation, framed as:

    "Given an existing 2-client federation (C0+C1), what is the advantage
     for a new client C2 of joining it?"

The two source datasets are merged into a unified feature space:

  3-client rows  (federation_advantage_3clients_vs_2clients.csv)
      C0 + C1  = existing federation entity
      C2       = new client joining

  2-client rows  (federation_advantage_client0.csv)
      C0       = existing federation entity  (solo client)
      C1 → C2  = new client joining
      (C1 columns are renamed to C2 before merging)

Targets
-------
  diff_nih_auc_macro        AUC increment (federated − standalone)
  diff_nih_ba_macro         Balanced-accuracy increment
  diff_tpr_gap              Hard TPR-gap decrease  (positive = fairer)
  diff_tpr_min              Hard min-TPR increment
  diff_nih_soft_tpr_gap     Soft TPR-gap decrease
  diff_nih_soft_tpr_min     Soft min-TPR increment

Features (10)
-------------
  fed_portion, client_2_portion, log_size_ratio,
  fed_male_weighted, c2_male_weighted, male_weighted_diff,
  fed_flip_weighted, c2_flip_weighted, flip_weighted_diff,
  n_clients

Usage
-----
    python meta_learning/train_federation_regressors.py

Environment variables
---------------------
    RESULTS_3C_VS_2C   path to federation_advantage_3clients_vs_2clients.csv
                       (default: <repo_root>/federation_advantage_3clients_vs_2clients.csv)
    RESULTS_2C         path to federation_advantage_client0.csv
                       (default: <repo_root>/federation_advantage_client0.csv)
    PLOT_DIR           directory where figures are saved
                       (default: meta_learning/plots_federation/)
"""

import os
import sys
import warnings
from itertools import product as iproduct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent

RESULTS_3C_VS_2C = Path(os.environ.get(
    "RESULTS_3C_VS_2C",
    str(REPO_ROOT / "repo_nih" / "federation_advantage_3clients_vs_2clients.csv"),
))
RESULTS_2C = Path(os.environ.get(
    "RESULTS_2C",
    str(REPO_ROOT / "repo_nih" / "federation_advantage_client0.csv"),
))
PLOT_DIR = Path(os.environ.get("PLOT_DIR", str(_SCRIPT_DIR / "plots_federation")))
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

df_3c = _load_csv(RESULTS_3C_VS_2C, "3-client-vs-2-client")
df_2c = _load_csv(RESULTS_2C,       "2-client")

if df_3c is None and df_2c is None:
    sys.exit("❌  No input CSVs found. Check RESULTS_3C_VS_2C / RESULTS_2C.")

if df_3c is not None:
    df_3c = df_3c[df_3c["two_client_found"] == True].reset_index(drop=True)
    print(f"  3-client: {len(df_3c)} rows after filtering to matched pairs")


# =============================================================================
# 2. DATA AUGMENTATION
#    The param-grid enforces flip_frac=0 for all-male clients.  We synthesise
#    copies with non-zero flip fracs to prevent the model learning a spurious
#    floor effect.  For multi-client rows where several clients are all-male,
#    a full cross-product of flip values is generated (excluding all-zero).
# =============================================================================

_EXTRA_FLIPS = [0.15, 0.30, 0.45]
_ALL_FLIPS   = [0.0] + _EXTRA_FLIPS


def _expand_all_male_rows(df: pd.DataFrame, client_ids: list[int]) -> pd.DataFrame:
    """
    For each row, identify which of the given client IDs are all-male
    (male_pct == 1.0, flip_frac == 0.0) and synthesise copies covering
    the missing flip_frac values.  When multiple clients trigger, a full
    cross-product is generated (all-zero combo excluded).

    Parameters
    ----------
    df          : input DataFrame
    client_ids  : list of client indices to check (e.g. [0, 1, 2])
    """
    # Identify which client columns exist and can trigger expansion
    trigger_cols: dict[int, tuple[str, str]] = {}
    for c in client_ids:
        male_col = f"client_{c}_male_pct"
        flip_col = f"client_{c}_flip_frac"
        if male_col in df.columns and flip_col in df.columns:
            trigger_cols[c] = (male_col, flip_col)

    extra_rows = []
    for _, row in df.iterrows():
        # Which clients are all-male with flip=0 in this row?
        triggered = {
            c: flip_col
            for c, (male_col, flip_col) in trigger_cols.items()
            if row[male_col] == 1.0 and row[flip_col] == 0.0
        }
        if not triggered:
            continue

        flip_cols_ordered = list(triggered.values())
        if len(triggered) == 1:
            combos = [(v,) for v in _EXTRA_FLIPS]
        else:
            combos = [
                combo for combo in iproduct(*[_ALL_FLIPS] * len(triggered))
                if not all(v == 0.0 for v in combo)
            ]

        for combo in combos:
            new_row = row.copy()
            for flip_col, val in zip(flip_cols_ordered, combo):
                new_row[flip_col] = val
            extra_rows.append(new_row)

    if not extra_rows:
        print("ℹ️  No all-male / flip=0.0 rows found — nothing to expand.")
        return df

    df_out = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
    print(f"  {len(df)} original + {len(extra_rows)} synthetic = {len(df_out)} total rows")
    return df_out


print("\n" + "=" * 70)
print("🔧 DATA AUGMENTATION")
print("=" * 70)

if df_3c is not None:
    print("  3-client dataset:")
    df_3c = _expand_all_male_rows(df_3c, client_ids=[0, 1, 2])

if df_2c is not None:
    print("  2-client dataset:")
    df_2c = _expand_all_male_rows(df_2c, client_ids=[0, 1])

print(f"\nPost-expansion — "
      f"3-client: {len(df_3c) if df_3c is not None else 'N/A'}, "
      f"2-client: {len(df_2c) if df_2c is not None else 'N/A'}")


# =============================================================================
# 3. MERGE
#    Rename 2-client client_1_* → client_2_* so that C2 always means
#    "new client joining" in both datasets, then concatenate.
# =============================================================================

print("\n" + "=" * 70)
print("🔗 MERGING DATASETS")
print("=" * 70)

parts = []

if df_3c is not None:
    df_3c = df_3c.copy()
    df_3c["source"]   = "3client"
    df_3c["n_clients"] = 3
    parts.append(df_3c)

if df_2c is not None:
    df_2c = df_2c.rename(columns={
        "client_1_portion":    "client_2_portion",
        "client_1_male_pct":   "client_2_male_pct",
        "client_1_female_pct": "client_2_female_pct",
        "client_1_flip_frac":  "client_2_flip_frac",
    })
    df_2c["source"]   = "2client"
    df_2c["n_clients"] = 2
    parts.append(df_2c)

df = pd.concat(parts, ignore_index=True, sort=False)
print(f"  Merged: {len(df)} rows  "
      f"({len(df_3c) if df_3c is not None else 0} three-client + "
      f"{len(df_2c) if df_2c is not None else 0} two-client)")


# =============================================================================
# 4. FEATURE ENGINEERING
#    Aggregate C0 + C1 into a single "existing federation" entity.
#    For 2-client rows, C1 columns are absent (NaN → 0), so the aggregation
#    naturally collapses to C0 only.
# =============================================================================

print("\n" + "=" * 70)
print("⚙️  FEATURE ENGINEERING")
print("=" * 70)


def _col_or_zeros(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if it exists, else a zero Series of the same length."""
    return df[col].fillna(0.0) if col in df.columns else pd.Series(0.0, index=df.index)


X = pd.DataFrame(index=df.index)

# Per-client weighted features
for c in [0, 1, 2]:
    portion   = _col_or_zeros(df, f"client_{c}_portion")
    male_pct  = _col_or_zeros(df, f"client_{c}_male_pct")
    flip_frac = _col_or_zeros(df, f"client_{c}_flip_frac")
    X[f"c{c}_male_weighted"]   = portion * male_pct
    X[f"c{c}_female_weighted"] = portion * (1.0 - male_pct)
    X[f"c{c}_flip_weighted"]   = X[f"c{c}_female_weighted"] * flip_frac

# Aggregate C0 + C1 → existing federation entity
X["fed_portion"]         = _col_or_zeros(df, "client_0_portion") + _col_or_zeros(df, "client_1_portion")
X["fed_male_weighted"]   = X["c0_male_weighted"]   + X["c1_male_weighted"]
X["fed_female_weighted"] = X["c0_female_weighted"] + X["c1_female_weighted"]
X["fed_flip_weighted"]   = X["c0_flip_weighted"]   + X["c1_flip_weighted"]

# C2 = new client joining
X["client_2_portion"] = _col_or_zeros(df, "client_2_portion")

# Cross-entity difference features (new client C2 − existing federation)
X["male_weighted_diff"] = X["fed_male_weighted"] - X["c2_male_weighted"]
X["flip_weighted_diff"] = X["fed_flip_weighted"] - X["c2_flip_weighted"]

# Log size ratio: how much larger is the federation than C2?
X["log_size_ratio"] = np.log(
    X["fed_portion"].replace(0.0, np.nan) / X["client_2_portion"].replace(0.0, np.nan)
)

# Number of clients already in the federation
X["n_clients"] = df["n_clients"].fillna(2).astype(int)

FEATURE_COLS = [
    "fed_portion",        "client_2_portion",
    "log_size_ratio",
    "fed_male_weighted",  "c2_male_weighted",  "male_weighted_diff",
    "fed_flip_weighted",  "c2_flip_weighted",  "flip_weighted_diff",
    "n_clients",
]

print(f"  Feature matrix: {len(X)} rows × {len(FEATURE_COLS)} features")
for f in FEATURE_COLS:
    print(f"    - {f}")


# =============================================================================
# 5. TARGET DEFINITIONS AND VALID-ROW MASKS
# =============================================================================

TARGET_COLS = [
    "diff_nih_auc_macro",
    "diff_nih_ba_macro",
    "diff_tpr_gap",
    "diff_tpr_min",
    "diff_nih_soft_tpr_gap",
    "diff_nih_soft_tpr_min",
]
# Only keep targets that actually exist in the merged data
TARGET_COLS = [c for c in TARGET_COLS if c in df.columns]

feature_valid = ~(
    X[FEATURE_COLS].isna().any(axis=1) | np.isinf(X[FEATURE_COLS]).any(axis=1)
)
target_valid = df[TARGET_COLS].notna().all(axis=1)
valid_mask   = feature_valid & target_valid

X_valid  = X.loc[valid_mask, FEATURE_COLS].reset_index(drop=True)
y_dict   = {col: df.loc[valid_mask, col].values for col in TARGET_COLS}
source_valid = df.loc[valid_mask, "source"].values

print(f"\n  Valid samples: {len(X_valid)}")
print(f"    3-client rows: {(source_valid == '3client').sum()}")
print(f"    2-client rows: {(source_valid == '2client').sum()}")

print("\nNaN counts per source per target:")
for col in TARGET_COLS:
    counts = df.groupby("source")[col].apply(lambda s: s.isna().sum())
    print(f"  {col}:")
    for src, cnt in counts.items():
        print(f"    {src}: {cnt} NaN")


# =============================================================================
# 6. MODEL DEFINITIONS
# =============================================================================

CV      = KFold(n_splits=5, shuffle=True, random_state=42)
N_ITER  = 30
SCORING = "r2"


def build_search_spaces() -> dict:
    spaces = {
        "Linear Regression": (
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=0.0))]),
            {"model__alpha": [0.0]},
        ),
        "Ridge Regression": (
            Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": np.logspace(-3, 3, 100).tolist()},
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
                random_state=42, n_jobs=-1, verbosity=0,
                eval_metric="rmse", device="cpu",
            ),
            {
                "n_estimators":     [100, 200, 300, 500],
                "learning_rate":    np.logspace(-2, 0, 30).tolist(),
                "max_depth":        [2, 3, 4, 5, 6],
                "subsample":        [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
                "reg_alpha":        [0, 0.1, 0.5, 1.0],
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
            f"  {model_name:22s}  "
            f"CV R²: {cv_r2.mean():+.4f} ± {cv_r2.std():.4f}  "
            f"RMSE: {(-cv_rmse).mean():.6f} ± {(-cv_rmse).std():.6f}"
        )

print("\n✅ Training complete!")


# =============================================================================
# 8. HELPERS
# =============================================================================

_NICE_NAMES = {
    "diff_nih_auc_macro":        "AUC Macro Δ",
    "diff_nih_ba_macro":         "Balanced Accuracy Δ",
    "diff_tpr_gap":              "Hard TPR Gap Δ (↑ = fairer)",
    "diff_tpr_min":              "Hard TPR Min Δ",
    "diff_nih_soft_tpr_gap":     "Soft TPR Gap Δ (↑ = fairer)",
    "diff_nih_soft_tpr_min":     "Soft TPR Min Δ",
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
    print(f"  {target_name:<35s}  {bm:<22s}  CV R²={r2:+.4f}")
