"""
train_meta_regressors.py
=========================
Builds a meta-dataset from the federation-advantage CSVs produced by
compute_client0_advantages_*.py, engineers aggregated features, and trains
regression models that predict the benefit (or harm) of federating for
client 0.

Targets
-------
  Main (available for all setups):
    diff_nih_auc_macro       AUC increment (federated - standalone)
    diff_nih_ba_macro        Balanced-accuracy increment
    diff_tpr_gap             TPR-gap decrease  (positive = fairer)
    diff_tpr_min             Min-TPR increment (positive = fairer)

  Soft-TPR only (3- and 5-client, and 2-client if the column is present):
    diff_nih_soft_tpr_gap    Soft-TPR gap decrease
    diff_nih_soft_tpr_min    Soft min-TPR increment

Usage
-----
    python meta_learning/train_meta_regressors.py

Environment variables
---------------------
    RESULTS_2C   path to federation_advantage_client0_2clients.csv
                 (default: <repo_root>/federation_advantage_client0_2clients.csv)
    RESULTS_3C   path to federation_advantage_client0_3clients.csv
    RESULTS_5C   path to federation_advantage_client0_5clients.csv
    PLOT_DIR     directory where figures are saved (default: meta_learning/plots/)
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
REPO_ROOT  = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent

RESULTS_2C = Path(os.environ.get("RESULTS_2C", str(REPO_ROOT / "repo_nih" / "federation_advantage_client0_2clients.csv")))
RESULTS_3C = Path(os.environ.get("RESULTS_3C", str(REPO_ROOT / "repo_nih" / "federation_advantage_client0_3clients.csv")))
RESULTS_5C = Path(os.environ.get("RESULTS_5C", str(REPO_ROOT / "repo_nih" / "federation_advantage_client0_5clients.csv")))
PLOT_DIR   = Path(os.environ.get("PLOT_DIR",   str(_SCRIPT_DIR / "plots")))
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"]  = 100
plt.rcParams["savefig.dpi"] = 300


# =============================================================================
# 1. LOAD DATA
# =============================================================================

def load_csv(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"⚠️  {label} not found: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  {label}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


print("\n" + "=" * 70)
print("📂 LOADING DATA")
print("=" * 70)

df2 = load_csv(RESULTS_2C, "2-client")
df3 = load_csv(RESULTS_3C, "3-client")
df5 = load_csv(RESULTS_5C, "5-client")

if all(df is None for df in (df2, df3, df5)):
    sys.exit("❌  No input CSVs found. Check RESULTS_2C / RESULTS_3C / RESULTS_5C.")


# =============================================================================
# 2. DATA AUGMENTATION
#    The param-grid enforces flip_frac=0 when gender=all-male, so those rows
#    have no variance along the flip_frac axis.  We synthetically add rows
#    with non-zero flip fracs (duplicating the metric values) to prevent the
#    model from learning a spurious floor effect.  The number of extra rows
#    added to df3 / df5 is capped so that the augmentation proportion matches
#    that of df2 (which is always fully expanded).
# =============================================================================

EXTRA_FLIPS = [0.15, 0.30, 0.45]

# Column pairs: (male_pct_col, flip_frac_col) per client, per setup
_CLIENT_FLIP_COLS = {
    2: [("client_0_male_pct", "client_0_flip_frac"),
        ("client_1_male_pct", "client_1_flip_frac")],
    3: [("client_0_male_pct", "client_0_flip_frac"),
        ("client_1_male_pct", "client_1_flip_frac"),
        ("client_2_male_pct", "client_2_flip_frac")],
    5: [("client_0_male_pct", "client_0_flip_frac"),
        ("client_1_male_pct", "client_1_flip_frac"),
        ("client_2_male_pct", "client_2_flip_frac"),
        ("client_3_male_pct", "client_3_flip_frac"),
        ("client_4_male_pct", "client_4_flip_frac")],
}


def _count_augmented_rows(df: pd.DataFrame, client_flip_cols: list) -> int:
    """Total extra rows that uncapped augmentation would add."""
    return sum(
        (df[male_col].round(6) == 1.0).sum() * len(EXTRA_FLIPS)
        for male_col, _ in client_flip_cols
        if male_col in df.columns
    )


def expand_all_male_rows(
    df: pd.DataFrame,
    n_clients: int,
    max_rows_per_client: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    For every all-male client column, duplicate its rows with each extra flip
    fraction.  `max_rows_per_client` caps how many base rows are sampled per
    client column (None = no cap).
    """
    client_flip_cols = [
        (mc, fc) for mc, fc in _CLIENT_FLIP_COLS[n_clients]
        if mc in df.columns
    ]
    extra_rows = []
    for male_col, flip_col in client_flip_cols:
        base = df[df[male_col].round(6) == 1.0].copy()
        if max_rows_per_client is not None and len(base) > max_rows_per_client:
            base = base.sample(n=max_rows_per_client, random_state=random_state)
        for flip_val in EXTRA_FLIPS:
            rows = base.copy()
            rows[flip_col] = flip_val
            extra_rows.append(rows)
    if extra_rows:
        return pd.concat([df] + extra_rows, ignore_index=True)
    return df


print("\n" + "=" * 70)
print("🔧 DATA AUGMENTATION")
print("=" * 70)

# Compute augmentation proportion from df2 (reference, always uncapped)
_expanded_dfs = {}
if df2 is not None:
    n_orig2 = len(df2)
    n_added2 = _count_augmented_rows(df2, _CLIENT_FLIP_COLS[2])
    target_proportion = n_added2 / n_orig2
    df2 = expand_all_male_rows(df2, n_clients=2, max_rows_per_client=None)
    _expanded_dfs[2] = df2
    print(f"  df2: {n_orig2} → {len(df2)} rows  "
          f"(+{len(df2)-n_orig2}, proportion {target_proportion:.4f})")
else:
    # Fall back to a generic 10 % target if df2 is missing
    target_proportion = 0.10
    print(f"  df2 missing — using fallback augmentation proportion {target_proportion}")

for n_cl, df_n, label in ((3, df3, "df3"), (5, df5, "df5")):
    if df_n is None:
        continue
    n_orig = len(df_n)
    n_target_added    = round(target_proportion * n_orig)
    n_flip_cols       = len(_CLIENT_FLIP_COLS[n_cl])
    max_rows_per_cli  = max(1, round(n_target_added / (n_flip_cols * len(EXTRA_FLIPS))))
    df_aug = expand_all_male_rows(df_n, n_clients=n_cl, max_rows_per_client=max_rows_per_cli)
    _expanded_dfs[n_cl] = df_aug
    print(f"  {label}: {n_orig} → {len(df_aug)} rows  "
          f"(+{len(df_aug)-n_orig}, max {max_rows_per_cli} rows/client)")

df2 = _expanded_dfs.get(2)
df3 = _expanded_dfs.get(3)
df5 = _expanded_dfs.get(5)


# =============================================================================
# 3. FEATURE ENGINEERING — aggregate "other clients" into 3 scalars
# =============================================================================

def _build_features(df: pd.DataFrame, n_clients: int) -> pd.DataFrame:
    """
    Returns a copy of df with three new aggregated columns:
      agg_portion    total data fraction held by clients 1..N-1
      agg_male_pct   portion-weighted male fraction of clients 1..N-1
      agg_flip_frac  effective flip burden of clients 1..N-1
                     = Σ flip_i * (1 - male_i) * portion_i
    """
    feat = df.copy()
    other_ids = list(range(1, n_clients))

    feat["agg_portion"] = sum(
        feat[f"client_{i}_portion"] for i in other_ids
    )
    feat["agg_male_pct"] = sum(
        feat[f"client_{i}_male_pct"] * feat[f"client_{i}_portion"] for i in other_ids
    )
    feat["agg_flip_frac"] = sum(
        feat[f"client_{i}_flip_frac"]
        * (1 - feat[f"client_{i}_male_pct"])
        * feat[f"client_{i}_portion"]
        for i in other_ids
    )
    feat["n_clients"] = n_clients

    # Drop the now-redundant per-client columns for clients 1..N-1
    other_cols = [
        col for col in feat.columns
        if any(col.startswith(f"client_{i}_") for i in other_ids)
    ]
    feat = feat.drop(columns=other_cols)

    return feat


print("\n" + "=" * 70)
print("⚙️  FEATURE ENGINEERING")
print("=" * 70)

parts = []
for n_cl, df_n in ((2, df2), (3, df3), (5, df5)):
    if df_n is None:
        continue
    parts.append(_build_features(df_n, n_cl))

df_merged = pd.concat(parts, ignore_index=True)
print(f"  Merged dataset: {df_merged.shape[0]} rows × {df_merged.shape[1]} columns")

# ── Derived features ─────────────────────────────────────────────────────────
df_merged["client0_female_pct_weighted"] = (
    df_merged["client_0_portion"] * (1 - df_merged["client_0_male_pct"])
)
df_merged["client0_male_pct_weighted"]   = (
    df_merged["client_0_portion"] * df_merged["client_0_male_pct"]
)
df_merged["client0_flip_frac_weighted"]  = (
    df_merged["client0_female_pct_weighted"] * df_merged["client_0_flip_frac"]
)
df_merged["male_pct_weighted_diff"]      = (
    df_merged["agg_male_pct"] - df_merged["client0_male_pct_weighted"]
)
df_merged["flip_frac_weighted_diff"]     = (
    df_merged["agg_flip_frac"] - df_merged["client0_flip_frac_weighted"]
)
df_merged["log_size_ratio"]              = np.log(
    df_merged["agg_portion"] / df_merged["client_0_portion"]
)
df_merged["male_pct_diff"]               = (
    df_merged["agg_male_pct"] - df_merged["client_0_male_pct"]
)
df_merged["flip_frac_diff"]              = (
    df_merged["agg_flip_frac"] - df_merged["client_0_flip_frac"]
)

FEATURE_COLS = [
    "client_0_portion",         "agg_portion",
    "log_size_ratio",
    "client0_male_pct_weighted", "agg_male_pct",         "male_pct_weighted_diff",
    "client0_flip_frac_weighted","agg_flip_frac",         "flip_frac_weighted_diff",
    "n_clients",
]


# =============================================================================
# 4. TARGET DEFINITIONS AND VALID-ROW MASKS
# =============================================================================

TARGET_COLS_MAIN = [
    "diff_nih_auc_macro",
    "diff_nih_ba_macro",
    "diff_tpr_gap",
    "diff_tpr_min",
]
TARGET_COLS_SOFT = [
    "diff_nih_soft_tpr_gap",
    "diff_nih_soft_tpr_min",
]

# Only keep targets that actually exist in the merged data
TARGET_COLS_MAIN = [c for c in TARGET_COLS_MAIN if c in df_merged.columns]
TARGET_COLS_SOFT = [c for c in TARGET_COLS_SOFT if c in df_merged.columns]
ALL_TARGETS      = TARGET_COLS_MAIN + TARGET_COLS_SOFT

X_full = df_merged[FEATURE_COLS]
feature_valid = ~(X_full.isna().any(axis=1) | np.isinf(X_full).any(axis=1))

valid_mask_main = feature_valid & df_merged[TARGET_COLS_MAIN].notna().all(axis=1)
valid_mask_soft = feature_valid & df_merged[TARGET_COLS_SOFT].notna().all(axis=1) \
    if TARGET_COLS_SOFT else pd.Series(False, index=df_merged.index)

X_main = X_full[valid_mask_main].copy()
X_soft = X_full[valid_mask_soft].copy()

y_main = {col: df_merged.loc[valid_mask_main, col].values for col in TARGET_COLS_MAIN}
y_soft = {col: df_merged.loc[valid_mask_soft, col].values for col in TARGET_COLS_SOFT}

n_cl_main = df_merged.loc[valid_mask_main, "n_clients"].values
n_cl_soft = df_merged.loc[valid_mask_soft, "n_clients"].values

print(f"\n  Main targets — valid rows: {len(X_main)}")
for n_cl in sorted(df_merged["n_clients"].unique()):
    print(f"    n_clients={int(n_cl)}: {(n_cl_main == n_cl).sum()}")

if TARGET_COLS_SOFT:
    print(f"\n  Soft-TPR targets — valid rows: {len(X_soft)}")
    for n_cl in sorted(df_merged["n_clients"].unique()):
        print(f"    n_clients={int(n_cl)}: {(n_cl_soft == n_cl).sum()}")

print("\nNaN counts per setup per target:")
for col in ALL_TARGETS:
    if col in df_merged.columns:
        counts = df_merged.groupby("n_clients")[col].apply(lambda s: s.isna().sum())
        print(f"  {col}:")
        for n_cl, cnt in counts.items():
            print(f"    n_clients={int(n_cl)}: {cnt} NaN")


# =============================================================================
# 5. MODEL DEFINITIONS
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
                "max_depth":        [3, 4, 5, 6, 7],
                "min_child_weight": [1, 3, 5, 7],
                "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
                "reg_alpha":        np.logspace(-3, 1, 20).tolist(),
                "reg_lambda":       np.logspace(-3, 1, 20).tolist(),
                "gamma":            [0, 0.1, 0.3, 0.5, 1.0],
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

training_tasks = (
    [(t, y_main[t], X_main) for t in TARGET_COLS_MAIN]
    + [(t, y_soft[t], X_soft) for t in TARGET_COLS_SOFT]
)

for target_name, y, X_use in training_tasks:
    print(f"\n{'='*70}\n  Target: {target_name}  ({len(y)} samples)\n{'='*70}")
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
        search.fit(X_use.values, y)
        best_est = search.best_estimator_

        cv_r2   = cross_val_score(best_est, X_use.values, y, cv=CV, scoring="r2")
        cv_rmse = cross_val_score(best_est, X_use.values, y, cv=CV,
                                  scoring="neg_root_mean_squared_error")
        cv_mae  = cross_val_score(best_est, X_use.values, y, cv=CV,
                                  scoring="neg_mean_absolute_error")

        results[target_name][model_name] = {
            "model":   best_est,
            "X":       X_use,
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
# 7. HELPERS
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
# 8. PLOTS
# =============================================================================

def _plot_predictions(target_names: list, filename: str) -> None:
    """Predicted vs Actual scatter plot for a list of targets."""
    n_targets = len(target_names)
    if n_targets == 0:
        return
    ncols = 2
    nrows = (n_targets + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 5))
    axes = np.array(axes).flatten()   # works for both 1-row and multi-row grids

    for idx, target_name in enumerate(target_names):
        ax = axes[idx]
        bm       = best_model_name(target_name)
        entry    = results[target_name][bm]
        best_est = entry["model"]
        X_use    = entry["X"]
        y_true   = entry["y"]

        # Use the same X that the model was trained on
        y_pred = cross_val_predict(best_est, X_use.values, y_true, cv=CV, n_jobs=-1)

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


def _plot_feature_importance(target_names: list, filename: str) -> None:
    """Bar chart of feature importances for tree-based best models."""
    tree_based = {"Random Forest", "Gradient Boosting", "XGBoost"}
    eligible = [
        t for t in target_names
        if best_model_name(t) in tree_based
    ]
    if not eligible:
        print("ℹ️  No tree-based best models — skipping feature importance plot.")
        return

    ncols = min(2, len(eligible))
    nrows = (len(eligible) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for idx, target_name in enumerate(eligible):
        ax = axes[idx]
        bm    = best_model_name(target_name)
        model = results[target_name][bm]["model"]

        # Handle Pipeline wrappers
        est = model.named_steps["model"] if hasattr(model, "named_steps") else model
        importances = est.feature_importances_

        sorted_idx = np.argsort(importances)[::-1]
        ax.bar(range(len(FEATURE_COLS)), importances[sorted_idx])
        ax.set_xticks(range(len(FEATURE_COLS)))
        ax.set_xticklabels(
            [FEATURE_COLS[i] for i in sorted_idx], rotation=45, ha="right", fontsize=11
        )
        ax.set_title(f"{nice_name(target_name)} — {bm}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Importance", fontsize=12)

    for ax in axes[len(eligible):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = PLOT_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved → {out}")


def _plot_cv_r2_comparison(target_names: list, filename: str) -> None:
    """Grouped bar chart comparing CV R² across models for each target."""
    model_names = list(results[target_names[0]].keys())
    x = np.arange(len(target_names))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(max(12, 3 * len(target_names)), 6))

    for i, model_name in enumerate(model_names):
        r2_means = [results[t][model_name]["metrics"]["cv_r2_mean"] for t in target_names]
        r2_stds  = [results[t][model_name]["metrics"]["cv_r2_std"]  for t in target_names]
        offset   = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, r2_means, width, yerr=r2_stds,
               label=model_name, capsize=4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([nice_name(t) for t in target_names], rotation=20, ha="right", fontsize=12)
    ax.set_ylabel("CV R² (5-fold)", fontsize=13)
    ax.set_title("Model Comparison — Cross-validated R²", fontsize=15, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=11, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = PLOT_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved → {out}")


print("\n" + "=" * 70)
print("📊 GENERATING PLOTS")
print("=" * 70)

_plot_predictions(TARGET_COLS_MAIN, "predictions_vs_actual_main.png")

if TARGET_COLS_SOFT:
    _plot_predictions(TARGET_COLS_SOFT, "predictions_vs_actual_soft_tpr.png")




# =============================================================================
# 9. SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)

summary_rows = []
for target_name in ALL_TARGETS:
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
for target_name in ALL_TARGETS:
    bm = best_model_name(target_name)
    r2 = results[target_name][bm]["metrics"]["cv_r2_mean"]
    print(f"  {target_name:<35s}  {bm:<22s}  CV R²={r2:+.4f}")
