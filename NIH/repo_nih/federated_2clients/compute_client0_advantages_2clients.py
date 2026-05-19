"""
compute_client0_advantages_2clients.py
========================================
Compute Federation Advantages for Client 0 — 2-client setup.

This script computes the advantage for CLIENT 0 when federating with CLIENT 1.

For each federated model with:
  - Client 0: configuration A
  - Client 1: configuration B

We compute:
  Advantage = Federated metrics (Client 0 = config A) - Standalone Client 0 metrics (config A)

This gives us a clear, unambiguous question:
  "What is the advantage for a client with configuration A
   when federating with a client with configuration B?"

Metrics computed:
  - AUC increment               (federated - standalone)
  - Balanced Accuracy increment (federated - standalone)
  - Male TPR increment          (federated - standalone)
  - Female TPR increment        (federated - standalone)
  - TPR gap decrease            (standalone_gap - federated_gap, positive = improvement)
  - Minimum TPR increment       (federated - standalone)

OUTPUT FORMAT: Matches the structure of federated_vs_standalone.csv

Sources read directly (no intermediate CSVs required):
  Standalone → evaluation_results_standalone_client0/new_results_*.json
               (all timestamped JSON files are merged; later runs override earlier
                ones if the same model appears more than once)
  Federated  → evaluation_results_2clients/metrics_auc.csv
               evaluation_results_2clients/metrics_soft_tpr.csv  (or metrics_hard_tpr.csv)
               evaluation_results_2clients/metrics_balanced_accuracy.csv
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — works regardless of cwd
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]   # repo/
sys.path.append(str(REPO_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

FEDERATED_RESULTS_DIR  = REPO_ROOT / "repo_nih" / "evaluation_results_2clients"
STANDALONE_RESULTS_DIR = REPO_ROOT / "repo_nih" / "evaluation_results_standalone_client0"
OUTPUT_CSV             = REPO_ROOT / "repo_nih" / "federation_advantage_client0_2clients.csv"

# =============================================================================
# HELPERS
# =============================================================================

def _round4(v) -> float:
    return round(float(v), 4)


def _config_to_key(config: dict) -> tuple:
    """
    Build a lookup key from a standalone config dict:
      {"portion": ..., "gender": {"Male": ..., "Female": ...}, "flip_frac": ...}

    flip_frac here is the *config* value (already 1.0 for the F100/flip0.9 special
    case — parse_standalone_filename in the evaluation script handles that before
    saving to JSON).
    """
    return (
        _round4(config["portion"]),
        _round4(config["gender"]["Male"]),
        _round4(config["gender"]["Female"]),
        _round4(config["flip_frac"]),
    )


def _client0_key(row: pd.Series) -> tuple:
    """Config key for Client 0 from a federated metrics row."""
    return (
        _round4(row["client_0_portion"]),
        _round4(row["client_0_male_pct"]),
        _round4(row["client_0_female_pct"]),
        _round4(row["client_0_flip_frac"]),
    )


def _pick(d: dict, *keys):
    """Return the value of the first key that exists and is not None/NaN."""
    for k in keys:
        v = d.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return float(v)
    return np.nan


def _pick_row(row: pd.Series, *candidates) -> float:
    """Return the first column that exists and is not NaN, else np.nan."""
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return float(row[c])
    return np.nan


# =============================================================================
# LOAD STANDALONE CLIENT 0 RESULTS  (from new_results_*.json files)
# =============================================================================

def load_standalone(results_dir: Path) -> dict:
    """
    Glob all  new_results_*.json  files written by evaluate_standalone_client0.py,
    merge them (later timestamp wins on duplicate model paths), and return a
    lookup dict:

        (portion, male_frac, female_frac, flip_frac) -> {metric: value, ...}
    """
    json_files = sorted(results_dir.glob("new_results_*.json"))

    if not json_files:
        fallback = results_dir / "full_results.json"
        if fallback.exists():
            json_files = [fallback]
        else:
            sys.exit(
                f"❌  No standalone result files found in: {results_dir}\n"
                f"    Expected: new_results_*.json  (written by evaluate_standalone_client0.py)"
            )

    print(f"✅  Found {len(json_files)} standalone JSON file(s) in {results_dir}")

    raw: dict[str, dict] = {}
    for jf in json_files:
        with open(jf) as f:
            entries = json.load(f)
        for entry in entries:
            raw[entry["model_path"]] = entry

    print(f"   Total standalone entries after merge: {len(raw)}")

    lookup: dict[tuple, dict] = {}
    skipped = 0

    for model_path, entry in raw.items():
        config = entry.get("config", {})
        if not config:
            skipped += 1
            continue

        key = _config_to_key(config)

        nih      = entry.get("nih_test", {})
        auc      = nih.get("auc", {})
        hard_tpr = nih.get("hard_tpr", {})
        soft_tpr = nih.get("soft_tpr", {})
        ba       = nih.get("balanced_accuracy", {})

        hard_male   = _pick(hard_tpr, "cumulative_male")
        hard_female = _pick(hard_tpr, "cumulative_female")
        hard_gap    = (hard_male - hard_female) if not (np.isnan(hard_male) or np.isnan(hard_female)) else np.nan
        hard_min    = min(hard_male, hard_female) if not (np.isnan(hard_male) or np.isnan(hard_female)) else np.nan

        soft_male   = _pick(soft_tpr, "micro_male")
        soft_female = _pick(soft_tpr, "micro_female")
        soft_gap    = _pick(soft_tpr, "gap")
        soft_min    = _pick(soft_tpr, "min")

        lookup[key] = {
            "auc_macro":       _pick(auc, "avg", "macro"),
            "ba_macro":        _pick(ba,  "macro"),
            "hard_tpr_male":   hard_male,
            "hard_tpr_female": hard_female,
            "hard_tpr_gap":    hard_gap,
            "hard_tpr_min":    hard_min,
            "soft_tpr_male":   soft_male,
            "soft_tpr_female": soft_female,
            "soft_tpr_gap":    soft_gap,
            "soft_tpr_min":    soft_min,
        }

    print(f"   Lookup entries built: {len(lookup)}  (skipped: {skipped})")
    return lookup


# =============================================================================
# LOAD FEDERATED 2-CLIENT RESULTS
# =============================================================================

def _load_fed_csv(path: Path, required: bool = True) -> "pd.DataFrame | None":
    if not path.exists():
        if required:
            sys.exit(f"❌  Federated metrics CSV not found: {path}")
        print(f"⚠️   Optional federated CSV not found, skipping: {path}")
        return None
    df = pd.read_csv(path)
    print(f"✅  Loaded {path.name:<45s} — {len(df)} rows")
    return df


def load_federated(results_dir: Path) -> "tuple[pd.DataFrame, str]":
    """
    Merge the per-metric CSVs produced by evaluate_federated_2clients.py
    into a single wide DataFrame, one row per model.

    Returns (merged_df, tpr_source) where tpr_source is "soft" or "hard".
    """
    df_auc = _load_fed_csv(results_dir / "metrics_auc.csv",  required=True)
    df_ba  = _load_fed_csv(results_dir / "metrics_balanced_accuracy.csv", required=True)

    df_soft_tpr = _load_fed_csv(results_dir / "metrics_soft_tpr.csv", required=False)
    df_hard_tpr = _load_fed_csv(results_dir / "metrics_hard_tpr.csv", required=False)

    if df_soft_tpr is None and df_hard_tpr is None:
        sys.exit(f"❌  Neither metrics_soft_tpr.csv nor metrics_hard_tpr.csv found in {results_dir}")

    tpr_source = "soft" if df_soft_tpr is not None else "hard"
    print(f"   Primary TPR source: {tpr_source}")

    potential_keys = [
        "model_name", "num_clients",
        "client_0_portion", "client_0_male_pct", "client_0_female_pct", "client_0_flip_frac",
        "client_1_portion", "client_1_male_pct", "client_1_female_pct", "client_1_flip_frac",
    ]
    key_cols = [c for c in potential_keys if c in df_auc.columns]

    def _drop_dup_non_key(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(
            columns=[c for c in df.columns if c in df_auc.columns and c not in key_cols],
            errors="ignore"
        )

    merged = df_auc.copy()
    for df_extra in (df_ba, df_soft_tpr, df_hard_tpr):
        if df_extra is None:
            continue
        merged = merged.merge(_drop_dup_non_key(df_extra), on=key_cols, how="left")

    print(f"   Merged federated table: {len(merged)} rows × {merged.shape[1]} columns")
    return merged, tpr_source


# =============================================================================
# PICK METRIC VALUES FROM A MERGED FEDERATED ROW
# =============================================================================

def _fed_metrics(row: pd.Series, tpr_source: str) -> dict:
    auc = _pick_row(row, "nih_auc_macro")
    ba  = _pick_row(row, "nih_ba_macro", "nih_ba_micro")

    soft_micro  = _pick_row(row, "nih_soft_tpr_micro")
    soft_male   = _pick_row(row, "nih_soft_tpr_micro_male")
    soft_female = _pick_row(row, "nih_soft_tpr_micro_female")
    soft_gap    = _pick_row(row, "nih_soft_tpr_gap")
    soft_min    = _pick_row(row, "nih_soft_tpr_min")

    if np.isnan(soft_gap) and not (np.isnan(soft_male) or np.isnan(soft_female)):
        soft_gap = soft_male - soft_female
    if np.isnan(soft_min) and not (np.isnan(soft_male) or np.isnan(soft_female)):
        soft_min = min(soft_male, soft_female)

    hard_male   = _pick_row(row, "nih_hard_tpr_micro_male")
    hard_female = _pick_row(row, "nih_hard_tpr_micro_female")

    hard_gap = np.nan
    hard_min = np.nan
    if not (np.isnan(hard_male) or np.isnan(hard_female)):
        hard_gap = hard_male - hard_female
        hard_min = min(hard_male, hard_female)

    return {
        "auc_macro":       auc,
        "ba_macro":        ba,
        "tpr_male":        hard_male,
        "tpr_female":      hard_female,
        "tpr_gap":         hard_gap,
        "tpr_min":         hard_min,
        "soft_tpr_micro":  soft_micro,
        "soft_tpr_male":   soft_male,
        "soft_tpr_female": soft_female,
        "soft_tpr_gap":    soft_gap,
        "soft_tpr_min":    soft_min,
    }


# =============================================================================
# BUILD STANDALONE MODEL NAME
# =============================================================================

def _standalone_model_name(key: tuple) -> str:
    portion, male_frac, _female_frac, flip_frac = key
    male_int = int(round(male_frac * 100))
    return f"standalone_p{portion}_M{male_int}_flip{flip_frac}"


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_advantages(fed_df: pd.DataFrame, tpr_source: str, sa_lookup: dict) -> pd.DataFrame:
    rows_out  = []
    matched   = 0
    unmatched = 0

    for _, row in fed_df.iterrows():
        c0_key = _client0_key(row)
        sa     = sa_lookup.get(c0_key)

        if sa is None:
            unmatched += 1
            sa = {k: np.nan for k in (
                "auc_macro", "ba_macro",
                "hard_tpr_male", "hard_tpr_female", "hard_tpr_gap", "hard_tpr_min",
                "soft_tpr_male", "soft_tpr_female", "soft_tpr_gap", "soft_tpr_min",
            )}
            standalone_found = False
        else:
            matched += 1
            standalone_found = True

        fed = _fed_metrics(row, tpr_source)

        diff_auc = fed["auc_macro"] - sa["auc_macro"]
        diff_ba  = fed["ba_macro"]  - sa["ba_macro"]

        diff_hard_male   = fed["tpr_male"]   - sa["hard_tpr_male"]
        diff_hard_female = fed["tpr_female"] - sa["hard_tpr_female"]
        diff_hard_gap    = sa["hard_tpr_gap"] - fed["tpr_gap"]   # positive = gap narrowed
        diff_hard_min    = fed["tpr_min"]     - sa["hard_tpr_min"]

        diff_soft_male   = fed["soft_tpr_male"]   - sa["soft_tpr_male"]
        diff_soft_female = fed["soft_tpr_female"] - sa["soft_tpr_female"]
        diff_soft_gap    = sa["soft_tpr_gap"]      - fed["soft_tpr_gap"]   # positive = gap narrowed
        diff_soft_min    = fed["soft_tpr_min"]     - sa["soft_tpr_min"]

        out = {
            "federated_model":           row.get("model_name", ""),
            "tpr_source":                tpr_source,

            "client_0_portion":          row["client_0_portion"],
            "client_0_male_pct":         row["client_0_male_pct"],
            "client_0_female_pct":       row["client_0_female_pct"],
            "client_0_flip_frac":        row["client_0_flip_frac"],

            "client_1_portion":          row["client_1_portion"],
            "client_1_male_pct":         row["client_1_male_pct"],
            "client_1_female_pct":       row["client_1_female_pct"],
            "client_1_flip_frac":        row["client_1_flip_frac"],

            "standalone_model":          _standalone_model_name(c0_key),
            "standalone_found":          standalone_found,

            "diff_nih_auc_macro":        diff_auc,
            "diff_nih_tpr_micro_female": diff_hard_female,
            "diff_nih_tpr_micro_male":   diff_hard_male,
            "diff_tpr_gap":              diff_hard_gap,
            "diff_tpr_min":              diff_hard_min,

            "fed_nih_soft_tpr_micro":         fed["soft_tpr_micro"],
            "diff_nih_soft_tpr_micro_male":   diff_soft_male,
            "diff_nih_soft_tpr_micro_female": diff_soft_female,
            "diff_nih_soft_tpr_gap":          diff_soft_gap,
            "diff_nih_soft_tpr_min":          diff_soft_min,

            "fed_nih_auc_macro":         fed["auc_macro"],
            "fed_nih_tpr_micro_female":  fed["tpr_female"],
            "fed_nih_tpr_micro_male":    fed["tpr_male"],
            "fed_tpr_gap":               fed["tpr_gap"],
            "fed_tpr_min":               fed["tpr_min"],
            "fed_nih_soft_tpr_male":     fed["soft_tpr_male"],
            "fed_nih_soft_tpr_female":   fed["soft_tpr_female"],
            "fed_nih_soft_tpr_gap":      fed["soft_tpr_gap"],
            "fed_nih_soft_tpr_min":      fed["soft_tpr_min"],

            "standalone_nih_auc_macro":        sa["auc_macro"],
            "standalone_nih_tpr_micro_female": sa["hard_tpr_female"],
            "standalone_nih_tpr_micro_male":   sa["hard_tpr_male"],
            "standalone_tpr_gap":              sa["hard_tpr_gap"],
            "standalone_tpr_min":              sa["hard_tpr_min"],
            "standalone_nih_soft_tpr_male":    sa["soft_tpr_male"],
            "standalone_nih_soft_tpr_female":  sa["soft_tpr_female"],
            "standalone_nih_soft_tpr_gap":     sa["soft_tpr_gap"],
            "standalone_nih_soft_tpr_min":     sa["soft_tpr_min"],

            "diff_nih_ba_macro":         diff_ba,
            "fed_nih_ba_macro":          fed["ba_macro"],
            "standalone_nih_ba_macro":   sa["ba_macro"],
        }
        rows_out.append(out)

    print(f"\n📊 Matching summary:")
    print(f"   Matched to standalone : {matched}")
    print(f"   No standalone found   : {unmatched}")
    print(f"   Total rows            : {len(rows_out)}")

    return pd.DataFrame(rows_out)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("🔍  COMPUTING FEDERATION ADVANTAGE FOR CLIENT 0  (2-client setup)")
    print("=" * 70)

    sa_lookup          = load_standalone(STANDALONE_RESULTS_DIR)
    fed_df, tpr_source = load_federated(FEDERATED_RESULTS_DIR)

    result_df = compute_advantages(fed_df, tpr_source, sa_lookup)

    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅  Saved {len(result_df)} rows → {OUTPUT_CSV}")

    diff_cols  = [c for c in result_df.columns if c.startswith("diff_")]
    matched_df = result_df[result_df["standalone_found"] == True]
    if diff_cols and len(matched_df):
        print("\n📈  Advantage statistics (mean ± std, matched rows only):")
        for col in diff_cols:
            vals = matched_df[col].dropna()
            if len(vals):
                print(f"   {col:<35s}  mean={vals.mean():+.4f}  std={vals.std():.4f}")

    print("\n" + "=" * 70)
    print("🏁  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
