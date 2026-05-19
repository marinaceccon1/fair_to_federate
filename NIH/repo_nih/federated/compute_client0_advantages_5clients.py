"""
Compute Federation Advantages for Client 0
===========================================
This script computes the advantage for CLIENT 0 when federating with CLIENTS 1 through 4
(i.e. in a 5-client federation).

For each federated model with:
  - Client 0: configuration A
  - Client 1: configuration B
  - Client 2: configuration C
  - Client 3: configuration D
  - Client 4: configuration E
    
We compute:
  Advantage = Federated metrics (Client 0 = config A) - Standalone Client 0 metrics (config A)

This gives us a clear, unambiguous question:
  "What is the advantage for a client with configuration A
   when federating when federating with 4 other clients?""

Metrics computed:
  - AUC increment               (federated - standalone)
  - Balanced Accuracy increment (federated - standalone)
  - Male TPR increment          (federated - standalone)
  - Female TPR increment        (federated - standalone)
  - TPR gap decrease            (standalone_gap - federated_gap, positive = improvement)
  - Minimum TPR increment       (federated - standalone)

OUTPUT FORMAT: Matches the structure of federated_vs_standalone.csv, extended with
               client_2 through client_4 config columns.

Sources read directly (no intermediate CSVs required):
  Standalone → evaluation_results_standalone_client0/new_results_*.json
               (all timestamped JSON files are merged; later runs override earlier
                ones if the same model appears more than once)
  Federated  → evaluation_results_5clients/metrics_auc.csv
               evaluation_results_5clients/metrics_soft_tpr.csv  (or metrics_hard_tpr.csv)
               evaluation_results_5clients/metrics_balanced_accuracy.csv
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION — update these paths as needed
# =============================================================================

FEDERATED_RESULTS_DIR  = Path("evaluation_results_5clients")
STANDALONE_RESULTS_DIR = Path("evaluation_results_standalone_client0")
OUTPUT_CSV             = Path("federation_advantage_client0_5clients.csv")

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

    The JSON structure per entry is:
        {
          "model_path": "...",
          "config": {
              "portion": float,
              "gender": {"Male": float, "Female": float},
              "flip_frac": float,          # config value (1.0 for F100/flip0.9 special case)
              "model_flip_frac": float     # actual value used in training
          },
          "nih_test": {
              "auc":               {"avg": ..., "cumulative_male": ..., "cumulative_female": ...},
              "tpr":               {"cumulative_male": ..., "cumulative_female": ...},
              "balanced_accuracy": {"macro": ..., "micro": ..., ...}
          }
        }
    """
    json_files = sorted(results_dir.glob("new_results_*.json"))

    if not json_files:
        # Also accept a merged full_results.json if someone ran it twice
        fallback = results_dir / "full_results.json"
        if fallback.exists():
            json_files = [fallback]
        else:
            sys.exit(
                f"❌  No standalone result files found in: {results_dir}\n"
                f"    Expected: new_results_*.json  (written by evaluate_standalone_client0.py)"
            )

    print(f"✅  Found {len(json_files)} standalone JSON file(s) in {results_dir}")

    # Merge all files; later timestamp overrides earlier for the same model_path
    raw: dict[str, dict] = {}   # model_path -> result dict
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
        hard_tpr = nih.get("hard_tpr", {})   # was incorrectly "tpr"
        soft_tpr = nih.get("soft_tpr", {})
        ba       = nih.get("balanced_accuracy", {})

        male_tpr   = _pick(hard_tpr, "cumulative_male")
        female_tpr = _pick(hard_tpr, "cumulative_female")
        gap        = (male_tpr - female_tpr) if not (np.isnan(male_tpr) or np.isnan(female_tpr)) else np.nan
        min_tpr    = min(male_tpr, female_tpr) if not (np.isnan(male_tpr) or np.isnan(female_tpr)) else np.nan

        sa_soft_male   = _pick(soft_tpr, "micro_male")
        sa_soft_female = _pick(soft_tpr, "micro_female")
        sa_soft_gap    = _pick(soft_tpr, "gap")
        sa_soft_min    = _pick(soft_tpr, "min")

        lookup[key] = {
            # AUC: "avg" is the macro AUC alias set in compute_auc_sklearn
            "auc_macro":       _pick(auc, "avg", "macro"),
            # BA: "macro" is what create_summary_csv saves as nih_bacc_macro
            "ba_macro":        _pick(ba,  "macro"),
            "tpr_male":        male_tpr,
            "tpr_female":      female_tpr,
            "tpr_gap":         gap,
            "tpr_min":         min_tpr,
            "soft_tpr_male":   sa_soft_male,
            "soft_tpr_female": sa_soft_female,
            "soft_tpr_gap":    sa_soft_gap,
            "soft_tpr_min":    sa_soft_min,
        }

    print(f"   Lookup entries built: {len(lookup)}  (skipped: {skipped})")
    return lookup


# =============================================================================
# LOAD FEDERATED 7-CLIENT RESULTS
# =============================================================================

def _load_fed_csv(path: Path, required: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            sys.exit(f"❌  Federated metrics CSV not found: {path}")
        print(f"⚠️   Optional federated CSV not found, skipping: {path}")
        return None
    df = pd.read_csv(path)
    print(f"✅  Loaded {path.name:<45s} — {len(df)} rows")
    return df


def load_federated(results_dir: Path) -> tuple[pd.DataFrame, str]:
    """
    Load the three main metric CSVs from evaluation_results_5clients/ and merge them
    into a single wide DataFrame, one row per model.

    Returns (merged_df, tpr_source) where tpr_source is "soft" or "hard".
    """
    df_auc = _load_fed_csv(results_dir / "metrics_auc.csv",  required=True)
    df_ba  = _load_fed_csv(results_dir / "metrics_balanced_accuracy.csv", required=True)

    df_tpr = _load_fed_csv(results_dir / "metrics_soft_tpr.csv", required=False)
    if df_tpr is None:
        df_tpr = _load_fed_csv(results_dir / "metrics_hard_tpr.csv", required=True)
        tpr_source = "hard"
    else:
        tpr_source = "soft"

    print(f"   Using {tpr_source} TPR for federated metrics")

    # Include all 7 client config columns
    potential_keys = [
        "model_name", "num_clients",
        "client_0_portion", "client_0_male_pct", "client_0_female_pct", "client_0_flip_frac",
        "client_1_portion", "client_1_male_pct", "client_1_female_pct", "client_1_flip_frac",
        "client_2_portion", "client_2_male_pct", "client_2_female_pct", "client_2_flip_frac",
        "client_3_portion", "client_3_male_pct", "client_3_female_pct", "client_3_flip_frac",
        "client_4_portion", "client_4_male_pct", "client_4_female_pct", "client_4_flip_frac",
    ]
    key_cols = [c for c in potential_keys if c in df_auc.columns]

    def _drop_dup_non_key(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(
            columns=[c for c in df.columns if c in df_auc.columns and c not in key_cols],
            errors="ignore"
        )

    merged = df_auc.copy()
    for df_extra in (df_ba, df_tpr):
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

    if tpr_source == "soft":
        micro  = _pick_row(row, "nih_soft_tpr_micro")
        male   = _pick_row(row, "nih_soft_tpr_micro_male")
        female = _pick_row(row, "nih_soft_tpr_micro_female")
        gap    = _pick_row(row, "nih_soft_tpr_gap")
        min_t  = _pick_row(row, "nih_soft_tpr_min")
    else:
        micro  = np.nan
        male   = _pick_row(row, "nih_hard_tpr_micro_male")
        female = _pick_row(row, "nih_hard_tpr_micro_female")
        gap    = np.nan
        min_t  = np.nan

    if np.isnan(gap)   and not (np.isnan(male) or np.isnan(female)):
        gap   = male - female
    if np.isnan(min_t) and not (np.isnan(male) or np.isnan(female)):
        min_t = min(male, female)

    return {"auc_macro": auc, "ba_macro": ba,
            "soft_tpr_micro": micro,
            "tpr_male": male, "tpr_female": female,
            "tpr_gap": gap,   "tpr_min": min_t}


# =============================================================================
# BUILD STANDALONE MODEL NAME
# =============================================================================

def _standalone_model_name(key: tuple) -> str:
    """
    Matches the naming convention used in federated_vs_standalone.csv:
      standalone_p<portion>_M<male_int>_flip<flip>
    """
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
            sa = {k: np.nan for k in ("auc_macro", "ba_macro", "tpr_male", "tpr_female", "tpr_gap", "tpr_min",
                                       "soft_tpr_male", "soft_tpr_female", "soft_tpr_gap", "soft_tpr_min")}
            standalone_found = False
        else:
            matched += 1
            standalone_found = True

        fed = _fed_metrics(row, tpr_source)

        diff_auc         = fed["auc_macro"]   - sa["auc_macro"]
        diff_ba          = fed["ba_macro"]     - sa["ba_macro"]
        diff_male        = fed["tpr_male"]     - sa["tpr_male"]
        diff_female      = fed["tpr_female"]   - sa["tpr_female"]
        diff_gap         = sa["tpr_gap"]       - fed["tpr_gap"]   # positive = gap narrowed
        diff_min         = fed["tpr_min"]      - sa["tpr_min"]
        diff_soft_male   = fed["tpr_male"]     - sa["soft_tpr_male"]
        diff_soft_female = fed["tpr_female"]   - sa["soft_tpr_female"]
        diff_soft_gap    = (sa["soft_tpr_gap"] - fed["tpr_gap"]
                            if not (np.isnan(sa["soft_tpr_gap"]) or np.isnan(fed["tpr_gap"]))
                            else np.nan)
        diff_soft_min    = fed["tpr_min"]      - sa["soft_tpr_min"]

        out = {
            "federated_model": row.get("model_name", ""),

            # Client 0 config
            "client_0_portion":    row["client_0_portion"],
            "client_0_male_pct":   row["client_0_male_pct"],
            "client_0_female_pct": row["client_0_female_pct"],
            "client_0_flip_frac":  row["client_0_flip_frac"],

            # Client 1 config
            "client_1_portion":    row["client_1_portion"],
            "client_1_male_pct":   row["client_1_male_pct"],
            "client_1_female_pct": row["client_1_female_pct"],
            "client_1_flip_frac":  row["client_1_flip_frac"],

            # Client 2 config
            "client_2_portion":    row.get("client_2_portion",    np.nan),
            "client_2_male_pct":   row.get("client_2_male_pct",   np.nan),
            "client_2_female_pct": row.get("client_2_female_pct", np.nan),
            "client_2_flip_frac":  row.get("client_2_flip_frac",  np.nan),

            # Client 3 config
            "client_3_portion":    row.get("client_3_portion",    np.nan),
            "client_3_male_pct":   row.get("client_3_male_pct",   np.nan),
            "client_3_female_pct": row.get("client_3_female_pct", np.nan),
            "client_3_flip_frac":  row.get("client_3_flip_frac",  np.nan),

            # Client 4 config
            "client_4_portion":    row.get("client_4_portion",    np.nan),
            "client_4_male_pct":   row.get("client_4_male_pct",   np.nan),
            "client_4_female_pct": row.get("client_4_female_pct", np.nan),
            "client_4_flip_frac":  row.get("client_4_flip_frac",  np.nan),

            "standalone_model": _standalone_model_name(c0_key),
            "standalone_found":          standalone_found,

            # diff_* — same naming as federated_vs_standalone.csv
            "diff_nih_auc_macro":        diff_auc,
            "diff_nih_tpr_micro_female": diff_female,
            "diff_nih_tpr_micro_male":   diff_male,
            "diff_tpr_gap":              diff_gap,
            "diff_tpr_min":              diff_min,

            # soft TPR columns — diffs use standalone soft-TPR as the baseline
            "fed_nih_soft_tpr_micro":         fed["soft_tpr_micro"],
            "diff_nih_soft_tpr_micro_male":   diff_soft_male,
            "diff_nih_soft_tpr_micro_female": diff_soft_female,
            "diff_nih_soft_tpr_gap":          diff_soft_gap,
            "diff_nih_soft_tpr_min":          diff_soft_min,

            # raw federated
            "fed_nih_auc_macro":         fed["auc_macro"],
            "fed_nih_tpr_micro_female":  fed["tpr_female"],
            "fed_nih_tpr_micro_male":    fed["tpr_male"],
            "fed_tpr_gap":               fed["tpr_gap"],
            "fed_tpr_min":               fed["tpr_min"],

            # raw standalone
            "standalone_nih_auc_macro":        sa["auc_macro"],
            "standalone_nih_tpr_micro_female": sa["tpr_female"],
            "standalone_nih_tpr_micro_male":   sa["tpr_male"],
            "standalone_tpr_gap":              sa["tpr_gap"],
            "standalone_tpr_min":              sa["tpr_min"],

            # balanced accuracy — same naming as federated_vs_standalone.csv
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
    print("🔍  COMPUTING FEDERATION ADVANTAGE FOR CLIENT 0  (5-CLIENT SETUP)")
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
