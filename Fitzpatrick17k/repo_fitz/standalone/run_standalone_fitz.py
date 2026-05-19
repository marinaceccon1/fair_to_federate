"""
run_standalone_fitz.py
───────────────────────
Iterates over the standalone param grid (num_clients=1) and launches
train_standalone_fitz.py as a subprocess for each untried configuration.

Mirrors the NIH standalone run_experiments.py pattern:
  • Config sampled and reserved atomically via experiment_manager_fitz
  • train_standalone_fitz.py runs in a subprocess with CONFIG_PATH + DATA_SEED
    forwarded via environment variables
  • On success: config confirmed as completed
  • On failure: config released back to the pool for retry (no interactive prompt)
  • Multiple workers can run in parallel (distinct WORKER_ID → distinct config files)

Usage:
    python standalone/run_standalone_fitz.py [--worker-id 0]
    python standalone/run_standalone_fitz.py [--worker-id 1]  # parallel worker
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root anchored on __file__
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANAGER_DIR = _REPO_ROOT / "federated" / "2clients"
for _p in (str(_REPO_ROOT), str(_MANAGER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_manager_fitz import (
    sample_unused_config,
    mark_config_used,
    release_reserved_config,
    mark_config_failed,
    num_total_configs,
    num_used_configs,
)

# =============================================================================
# SETTINGS
# =============================================================================
NUM_CLIENTS  = 1       # standalone always trains a single client
DATA_SEED    = 42
RANDOM_ORDER = True    # set False for sequential iteration
MAX_CONSECUTIVE_FAILURES = 5

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(description="Fitzpatrick standalone experiment runner")
parser.add_argument(
    "--worker-id", type=int, default=0,
    help="Unique integer ID for this runner process (0, 1, 2, …). "
         "Controls the per-worker config file so multiple terminals can run "
         "in parallel without collision."
)
args = parser.parse_args()

WORKER_ID   = args.worker_id
CONFIG_PATH = _REPO_ROOT / "experiments_standalone" / f"current_config_worker{WORKER_ID}.json"
SAVE_DIR    = _REPO_ROOT / "best_models_fitz"

# Environment variables forwarded to every subprocess spawned by this worker
WORKER_ENV = {
    **os.environ,
    "CONFIG_PATH": str(CONFIG_PATH),
    "DATA_SEED":   str(DATA_SEED),
    "SAVE_DIR":    str(SAVE_DIR),
}

print("\n" + "=" * 70)
print("🔬 FITZPATRICK STANDALONE EXPERIMENT RUNNER")
print("=" * 70)
print(f"Worker ID     : {WORKER_ID}")
print(f"Config file   : {CONFIG_PATH}")
print(f"Save dir      : {SAVE_DIR}")
print(f"Sampling mode : {'RANDOM' if RANDOM_ORDER else 'SEQUENTIAL'}")
print(f"Data seed     : {DATA_SEED}")
print(f"Total configs : {num_total_configs(NUM_CLIENTS)}")
print(f"Completed     : {num_used_configs()}")
print(f"Remaining     : {num_total_configs(NUM_CLIENTS) - num_used_configs()}")
print("=" * 70 + "\n")

experiment_count      = 0
consecutive_failures  = 0

# =============================================================================
# MAIN LOOP
# =============================================================================
while True:
    experiment_count += 1

    # ── Sample + reserve next config (atomic, lock-protected) ─────────────────
    try:
        config = sample_unused_config(NUM_CLIENTS, random_order=RANDOM_ORDER)
    except RuntimeError:
        print("\n✅ ALL CONFIGURATIONS COMPLETED!")
        break

    print(f"\n{'=' * 70}")
    print(f"🧪 EXPERIMENT {experiment_count}  [worker {WORKER_ID}]")
    print(f"{'=' * 70}")
    print(f"  portion={config[0]['portion']:.2f}, "
          f"composition={config[0]['composition']}, "
          f"flip_frac={config[0]['flip_frac']}")
    print("=" * 70 + "\n")

    # ── Write per-worker config file ──────────────────────────────────────────
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📝 Config written to {CONFIG_PATH}")

    success = False

    try:
        print(f"🚀 Launching train_standalone_fitz.py ...")
        print("─" * 70)

        proc = subprocess.run(
            [sys.executable,
             str(_REPO_ROOT / "standalone" / "train_standalone_fitz.py")],
            env=WORKER_ENV,
            cwd=str(_REPO_ROOT),
        )

        success = (proc.returncode == 0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Keyboard interrupt detected!")
        release_reserved_config(config)
        print("🛑 Stopping experiment runner")
        break

    except Exception as e:
        import traceback
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        success = False

    # ── Handle result ─────────────────────────────────────────────────────────
    if success:
        print("\n✅ Training completed successfully!")
        mark_config_used(config)
        consecutive_failures = 0
        print("📊 Configuration confirmed as completed\n")
    else:
        print(f"\n❌ Training failed (exit code: {proc.returncode if 'proc' in dir() else 'N/A'})")
        mark_config_failed(config)
        release_reserved_config(config)
        consecutive_failures += 1
        print(f"🔁 Config released — will be retried "
              f"(failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})\n")

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"\n🛑 {MAX_CONSECUTIVE_FAILURES} consecutive failures — stopping.")
            break

    print(f"\n{'=' * 70}")
    print(f"[Worker {WORKER_ID}] Progress: "
          f"{num_used_configs()}/{num_total_configs(NUM_CLIENTS)} "
          f"({100 * num_used_configs() / num_total_configs(NUM_CLIENTS):.1f}%)")
    print(f"{'=' * 70}\n")

    time.sleep(1)

print("\n" + "=" * 70)
print(f"🏁 EXPERIMENT RUNNER FINISHED  [worker {WORKER_ID}]")
print(f"Completed: {num_used_configs()}/{num_total_configs(NUM_CLIENTS)}")
print("=" * 70)
