"""
experiment_manager_2clients.py
================================
Tracks which of the 729 2-client configurations have been completed.

Design notes
------------
* Uses file locking (fcntl) — safe for multiple parallel workers.
* Configs are stored as canonical JSON keys (deterministic across runs).
* Pre-reserves a config atomically before returning it, preventing two
  workers from running the same experiment simultaneously.
* State files live in  experiments/  relative to the repo root.
"""

import fcntl
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — works regardless of cwd
# ---------------------------------------------------------------------------
_DIR       = Path(__file__).resolve().parent          # federated_2clients/
_REPO_ROOT = _DIR.parent.parent                       # NIH/repo/ → NIH/

import sys
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from param_grid_2clients import generate_param_grid

# ---------------------------------------------------------------------------
# State files
# ---------------------------------------------------------------------------
_EXPERIMENTS_DIR = _REPO_ROOT / "repo_nih" / "experiments"
_EXPERIMENTS_DIR.mkdir(exist_ok=True)

USED_FILE    = _EXPERIMENTS_DIR / "used_configs_2clients.json"
FAILED_FILE  = _EXPERIMENTS_DIR / "failed_configs_2clients.json"
LOCK_FILE    = _EXPERIMENTS_DIR / "used_configs_2clients.lock"
RESERVED_FILE = _EXPERIMENTS_DIR / "reserved_configs_2clients.json"


# =============================================================================
# Canonical key
# =============================================================================

def _canonical_key(config: List[Dict]) -> str:
    """Deterministic, hashable representation of a 2-client config."""
    return json.dumps(config, sort_keys=True)


# =============================================================================
# Low-level I/O  (always called with the file lock held)
# =============================================================================

def _read_keys(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        content = path.read_text().strip()
        return json.loads(content) if content else []
    except (json.JSONDecodeError, ValueError):
        print(f"⚠️  {path.name} is corrupted — resetting to empty")
        return []


def _write_keys(path: Path, keys: List[str]) -> None:
    path.write_text(json.dumps(keys, indent=2))


# =============================================================================
# Public API
# =============================================================================

def sample_unused_config() -> List[Dict]:
    """
    Atomically reserve and return ONE unused configuration.

    The config is immediately written to the reserved list so that
    concurrent workers will not pick the same experiment.

    Raises:
        RuntimeError: if all 729 configurations are already done/reserved.
    """
    all_configs = generate_param_grid()

    with open(LOCK_FILE, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            used_keys     = set(_read_keys(USED_FILE))
            reserved_keys = set(_read_keys(RESERVED_FILE))
            blocked       = used_keys | reserved_keys

            unused = [c for c in all_configs if _canonical_key(c) not in blocked]

            if not unused:
                raise RuntimeError(
                    f"❌ All {len(all_configs)} configurations are completed or reserved."
                )

            chosen = unused[0]
            key    = _canonical_key(chosen)

            reserved = _read_keys(RESERVED_FILE)
            reserved.append(key)
            _write_keys(RESERVED_FILE, reserved)

        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)

    return chosen


def mark_config_used(config: List[Dict]) -> None:
    """
    Mark a configuration as successfully completed and release its reservation.
    Call this ONLY after a successful experiment.
    """
    key = _canonical_key(config)

    with open(LOCK_FILE, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            used = _read_keys(USED_FILE)
            if key not in used:
                used.append(key)
                _write_keys(USED_FILE, used)

            reserved = [k for k in _read_keys(RESERVED_FILE) if k != key]
            _write_keys(RESERVED_FILE, reserved)

        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)

    print(f"✅ Configuration marked as used")


def release_reservation(config: List[Dict]) -> None:
    """
    Release a reservation without marking the config as used.
    Call this when an experiment fails so the config can be retried.
    """
    key = _canonical_key(config)

    with open(LOCK_FILE, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            reserved = [k for k in _read_keys(RESERVED_FILE) if k != key]
            _write_keys(RESERVED_FILE, reserved)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def mark_config_failed(config: List[Dict]) -> None:
    """
    Log a failed configuration (for debugging). Does NOT prevent retries.
    Also releases the reservation.
    """
    key = _canonical_key(config)

    with open(LOCK_FILE, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            failed = _read_keys(FAILED_FILE)
            if key not in failed:
                failed.append(key)
                _write_keys(FAILED_FILE, failed)

            reserved = [k for k in _read_keys(RESERVED_FILE) if k != key]
            _write_keys(RESERVED_FILE, reserved)

        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)

    print(f"❌ Configuration marked as failed")


def get_progress_summary() -> Dict:
    """Return a snapshot of experiment progress."""
    total     = len(generate_param_grid())
    completed = len(_read_keys(USED_FILE))
    failed    = len(_read_keys(FAILED_FILE))
    reserved  = len(_read_keys(RESERVED_FILE))
    remaining = total - completed - reserved

    return {
        "total":            total,
        "completed":        completed,
        "failed":           failed,
        "reserved":         reserved,
        "remaining":        remaining,
        "percent_complete": completed / total * 100 if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Convenience counters (used by run_experiments_2clients.py)
# ---------------------------------------------------------------------------

def num_total_configs()   -> int: return len(generate_param_grid())
def num_used_configs()    -> int: return len(_read_keys(USED_FILE))
def num_failed_configs()  -> int: return len(_read_keys(FAILED_FILE))


# ---------------------------------------------------------------------------
# Dangerous resets (for development use only)
# ---------------------------------------------------------------------------

def reset_used_configs() -> None:
    if USED_FILE.exists():
        USED_FILE.unlink()
        print("🗑️  Used configs cleared")

def reset_failed_configs() -> None:
    if FAILED_FILE.exists():
        FAILED_FILE.unlink()
        print("🗑️  Failed configs cleared")

def reset_reserved_configs() -> None:
    if RESERVED_FILE.exists():
        RESERVED_FILE.unlink()
        print("🗑️  Reserved configs cleared")
