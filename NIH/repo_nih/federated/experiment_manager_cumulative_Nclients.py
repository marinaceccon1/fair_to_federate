"""
experiment_manager_cumulative_Nclients.py
==========================================
Experiment tracking and stratified sampling for the N-client cumulative
federated learning setup.

NUM_CLIENTS is read from the environment variable NUM_CLIENTS (default: 7).
Tracking files are named with the client count so that 3/5/7-client
experiment histories never collide.

Usage
-----
    export NUM_CLIENTS=5
    python experiment_manager_cumulative_Nclients.py  # self-test
"""

import fcntl
import json
import os
import random as _random
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure federated/ is on sys.path so the bare sibling import below works
# regardless of CWD.
# ---------------------------------------------------------------------------
import sys as _sys
from typing import Dict, List, Optional, Tuple
_FEDERATED_DIR = Path(__file__).resolve().parent
_REPO_ROOT     = _FEDERATED_DIR.parent.parent
if str(_FEDERATED_DIR) not in _sys.path:
    _sys.path.insert(0, str(_FEDERATED_DIR))

from param_grid_cumulative_Nclients import (
    NUM_CLIENTS,
    generate_cumulative_param_grid,
    find_matching_client_configs,
)

# =============================================================================
# File paths  (keyed by client count so histories never collide)
# =============================================================================
EXPERIMENTS_DIR = _REPO_ROOT / "repo_nih" / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)


def _get_used_file(num_clients: int) -> Path:
    return EXPERIMENTS_DIR / f"used_configs_{num_clients}clients_cumulative.json"


def _get_failed_file(num_clients: int) -> Path:
    return EXPERIMENTS_DIR / f"failed_configs_{num_clients}clients_cumulative.json"


def _lock_path(num_clients: int) -> Path:
    return EXPERIMENTS_DIR / f"used_configs_{num_clients}clients_cumulative.lock"


# =============================================================================
# Stratum key  (client_0 uniquely defines the stratum)
# =============================================================================
def _stratum_key(config: Dict) -> Tuple:
    c0 = config["client_0"]
    return (c0["portion"], c0["gender"]["Male"], c0["flip_frac"])


# =============================================================================
# Persistence helpers
# =============================================================================
def load_used(num_clients: int) -> List[str]:
    """Load used config keys — call only while holding the file lock."""
    f = _get_used_file(num_clients)
    if f.exists():
        try:
            content = f.read_text().strip()
            return json.loads(content) if content else []
        except (json.JSONDecodeError, ValueError):
            print(f"⚠️  Warning: {f} is corrupted, resetting to empty")
    return []


def save_used(num_clients: int, used: List[str]) -> None:
    _get_used_file(num_clients).write_text(json.dumps(used, indent=2))


def load_failed(num_clients: int) -> List[str]:
    f = _get_failed_file(num_clients)
    if f.exists():
        try:
            content = f.read_text().strip()
            return json.loads(content) if content else []
        except (json.JSONDecodeError, ValueError):
            print(f"⚠️  Warning: {f} is corrupted, resetting to empty")
    return []


def save_failed(num_clients: int, failed: List[str]) -> None:
    _get_failed_file(num_clients).write_text(json.dumps(failed, indent=2))


# =============================================================================
# Canonical key
# =============================================================================
def _canonical_key(config: Dict) -> str:
    canonical = {
        "client_0":           config["client_0"],
        "cumulative_metrics": config["cumulative_metrics"],
        "num_other_clients":  config["num_other_clients"],
    }
    return json.dumps(canonical, sort_keys=True)


# =============================================================================
# Core sampler — stratified uniform
# =============================================================================
def sample_unused_config(
    num_clients: int,
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict:
    """
    Return ONE unused configuration using stratified uniform sampling.

    The space is stratified by the client_0 configuration (one stratum per
    distinct client_0 triple).  At each call:
      1. Collect all unused configs, grouped by stratum.
      2. Pick one non-empty stratum uniformly at random.
      3. Pick one config uniformly at random within that stratum.

    The chosen config is atomically pre-reserved (written to the used-list
    under a file lock) before returning, so parallel workers are safe.
    """
    rng = _random.Random(seed)
    all_configs = generate_cumulative_param_grid(num_clients)

    lock_file = _lock_path(num_clients)
    lock_fh   = open(lock_file, "a")
    fcntl.flock(lock_fh, fcntl.LOCK_EX)
    try:
        used_keys = set(load_used(num_clients))

        if max_samples is not None and len(used_keys) >= max_samples:
            raise RuntimeError(
                f"✅ Reached max_samples limit: {max_samples} configurations completed."
            )

        unused = [c for c in all_configs if _canonical_key(c) not in used_keys]
        if not unused:
            raise RuntimeError(
                f"✅ All {len(all_configs)} configurations have been used."
            )

        # Group by client_0 stratum
        by_stratum: Dict[Tuple, List[Dict]] = {}
        for cfg in unused:
            by_stratum.setdefault(_stratum_key(cfg), []).append(cfg)

        # Stratified uniform draw
        chosen_stratum = rng.choice(list(by_stratum.keys()))
        chosen = rng.choice(by_stratum[chosen_stratum])

        # Pre-reserve atomically
        used_keys.add(_canonical_key(chosen))
        save_used(num_clients, list(used_keys))

    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()

    return chosen


# =============================================================================
# Expand cumulative config → per-client list
# =============================================================================
def expand_config_to_individual_clients(
    config: Dict,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Expand a cumulative configuration into one dict per client.
    Returns [client_0_config, client_1_config, ..., client_{N-1}_config].
    """
    full_config = [config["client_0"]]
    others = find_matching_client_configs(
        config["cumulative_metrics"],
        config["num_other_clients"],
        seed=seed,
    )
    full_config.extend(others)
    return full_config


# =============================================================================
# Mark used / failed
# =============================================================================
def mark_config_used(config: Dict) -> None:
    """Confirm a config as successfully completed (idempotent)."""
    nc  = 1 + config["num_other_clients"]
    key = _canonical_key(config)

    lock_fh = open(_lock_path(nc), "a")
    fcntl.flock(lock_fh, fcntl.LOCK_EX)
    try:
        used = load_used(nc)
        if key not in used:
            used.append(key)
            save_used(nc, used)
            print(f"✅ Configuration marked as used ({len(used)} total)")
        else:
            print(f"✅ Configuration already reserved ({len(used)} total)")
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


def mark_config_failed(config: Dict) -> None:
    """Log a failed config (not skipped on retry)."""
    nc     = 1 + config["num_other_clients"]
    failed = load_failed(nc)
    key    = _canonical_key(config)
    if key not in failed:
        failed.append(key)
        save_failed(nc, failed)
        print(f"❌ Configuration marked as failed ({len(failed)} total)")


# =============================================================================
# Counting helpers
# =============================================================================
def num_total_configs(num_clients: int) -> int:
    return len(generate_cumulative_param_grid(num_clients))


def num_used_configs(num_clients: int) -> int:
    return len(load_used(num_clients))


def num_failed_configs(num_clients: int) -> int:
    return len(load_failed(num_clients))


# =============================================================================
# Progress summary
# =============================================================================
def get_progress_summary(
    num_clients: int,
    max_samples: Optional[int] = None,
) -> Dict:
    all_configs = generate_cumulative_param_grid(num_clients)
    used_keys   = set(load_used(num_clients))
    completed   = len(used_keys)
    failed      = num_failed_configs(num_clients)
    total       = len(all_configs)

    if max_samples is not None:
        target    = min(max_samples, total)
        remaining = max(0, target - completed)
        percent   = (completed / target * 100) if target > 0 else 0.0
    else:
        target    = total
        remaining = total - completed
        percent   = (completed / total * 100) if total > 0 else 0.0

    total_by_s: Dict[Tuple, int] = {}
    used_by_s:  Dict[Tuple, int] = {}
    for cfg in all_configs:
        sk = _stratum_key(cfg)
        total_by_s[sk] = total_by_s.get(sk, 0) + 1
        if _canonical_key(cfg) in used_keys:
            used_by_s[sk] = used_by_s.get(sk, 0) + 1

    stratum_counts = {
        str(sk): {
            "total": total_by_s[sk],
            "used":  used_by_s.get(sk, 0),
            "pct":   round(used_by_s.get(sk, 0) / total_by_s[sk] * 100, 1)
                     if total_by_s[sk] > 0 else 0.0,
        }
        for sk in total_by_s
    }

    return {
        "total":            total,
        "target":           target,
        "completed":        completed,
        "failed":           failed,
        "remaining":        remaining,
        "percent_complete": round(percent, 1),
        "stratum_counts":   stratum_counts,
    }


# =============================================================================
# Reset helpers
# =============================================================================
def reset_used_configs(num_clients: int) -> None:
    f = _get_used_file(num_clients)
    if f.exists():
        f.unlink()
        print(f"🗑️  Used configs cleared for {num_clients} clients")


def reset_failed_configs(num_clients: int) -> None:
    f = _get_failed_file(num_clients)
    if f.exists():
        f.unlink()
        print(f"🗑️  Failed configs cleared for {num_clients} clients")


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    import unittest.mock as _mock

    print("=" * 70)
    print(f"EXPERIMENT MANAGER — {NUM_CLIENTS}-CLIENT STRATIFIED UNIFORM SAMPLING TEST")
    print("=" * 70)

    all_cfgs = generate_cumulative_param_grid(NUM_CLIENTS)
    total    = len(all_cfgs)

    by_stratum: Dict[Tuple, int] = {}
    for cfg in all_cfgs:
        sk = _stratum_key(cfg)
        by_stratum[sk] = by_stratum.get(sk, 0) + 1

    print(f"\nConfig space ({NUM_CLIENTS} clients): {total:,} total")
    print(f"Number of client_0 strata : {len(by_stratum)}")
    sizes = list(by_stratum.values())
    print(f"Configs per stratum       : min={min(sizes)}, max={max(sizes)}, "
          f"avg={sum(sizes)/len(sizes):.1f}")

    N_DRAWS = len(by_stratum) * 10
    print(f"\nSimulating {N_DRAWS} draws (empty used set):")
    draw_counts: Dict[Tuple, int] = {}
    with _mock.patch(__name__ + ".load_used", return_value=[]):
        for i in range(N_DRAWS):
            cfg = sample_unused_config(NUM_CLIENTS, seed=i)
            sk  = _stratum_key(cfg)
            draw_counts[sk] = draw_counts.get(sk, 0) + 1

    uniform_pct = 100.0 / len(by_stratum)
    print(f"\n  {'stratum (portion, male_pct, flip)':40s}  {'drawn':>6}  {'draw%':>6}  {'uniform%':>9}")
    print(f"  {'-'*65}")
    for sk in sorted(by_stratum.keys()):
        n = draw_counts.get(sk, 0)
        print(f"  {str(sk):40s}  {n:>6}  {n/N_DRAWS*100:>5.1f}%  {uniform_pct:>8.1f}%")

    s = get_progress_summary(NUM_CLIENTS, max_samples=200)
    print(f"\nProgress summary: {s['completed']}/{s['target']} ({s['percent_complete']}%)")
    print("\n✅ Self-test complete!")
