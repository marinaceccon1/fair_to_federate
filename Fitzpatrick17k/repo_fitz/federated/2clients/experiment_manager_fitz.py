import json
import fcntl
import sys
from pathlib import Path
from typing import List, Dict

# ---------------------------------------------------------------------------
# Repo root anchored on __file__
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.param_grid_fitz import generate_param_grid

# =============================================================================
# Files
# =============================================================================
EXPERIMENTS_DIR = REPO_ROOT / "experiments_fitz"
EXPERIMENTS_DIR.mkdir(exist_ok=True)
USED_FILE   = EXPERIMENTS_DIR / "used_configs.json"
FAILED_FILE = EXPERIMENTS_DIR / "failed_configs.json"
_LOCK_FILE  = EXPERIMENTS_DIR / ".manager.lock"


# =============================================================================
# Internal lock context manager
# =============================================================================
class _FileLock:
    """
    Exclusive advisory lock using fcntl.flock (Linux/macOS, same filesystem).
    Blocks until the lock is acquired — safe across processes on one machine.
    """
    def __init__(self, path: Path):
        self._path = path
        self._fh   = None

    def __enter__(self):
        self._fh = open(self._path, "a")      # 'a' creates file if absent
        fcntl.flock(self._fh, fcntl.LOCK_EX)  # block until exclusive lock
        return self

    def __exit__(self, *_):
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()


# =============================================================================
# Utilities (all reads/writes go through the lock)
# =============================================================================
def _canonical_key(config: List[Dict]) -> str:
    return json.dumps(config, sort_keys=True)


def _read_used() -> List[str]:
    """Read used config keys. Caller must hold the lock."""
    if USED_FILE.exists():
        try:
            content = USED_FILE.read_text().strip()
            if not content:
                return []
            data = json.loads(content)
            if data and isinstance(data[0], dict):
                return [_canonical_key(cfg) for cfg in data]
            return data
        except (json.JSONDecodeError, ValueError):
            print("⚠️  used_configs.json corrupted, resetting to empty")
            return []
    return []


def _write_used(used: List[str]) -> None:
    """Write used config keys. Caller must hold the lock."""
    USED_FILE.write_text(json.dumps(used, indent=2))


def load_failed() -> List[str]:
    if FAILED_FILE.exists():
        try:
            content = FAILED_FILE.read_text().strip()
            if not content:
                return []
            data = json.loads(content)
            if data and isinstance(data[0], dict):
                return [_canonical_key(cfg) for cfg in data]
            return data
        except (json.JSONDecodeError, ValueError):
            print("⚠️  failed_configs.json corrupted, resetting to empty")
            return []
    return []


def save_failed(failed: List[str]) -> None:
    FAILED_FILE.write_text(json.dumps(failed, indent=2))


# =============================================================================
# Public API
# =============================================================================
def sample_unused_config(num_clients: int, random_order: bool = False) -> List[Dict]:
    """
    Atomically sample ONE unused config AND immediately reserve it by writing
    it to used_configs.json — all inside a single exclusive file lock.

    This guarantees that two parallel workers can never pick the same config.
    On experiment success call mark_config_used() (a no-op / confirmation).
    On experiment failure call release_reserved_config() to make it retryable.

    Raises RuntimeError if all configurations have been used/reserved.
    """
    all_configs = generate_param_grid(num_clients)

    with _FileLock(_LOCK_FILE):
        used     = _read_used()
        used_set = set(used)
        unused   = [cfg for cfg in all_configs if _canonical_key(cfg) not in used_set]

        if not unused:
            raise RuntimeError(
                f"❌ All {len(all_configs)} configurations have been used/reserved."
            )

        chosen = (
            __import__("random").choice(unused) if random_order else unused[0]
        )

        # Reserve immediately — no other worker can pick the same config
        used.append(_canonical_key(chosen))
        _write_used(used)
        print(f"🔒 Config reserved (pool: {len(used)}/{len(all_configs)} used/reserved)")

    return chosen


def release_reserved_config(config: List[Dict]) -> None:
    """
    Remove a previously reserved config from used_configs.json so it can be
    retried. Call this when an experiment fails.
    """
    with _FileLock(_LOCK_FILE):
        used = _read_used()
        key  = _canonical_key(config)
        if key in used:
            used.remove(key)
            _write_used(used)
            print(f"🔓 Config released back to pool ({len(used)} still used/reserved)")
        else:
            print("⚠️  Config not found in used list — nothing to release")


def mark_config_used(config: List[Dict]) -> None:
    """
    Confirm a config as successfully completed.
    sample_unused_config() already reserves the key, so this is mainly a
    confirmation log. Kept for API compatibility.
    """
    with _FileLock(_LOCK_FILE):
        used = _read_used()
        key  = _canonical_key(config)
        if key in used:
            print(f"✅ Config confirmed as completed ({len(used)} total)")
        else:
            used.append(key)
            _write_used(used)
            print(f"✅ Config marked as used ({len(used)} total) [was missing — added]")


def mark_config_failed(config: List[Dict]) -> None:
    """Record a failure for debugging. Does not affect retry — use release_reserved_config()."""
    failed = load_failed()
    key    = _canonical_key(config)
    if key not in failed:
        failed.append(key)
        save_failed(failed)
        print(f"❌ Config marked as failed ({len(failed)} total)")


def num_total_configs(num_clients: int) -> int:
    return len(generate_param_grid(num_clients))


def num_used_configs() -> int:
    with _FileLock(_LOCK_FILE):
        return len(_read_used())


def num_failed_configs() -> int:
    return len(load_failed())


def reset_used_configs() -> None:
    """⚠️ DANGEROUS: wipe the entire used/reserved history."""
    with _FileLock(_LOCK_FILE):
        if USED_FILE.exists():
            USED_FILE.unlink()
            print("🗑️  Used configs cleared")


def reset_failed_configs() -> None:
    """⚠️ DANGEROUS: wipe the failed experiment log."""
    if FAILED_FILE.exists():
        FAILED_FILE.unlink()
        print("🗑️  Failed configs cleared")


def get_progress_summary(num_clients: int) -> Dict:
    total     = num_total_configs(num_clients)
    completed = num_used_configs()
    failed    = num_failed_configs()
    return {
        "total":            total,
        "completed":        completed,
        "failed":           failed,
        "remaining":        total - completed,
        "percent_complete": (completed / total * 100) if total > 0 else 0,
    }
