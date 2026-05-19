import json
import subprocess
import sys
import fcntl
import time
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root: three levels up from this file (NIH/repo/standalone/run_experiments.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from src.param_grid import generate_param_grid

CONFIG_DIR       = REPO_ROOT / "repo_nih" / "experiments" / "unique_configs"
EPOCHS_PER_MODEL = 20
SAVE_DIR         = REPO_ROOT / "repo_nih" / "single_models"
LOCK_FILE        = REPO_ROOT / "repo_nih" / "experiments" / "unique_experiment.lock"
COMPLETED_FILE   = REPO_ROOT / "repo_nih" / "experiments" / "unique_configs_completed.json"


def generate_all_unique_configs():
    """
    Generate all unique single-client configurations by extracting
    the distinct client configs from the 1-client param grid.
    """
    seen = set()
    configs = []
    for experiment in generate_param_grid(num_clients=1):
        config = experiment[0]
        key = (config["portion"], config["gender"]["Male"], config["flip_frac"])
        if key not in seen:
            seen.add(key)
            configs.append(config)
    return configs


def config_to_id(config):
    """Convert a config dict to a unique string ID."""
    return (f"p{config['portion']}_M{int(config['gender']['Male']*100)}"
            f"F{int(config['gender']['Female']*100)}_flip{config['flip_frac']}")


def load_completed_configs():
    if COMPLETED_FILE.exists():
        with open(COMPLETED_FILE) as f:
            return set(json.load(f))
    return set()


def save_completed_config(config_id: str):
    completed = load_completed_configs()
    completed.add(config_id)
    with open(COMPLETED_FILE, 'w') as f:
        json.dump(sorted(list(completed)), f, indent=2)


def acquire_lock(timeout: int = 300):
    lock_fd = open(LOCK_FILE, 'w')
    start   = time.time()
    while True:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_fd
        except IOError:
            if time.time() - start > timeout:
                raise TimeoutError("Could not acquire lock within timeout")
            time.sleep(0.5)


def release_lock(lock_fd):
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    lock_fd.close()


# =============================================================================
# MAIN
# =============================================================================
all_configs = generate_all_unique_configs()

print("\n" + "=" * 70)
print("🔬 UNIQUE CONFIG EXPERIMENT RUNNER")
print("=" * 70)
print(f"Total configurations to train: {len(all_configs)}")
print(f"  Breakdown:")
print(f"    - 100% Male (M=1.0, F=0.0): 3 portions × 1 flip (0.0) = 3 configs")
print(f"    - 50/50 Mix (M=0.5, F=0.5): 3 portions × 4 flips = 12 configs")
print(f"    - 100% Female (M=0.0, F=1.0): 3 portions × 4 flips = 12 configs")
print(f"Already completed: {len(load_completed_configs())}")
print(f"Save directory: {SAVE_DIR}")
print(f"Progress tracked in: {COMPLETED_FILE.name}")
print("=" * 70 + "\n")

for d in [SAVE_DIR, CONFIG_DIR, LOCK_FILE.parent]:
    d.mkdir(parents=True, exist_ok=True)
print("✅ Directories ready\n")

train_script     = Path(__file__).resolve().parent / "train_single_config.py"
experiment_count = 0

while True:
    experiment_count += 1

    print("🔒 Acquiring lock...")
    try:
        lock_fd = acquire_lock()
    except TimeoutError:
        print("⚠️  Could not acquire lock, retrying...")
        continue

    print("✅ Lock acquired")

    completed_ids     = load_completed_configs()
    remaining_configs = [c for c in all_configs if config_to_id(c) not in completed_ids]

    if not remaining_configs:
        print("\n✅ ALL CONFIGURATIONS COMPLETED!")
        release_lock(lock_fd)
        break

    config    = remaining_configs[0]
    config_id = config_to_id(config)

    pid         = os.getpid()
    config_path = CONFIG_DIR / f"config_{pid}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    release_lock(lock_fd)
    print("🔓 Lock released\n")

    print(f"\n{'='*70}")
    print(f"🧪 EXPERIMENT {experiment_count} (PID: {pid})")
    print(f"{'='*70}")
    print(f"  Config ID: {config_id}")
    print(f"  Portion:   {config['portion']}")
    print(f"  Gender:    M={config['gender']['Male']:.2f}, F={config['gender']['Female']:.2f}")
    print(f"  Flip:      {config['flip_frac']}")
    print("=" * 70 + "\n")

    expected_filename = SAVE_DIR / f"best_model__{config_id}.pt"
    if expected_filename.exists():
        print(f"⚠️  Model already exists: {expected_filename.name} — skipping")
        lock_fd = acquire_lock()
        save_completed_config(config_id)
        release_lock(lock_fd)
        config_path.unlink(missing_ok=True)
        continue

    success = False
    try:
        subprocess.run(
            [sys.executable, str(train_script),
             "--config", str(config_path),
             "--epochs", str(EPOCHS_PER_MODEL)],
            check=True,
            timeout=3600
        )
        print("✅ Training completed successfully")
        success = True

    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with return code {e.returncode}")

    except subprocess.TimeoutExpired:
        print("⚠️  Training exceeded timeout")

    except KeyboardInterrupt:
        print("\n\n⚠️  Keyboard interrupt — stopping")
        config_path.unlink(missing_ok=True)
        sys.exit(0)

    config_path.unlink(missing_ok=True)

    lock_fd = acquire_lock()
    if success:
        save_completed_config(config_id)
        print("📊 Configuration marked as completed")
    else:
        print("⚠️  Configuration NOT marked as completed (skipping to next)")
        release_lock(lock_fd)
        continue

    release_lock(lock_fd)
    print("🔓 Lock released")

    completed_ids = load_completed_configs()
    print(f"\n{'='*70}")
    print(f"Progress: {len(completed_ids)}/{len(all_configs)} "
          f"({100*len(completed_ids)/len(all_configs):.1f}%)")
    print(f"{'='*70}\n")

print("\n" + "=" * 70)
print("🏁 EXPERIMENT RUNNER FINISHED")
completed_ids = load_completed_configs()
print(f"Completed: {len(completed_ids)}/{len(all_configs)}")
print(f"All models saved to: {SAVE_DIR}")
print("=" * 70)
