"""
param_grid_cumulative_Nclients.py
==================================
Parameter grid for the N-client cumulative federated learning setup.

NUM_CLIENTS is read from the environment variable NUM_CLIENTS (default: 7).
The module validates that num_total_clients matches NUM_CLIENTS when
generate_cumulative_param_grid() is called.

Caching
-------
generate_all_cumulative_configs() is the expensive step (enumerates all
combinations for N-1 other clients). Its result is pickled to
experiments/cache/cumulative_configs_<N-1>clients.pkl on first run and
loaded from disk on every subsequent call, making startup essentially instant.

To force a rebuild (e.g. after changing PORTIONS / FLIP_FRACTIONS), delete
that file and re-run.
"""

import itertools
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# =============================================================================
# NUM_CLIENTS from environment
# =============================================================================
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "7"))

# =============================================================================
# Disk cache helpers
# =============================================================================
_CACHE_DIR = Path("experiments/cache")


def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{key}.pkl"


def _load_cache(key: str):
    p = _cache_path(key)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(key: str, obj) -> None:
    with open(_cache_path(key), "wb") as f:
        pickle.dump(obj, f)


# =============================================================================
# Allowed hyper-parameter values
# =============================================================================
PORTIONS = [0.12, 0.06, 0.03]
GENDER_PROPORTIONS = [
    {"Male": 1.0, "Female": 0.0},
    {"Male": 0.5, "Female": 0.5},
    {"Male": 0.0, "Female": 1.0},
]
FLIP_FRACTIONS = [0.0, 0.15, 0.30, 0.45]


# =============================================================================
# Single-client helpers
# =============================================================================
def is_valid_client_config(portion, gender, flip_frac) -> bool:
    """A config is invalid only when there are no women but flip_frac > 0."""
    if gender == {"Male": 1.0, "Female": 0.0} and flip_frac != 0.0:
        return False
    return True


def get_all_valid_single_configs() -> List[Tuple]:
    """Return every valid (portion, gender_dict, flip_frac) triple."""
    configs = []
    for portion, gender, flip_frac in itertools.product(
        PORTIONS, GENDER_PROPORTIONS, FLIP_FRACTIONS
    ):
        if is_valid_client_config(portion, gender, flip_frac):
            configs.append((portion, gender, flip_frac))
    return configs


# =============================================================================
# Cumulative-metrics helpers
# =============================================================================
def compute_cumulative_metrics(
    configs: List[Tuple],
    initial_dataset_size: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Aggregate a list of (portion, gender_dict, flip_frac) tuples into three
    scalar metrics that summarise the group without exposing each individual.

      cumulative_portion - total data fraction used by these clients
      cumulative_male    - weighted male data fraction
      cumulative_flip    - weighted female-flip data fraction
    """
    cumulative_portion = sum(c[0] for c in configs) * initial_dataset_size
    cumulative_male    = sum(c[0] * c[1]["Male"] for c in configs) * initial_dataset_size
    cumulative_flip    = (
        sum(c[0] * c[1]["Female"] * c[2] for c in configs) * initial_dataset_size
    )
    return (cumulative_portion, cumulative_male, cumulative_flip)


def generate_all_cumulative_configs(
    num_other_clients: int,
) -> Dict[Tuple, List[List[Tuple]]]:
    """
    Enumerate every valid combination of `num_other_clients` individual
    configs and group them by their cumulative metrics.

    The result is cached to experiments/cache/cumulative_configs_Nclients.pkl
    on first call and loaded from disk on all subsequent calls.

    To invalidate the cache, delete the corresponding .pkl file and re-run.

    Returns
    -------
    dict  cumulative_tuple -> list of combinations that yield those metrics.
    """
    cache_key = f"cumulative_configs_{num_other_clients}clients"
    cached = _load_cache(cache_key)
    if cached is not None:
        print(f"  Loaded cumulative config map from cache "
              f"({_cache_path(cache_key).name})")
        return cached

    print(f"  Building cumulative config map for {num_other_clients} other "
          f"clients — this runs once and is then cached...")

    all_single_configs = get_all_valid_single_configs()
    cumulative_to_configs: Dict[Tuple, List[List[Tuple]]] = {}

    for combo in itertools.product(all_single_configs, repeat=num_other_clients):
        if sum(c[0] for c in combo) > 1.0:
            continue
        key = compute_cumulative_metrics(combo)
        cumulative_to_configs.setdefault(key, []).append(list(combo))

    _save_cache(cache_key, cumulative_to_configs)
    print(f"  Saved to {_cache_path(cache_key)}")
    return cumulative_to_configs


# =============================================================================
# Main grid generator
# =============================================================================
def generate_cumulative_param_grid(num_total_clients: int) -> List[Dict]:
    """
    Build the full experiment grid for the given client count.

    Each entry is a dict with three keys:
      client_0           - explicit config for the anchor client
      cumulative_metrics - aggregated metrics for the remaining clients
      num_other_clients  - num_total_clients - 1

    Total-portion constraint (client_0 + others <= 1.0) is enforced.
    The slow generate_all_cumulative_configs step is cached to disk.
    """
    if num_total_clients != NUM_CLIENTS:
        raise ValueError(
            f"NUM_CLIENTS env var is set to {NUM_CLIENTS}, "
            f"but generate_cumulative_param_grid was called with "
            f"num_total_clients={num_total_clients}. "
            f"Set export NUM_CLIENTS={num_total_clients} or fix the caller."
        )

    num_other_clients = num_total_clients - 1

    client_0_configs  = get_all_valid_single_configs()
    cumulative_map    = generate_all_cumulative_configs(num_other_clients)
    unique_cumulative = list(cumulative_map.keys())

    print(f"\n Configuration Space for {num_total_clients} clients:")
    print(f"   Client 0 configs            : {len(client_0_configs)}")
    print(f"   Unique cumulative configs    : {len(unique_cumulative)}")
    total_unconstrained = len(client_0_configs) * len(unique_cumulative)
    print(f"   Combinations (pre-filter)    : {total_unconstrained:,}")

    experiments = []
    for c0 in client_0_configs:
        for cum_key in unique_cumulative:
            if c0[0] + cum_key[0] > 1.0:
                continue
            experiments.append({
                "client_0": {
                    "portion":   c0[0],
                    "gender":    c0[1],
                    "flip_frac": c0[2],
                },
                "cumulative_metrics": {
                    "portion": cum_key[0],
                    "male":    cum_key[1],
                    "flip":    cum_key[2],
                },
                "num_other_clients": num_other_clients,
            })

    print(f"   Valid experiments (post-filter): {len(experiments):,}")
    return experiments


# =============================================================================
# Config-expansion helper
# =============================================================================
def find_matching_client_configs(
    cumulative_metrics: Dict,
    num_other_clients: int,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Given a cumulative_metrics dict (portion / male / flip) and a client
    count, randomly pick one individual-config combination that realises
    those metrics.

    Raises ValueError if no combination exists (should not happen for valid
    grid entries).
    """
    cumulative_tuple = (
        cumulative_metrics["portion"],
        cumulative_metrics["male"],
        cumulative_metrics["flip"],
    )

    cumulative_map = generate_all_cumulative_configs(num_other_clients)

    if cumulative_tuple not in cumulative_map:
        raise ValueError(
            f"No client configurations found that produce "
            f"cumulative metrics: {cumulative_metrics}"
        )

    possible_combos = cumulative_map[cumulative_tuple]

    if seed is not None:
        rng = np.random.RandomState(seed)
        selected = possible_combos[rng.randint(len(possible_combos))]
    else:
        selected = possible_combos[0]

    return [
        {"portion": portion, "gender": gender, "flip_frac": flip_frac}
        for portion, gender, flip_frac in selected
    ]


# =============================================================================
# Quick self-test
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(f"CUMULATIVE CONFIGURATION TESTING — {NUM_CLIENTS} CLIENTS")
    print("=" * 80)

    configs = generate_cumulative_param_grid(NUM_CLIENTS)
    print(f"\nTotal valid experiments : {len(configs):,}")

    example = configs[0]
    print(f"\nExample configuration:")
    print(f"  client_0          : {example['client_0']}")
    print(f"  cumulative_metrics: {example['cumulative_metrics']}")
    print(f"  num_other_clients : {example['num_other_clients']}")

    print("\nExpanding to individual clients (seed=42)...")
    individual = find_matching_client_configs(
        example["cumulative_metrics"],
        example["num_other_clients"],
        seed=42,
    )
    for i, cfg in enumerate(individual):
        print(f"  Client {i + 1}: {cfg}")

    tuples   = [(c["portion"], c["gender"], c["flip_frac"]) for c in individual]
    computed = compute_cumulative_metrics(tuples)
    expected = (
        example["cumulative_metrics"]["portion"],
        example["cumulative_metrics"]["male"],
        example["cumulative_metrics"]["flip"],
    )
    print(f"\nVerification:")
    print(f"  Expected: {expected}")
    print(f"  Computed: {computed}")
    print(f"  Match   : {np.allclose(expected, computed)}")

    print("\nSelf-test complete!")
