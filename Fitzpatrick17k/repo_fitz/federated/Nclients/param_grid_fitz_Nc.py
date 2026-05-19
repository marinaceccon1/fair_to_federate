"""
param_grid_fitz_Nc.py
─────────────────────
Parameter grid and aggregate logic for N-client (N > 2) Fitzpatrick FL setups.

Architecture:
  • Client 0  — fixed anchor (iterated over all valid single-client options)
  • Clients 1…N-1 — their joint aggregate is the unit of tracking

Aggregate definition (all N-1 non-anchor clients together):
    cum_portion = sum_i( p_i )
    cum_14      = sum_i( pct14_i * p_i )
    cum_flip    = sum_i( flip_i * (1 - pct14_i) * p_i )

The canonical experiment key is (client0, agg_1…N-1).

Usage:
    from param_grid_fitz_Nc import (
        generate_param_grid, resample_non_anchor_clients,
        get_aggregate_keys, get_client0_strata,
        _aggregate, _AGGREGATE_INDEX, _tuple_to_dict,
    )
    grid = generate_param_grid(num_clients=3)   # or 5, 7, …
"""

import itertools
import random
from typing import List, Dict, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Allowed values
# ─────────────────────────────────────────────────────────────────────────────
PORTIONS = [0.70, 0.60, 0.50]

GROUP_COMPOSITIONS = [
    {"fitz_14": 1.00, "fitz_56": 0.00},   # pure light
    {"fitz_14": 0.75, "fitz_56": 0.25},   # 75 % light, 25 % dark
    {"fitz_14": 0.50, "fitz_56": 0.50},   # 50 / 50
]

FLIP_FRACTIONS = [0.0, 0.15, 0.30, 0.45]

FLIP_GROUP     = "fitz_56"
ROUND_DECIMALS = 4


def _has_flip_group(composition: dict) -> bool:
    return composition.get(FLIP_GROUP, 0.0) > 0.0


def _valid_client_options() -> List[Tuple]:
    """All valid (portion, composition, flip_frac) triples."""
    options = []
    for portion, composition, flip_frac in itertools.product(
        PORTIONS, GROUP_COMPOSITIONS, FLIP_FRACTIONS
    ):
        if not _has_flip_group(composition) and flip_frac != 0.0:
            continue
        options.append((portion, composition, flip_frac))
    return options


def _aggregate(*clients: Tuple) -> Tuple[float, float, float]:
    """
    Compute the joint aggregate for an arbitrary number of non-anchor clients.

    Each client is a (portion, composition, flip_frac) tuple.

    cum_portion = sum_i( p_i )
    cum_14      = sum_i( pct14_i * p_i )
    cum_flip    = sum_i( flip_i * (1 - pct14_i) * p_i )
    """
    if not clients:
        raise ValueError("At least one client required for aggregation")
    cum_portion = round(sum(c[0] for c in clients), ROUND_DECIMALS)
    cum_14      = round(sum(c[1]["fitz_14"] * c[0] for c in clients), ROUND_DECIMALS)
    cum_flip    = round(
        sum(c[2] * (1.0 - c[1]["fitz_14"]) * c[0] for c in clients),
        ROUND_DECIMALS,
    )
    return (cum_portion, cum_14, cum_flip)


def _build_aggregate_index(num_non_anchor: int) -> Dict[Tuple, List[Tuple]]:
    """
    Build  aggregate_key → [tuple_of_client_tuples, ...]  mapping over all
    valid (num_non_anchor)-tuples of client options.

    Note: this can be expensive for large num_non_anchor (e.g. 4 → 50,625
    combinations). It is computed once at import time via the module-level
    factory below, keyed by num_non_anchor.
    """
    options = _valid_client_options()
    index: Dict[Tuple, List[Tuple]] = {}
    for combo in itertools.product(options, repeat=num_non_anchor):
        key = _aggregate(*combo)
        index.setdefault(key, []).append(combo)
    return index


def _tuple_to_dict(t: Tuple) -> Dict:
    portion, composition, flip_frac = t
    return {"portion": portion, "composition": composition, "flip_frac": flip_frac}


def _client0_options() -> List[Dict]:
    return [
        {"portion": p, "composition": comp, "flip_frac": f}
        for p, comp, f in _valid_client_options()
    ]


# =============================================================================
# Module-level index cache — built on first use per num_non_anchor value
# =============================================================================
_INDEX_CACHE: Dict[int, Dict[Tuple, List[Tuple]]] = {}


def _get_index(num_non_anchor: int) -> Dict[Tuple, List[Tuple]]:
    if num_non_anchor not in _INDEX_CACHE:
        print(f"⚙️  Building aggregate index for {num_non_anchor} non-anchor clients "
              f"(one-time cost)...")
        _INDEX_CACHE[num_non_anchor] = _build_aggregate_index(num_non_anchor)
        n_keys = len(_INDEX_CACHE[num_non_anchor])
        print(f"   → {n_keys} distinct aggregate keys")
    return _INDEX_CACHE[num_non_anchor]


# Convenience aliases used by experiment_manager (kept for backward compat)
# These are populated lazily when get_aggregate_keys() / _AGGREGATE_INDEX is
# accessed, so importing this module is always cheap.
class _LazyIndexProxy:
    """Proxy that builds the index for a fixed num_non_anchor on first access."""
    def __init__(self, num_non_anchor: int):
        self._n = num_non_anchor
        self._d = None

    def _load(self):
        if self._d is None:
            self._d = _get_index(self._n)

    def __getitem__(self, key):
        self._load(); return self._d[key]

    def __contains__(self, key):
        self._load(); return key in self._d

    def keys(self):
        self._load(); return self._d.keys()

    def items(self):
        self._load(); return self._d.items()


# =============================================================================
# Public API
# =============================================================================

def get_client0_strata() -> List[Dict]:
    """Return all valid client-0 configurations (the strata)."""
    return _client0_options()


def get_aggregate_keys(num_clients: int) -> List[Tuple]:
    """Return sorted list of all distinct aggregate keys for (num_clients-1) non-anchor clients."""
    num_non_anchor = num_clients - 1
    return sorted(_get_index(num_non_anchor).keys())


def generate_param_grid(num_clients: int) -> List[List[Dict]]:
    """
    Enumerate all canonical (client0, agg_key) pairs for the given
    number of clients (N > 2).

    Each entry is an N-element list:
        [c0, sentinel_c1, …, sentinel_cN-1]
    where sentinels are the first client tuple in the sorted aggregate index
    for that key. Actual non-anchor clients are resampled at runtime by
    resample_non_anchor_clients().

    Args:
        num_clients: Total number of FL clients (must be > 2).
    """
    if num_clients < 3:
        raise ValueError(f"num_clients must be >= 3, got {num_clients}")

    num_non_anchor = num_clients - 1
    index     = _get_index(num_non_anchor)
    agg_keys  = sorted(index.keys())

    configs = []
    for c0 in _client0_options():
        for agg_key in agg_keys:
            sentinel_combo = index[agg_key][0]   # first realisation
            configs.append([c0] + [_tuple_to_dict(t) for t in sentinel_combo])
    return configs


def resample_non_anchor_clients(
    config: List[Dict],
    rng: Optional[random.Random] = None,
) -> List[Dict]:
    """
    Given a canonical N-element config, re-sample clients 1…N-1 from all
    tuples that realise the same joint aggregate key, leaving client 0
    unchanged.

    Args:
        config: N-element list of client config dicts (first element is c0).
        rng:    Optional random.Random instance for reproducibility.
    """
    if rng is None:
        rng = random.Random()

    num_non_anchor = len(config) - 1
    index = _get_index(num_non_anchor)

    c0      = config[0]
    non_anchor_tuples = tuple(
        (c["portion"], c["composition"], c["flip_frac"]) for c in config[1:]
    )
    agg_key = _aggregate(*non_anchor_tuples)

    new_combo = rng.choice(index[agg_key])
    return [c0] + [_tuple_to_dict(t) for t in new_combo]
