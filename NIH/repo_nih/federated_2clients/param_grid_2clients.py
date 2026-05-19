"""
param_grid_2clients.py
=======================
Full Cartesian-product parameter grid for 2-client federated experiments.

Unlike the cumulative grid used for N≥3 clients, with only 2 clients the
complete enumeration is tractable: 27 valid per-client configs × 27 = 729
total experiments.

Per-client valid combinations
------------------------------
  portion      ∈ [0.12, 0.06, 0.03]
  gender       ∈ [{Male:1, Female:0}, {Male:0.5, Female:0.5}, {Male:0, Female:1}]
  flip_frac    ∈ [0.0, 0.15, 0.30, 0.45]

Constraint: if gender == {Male:1, Female:0} then flip_frac must be 0.0.

Result: 3 portions × (3 genders × 4 flips − 3 invalid all-male/flip>0) = 27
        per client → 27² = 729 experiments total.
"""

import itertools
from typing import List, Dict

# ---------------------------------------------------------------------------
# Allowed values (identical to the Nclients grid)
# ---------------------------------------------------------------------------
PORTIONS = [0.12, 0.06, 0.03]
GENDER_PROPORTIONS = [
    {"Male": 1.0, "Female": 0.0},
    {"Male": 0.5, "Female": 0.5},
    {"Male": 0.0, "Female": 1.0},
]
FLIP_FRACTIONS = [0.0, 0.15, 0.30, 0.45]


def _valid_client_configs() -> List[Dict]:
    """Return all 27 valid per-client parameter combinations."""
    configs = []
    for portion, gender, flip_frac in itertools.product(
        PORTIONS, GENDER_PROPORTIONS, FLIP_FRACTIONS
    ):
        if gender == {"Male": 1.0, "Female": 0.0} and flip_frac != 0.0:
            continue
        configs.append({
            "portion":   portion,
            "gender":    gender,
            "flip_frac": flip_frac,
        })
    return configs


def generate_param_grid() -> List[List[Dict]]:
    """
    Return all 729 valid 2-client experiment configurations.

    Each entry is a list of exactly 2 client-config dicts:
        [
            {"portion": ..., "gender": {...}, "flip_frac": ...},  # client 0
            {"portion": ..., "gender": {...}, "flip_frac": ...},  # client 1
        ]
    """
    client_space = _valid_client_configs()
    return [
        [dict(c0), dict(c1)]
        for c0, c1 in itertools.product(client_space, repeat=2)
    ]
