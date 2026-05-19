import itertools

# ─────────────────────────────────────────────────────────────────────────────
# Allowed values
# ─────────────────────────────────────────────────────────────────────────────

# Fraction of each group's N-sized balanced pool allocated to training
# (val and test are always fixed at N*0.15 each)
PORTIONS = [0.70, 0.60, 0.50]

# Group compositions — each entry is a dict mapping group_name → fraction of
# the training budget drawn from that group.  Fractions must sum to 1.0.
# Imbalance is only allowed in favour of the light-skin group (fitz_14),
# so the darkest option is 50/50.  This ensures the fitz_56 training pool
# (N * 0.70) is never exhausted even when both clients pick the most
# dark-skin-heavy composition, keeping portion=1.0 safe (see data_setup_fitz).
GROUP_COMPOSITIONS = [
    {"fitz_14": 1.00, "fitz_56": 0.00},   # pure light
    {"fitz_14": 0.75, "fitz_56": 0.25},   # 75 % light, 25 % dark
    {"fitz_14": 0.50, "fitz_56": 0.50},   # 50 / 50
]

FLIP_FRACTIONS = [0.0, 0.15, 0.30, 0.45]

# Group whose positive labels can be flipped
FLIP_GROUP = "fitz_56"


def _has_flip_group(composition: dict) -> bool:
    """Return True if the composition contains any samples from the flip group."""
    return composition.get(FLIP_GROUP, 0.0) > 0.0


def generate_param_grid(num_clients: int):
    """
    Returns a list of experiment configs.
    Each config is a list of per-client dicts with keys:
        portion      : float  — fraction of N used as training budget
        composition  : dict   — {group_name: fraction_of_budget}
        flip_frac    : float  — fraction of fitz_56 positives to flip

    Constraint: if composition has no fitz_56 samples, flip_frac must be 0.0.
    """
    # Build the per-client option space (valid combinations only)
    client_space = []
    for portion, composition, flip_frac in itertools.product(
        PORTIONS, GROUP_COMPOSITIONS, FLIP_FRACTIONS
    ):
        if not _has_flip_group(composition) and flip_frac != 0.0:
            continue
        client_space.append((portion, composition, flip_frac))

    # Cartesian product across clients
    all_experiments = list(itertools.product(client_space, repeat=num_clients))

    # Convert to readable structure
    formatted = []
    for exp in all_experiments:
        formatted.append([
            {
                "portion":     c[0],
                "composition": c[1],
                "flip_frac":   c[2],
            }
            for c in exp
        ])

    return formatted