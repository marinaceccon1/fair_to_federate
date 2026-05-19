import itertools

# Allowed values
PORTIONS = [0.12, 0.06, 0.03]
GENDER_PROPORTIONS = [
    {"Male": 1.0, "Female": 0.0},
    {"Male": 0.5, "Female": 0.5},
    {"Male": 0.0, "Female": 1.0},
]
FLIP_FRACTIONS = [0.0, 0.15, 0.30, 0.45]

def generate_param_grid(num_clients: int):
    """
    Returns list of experiment configs.
    Each config is a list of client configs.
    
    Constraint: When gender is 100% male (Male: 1.0, Female: 0.0),
    flip_frac must be 0.0 (since there are no female labels to flip).
    """
    # Generate all combinations, then filter
    client_space = []
    for portion, gender, flip_frac in itertools.product(
        PORTIONS, GENDER_PROPORTIONS, FLIP_FRACTIONS
    ):
        # If all male (no females), skip non-zero flip fractions
        if gender == {"Male": 1.0, "Female": 0.0} and flip_frac != 0.0:
            continue
        client_space.append((portion, gender, flip_frac))
    
    # Cartesian product across clients
    all_experiments = list(
        itertools.product(client_space, repeat=num_clients)
    )
    
    # Convert to readable structure
    formatted = []
    for exp in all_experiments:
        formatted.append([
            {
                "portion": c[0],
                "gender": c[1],
                "flip_frac": c[2],
            }
            for c in exp
        ])
    
    return formatted
