import sys
import os
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root anchored on __file__ — works regardless of CWD.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =============================================================================
# LOAD CONFIGURATION
# =============================================================================
_default_config = str(REPO_ROOT / "experiments_fitz" / "current_config.json")
CONFIG_PATH = os.environ.get("CONFIG_PATH", _default_config)

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"❌ Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH) as f:
    client_configs = json.load(f)

num_clients = len(client_configs)

# =============================================================================
# CONSTANTS
# =============================================================================
# NOTE: no global `device` — each process sets its own via CUDA_VISIBLE_DEVICES.

FITZPATRICK_GROUPS = {
    "fitz_14": [1, 2, 3, 4],
    "fitz_56": [5, 6],
}
FLIP_GROUP = "fitz_56"   # only this group's positives can be flipped

LABEL      = "high"
BATCH_SIZE = 32

# =============================================================================
# DATA PATHS
# Set these environment variables before running, e.g.:
#   export IMAGE_DIR=/path/to/fitzpatrick17k/images
#   export CSV_PATH=/path/to/fitzpatrick17k.csv
# =============================================================================
IMAGE_DIR = os.environ.get("IMAGE_DIR")
if IMAGE_DIR is None:
    raise EnvironmentError(
        "❌ IMAGE_DIR environment variable is not set. "
        "Please set it to the folder containing the Fitzpatrick17k images, e.g.:\n"
        "  export IMAGE_DIR=/path/to/fitzpatrick17k/images"
    )

CSV_PATH = os.environ.get("CSV_PATH")
if CSV_PATH is None:
    raise EnvironmentError(
        "❌ CSV_PATH environment variable is not set. "
        "Please set it to the Fitzpatrick17k CSV file, e.g.:\n"
        "  export CSV_PATH=/path/to/fitzpatrick17k.csv"
    )

# =============================================================================
# TRANSFORMS
# =============================================================================
train_transform = transforms.Compose([
    transforms.Resize(320),
    transforms.RandomCrop(288),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(288),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# =============================================================================
# DATASET
# =============================================================================
class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.root_dir  = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["hasher"] + ".jpg")
        for attempt in range(len(self.df)):
            try:
                image = Image.open(img_path).convert("RGB")
                break
            except Exception as e:
                print(f"[warn] Skipping {img_path} (attempt {attempt + 1}): {e}")
                idx      = (idx + 1) % len(self.df)
                row      = self.df.iloc[idx]
                img_path = os.path.join(self.root_dir, row["hasher"] + ".jpg")
        else:
            raise RuntimeError(
                f"Could not load any valid image after {len(self.df)} attempts."
            )
        if self.transform:
            image = self.transform(image)
        return {
            "image":          image,
            LABEL:            row[LABEL],
            "hasher":         row["hasher"],
            "fitzpatrick":    row["fitzpatrick_scale"],
            "label_flipped":  int(row.get("label_flipped", False)),
        }


# =============================================================================
# MODEL
# =============================================================================
def create_model():
    model = models.efficientnet_b2(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model


# =============================================================================
# DATA HELPERS
# =============================================================================
def get_group(val):
    for group_name, values in FITZPATRICK_GROUPS.items():
        if val in values:
            return group_name
    return None


def stratified_subsample(df, label, n, seed):
    """Subsample df to exactly n rows, stratified on label."""
    if n == 0:
        return pd.DataFrame(columns=df.columns)
    if n >= len(df):
        return df.reset_index(drop=True)
    kept, _ = train_test_split(
        df,
        train_size=n,
        stratify=df[label] if df[label].nunique() > 1 else None,
        random_state=seed,
    )
    return kept.reset_index(drop=True)


def flip_positive_labels(df, label, fraction, seed):
    """
    Randomly flip `fraction` of positive labels in df to 0.
    Returns a copy of df with a boolean 'label_flipped' column marking changes.
    """
    df = df.copy()
    df["label_flipped"] = False
    pos_idx = df.index[df[label] == 1].tolist()
    n_flip  = int(round(len(pos_idx) * fraction))
    if n_flip == 0:
        return df
    rng          = np.random.default_rng(seed)
    flip_indices = rng.choice(pos_idx, size=n_flip, replace=False)
    df.loc[flip_indices, label]           = 0
    df.loc[flip_indices, "label_flipped"] = True
    print(f"  [label flip] Flipped {n_flip}/{len(pos_idx)} positives → 0")
    return df


def carve_val_test_per_group(group_dfs, label, N, n_val, n_test, seed):
    """
    For every Fitzpatrick group, carve out val and test sets (same logic as
    balance_and_split in the original script), prioritising reliable samples
    for val/test and leaving the rest as an overlap-free training pool.

    Rule 1 – Pool construction:
        Reliable samples fill up to N first; unreliable top up the remainder.
    Rule 2 – Val/test allocation:
        Val+test are drawn from reliable samples first, then unreliable.
        Train pool receives whatever is left.

    Returns
    -------
    val_sets    : dict {group_name: val_df}
    test_sets   : dict {group_name: test_df}
    train_pools : dict {group_name: train_df}   — overlap-free training pool
    """
    val_sets    = {}
    test_sets   = {}
    train_pools = {}

    for group_name, gdf in group_dfs.items():
        reliable   = gdf[gdf["fitz_disagreement"] == False].copy().reset_index(drop=True)
        unreliable = gdf[gdf["fitz_disagreement"] == True].copy().reset_index(drop=True)

        # ── Rule 1: build the N-sample pool (reliable first) ─────────────────
        n_rel_in_pool   = min(len(reliable), N)
        pool_rel        = stratified_subsample(reliable, label, n_rel_in_pool, seed)
        n_unrel_needed  = N - n_rel_in_pool
        n_unrel_in_pool = min(len(unreliable), n_unrel_needed)
        pool_unrel      = stratified_subsample(unreliable, label, n_unrel_in_pool, seed)

        if n_rel_in_pool + n_unrel_in_pool < N:
            print(f"  [warn] Group {group_name} has only "
                  f"{n_rel_in_pool + n_unrel_in_pool} / {N} samples.")

        # ── Rule 2: assign val+test from reliable first ───────────────────────
        n_valtest           = n_val + n_test
        n_rel_for_valtest   = min(len(pool_rel), n_valtest)
        n_unrel_for_valtest = n_valtest - n_rel_for_valtest

        # Step 2c — carve reliable portion for val+test
        if n_rel_for_valtest == len(pool_rel):
            valtest_rel = pool_rel.copy()
            train_rel   = pd.DataFrame(columns=gdf.columns)
        else:
            train_rel, valtest_rel = train_test_split(
                pool_rel,
                test_size=n_rel_for_valtest,
                stratify=pool_rel[label] if pool_rel[label].nunique() > 1 else None,
                random_state=seed,
            )

        # Step 2d — carve unreliable portion for val+test
        if n_unrel_for_valtest == 0:
            valtest_unrel = pd.DataFrame(columns=gdf.columns)
            train_unrel   = pool_unrel.copy()
        elif n_unrel_for_valtest == len(pool_unrel):
            valtest_unrel = pool_unrel.copy()
            train_unrel   = pd.DataFrame(columns=gdf.columns)
        else:
            train_unrel, valtest_unrel = train_test_split(
                pool_unrel,
                test_size=n_unrel_for_valtest,
                stratify=pool_unrel[label] if pool_unrel[label].nunique() > 1 else None,
                random_state=seed,
            )

        valtest_df = pd.concat([valtest_rel, valtest_unrel], ignore_index=True)
        train_df   = pd.concat([train_rel,   train_unrel],   ignore_index=True)

        # ── Split valtest → val / test (stratified) ───────────────────────────
        val_df, test_df = train_test_split(
            valtest_df,
            test_size=n_test,
            stratify=valtest_df[label] if valtest_df[label].nunique() > 1 else None,
            random_state=seed,
        )

        n_rel = lambda df: int((df["fitz_disagreement"] == False).sum())
        print(f"  Group {group_name} → train pool: {len(train_df)} (rel {n_rel(train_df)}) | "
              f"val: {len(val_df)} (rel {n_rel(val_df)}) | "
              f"test: {len(test_df)} (rel {n_rel(test_df)})")

        val_sets[group_name]    = val_df.reset_index(drop=True)
        test_sets[group_name]   = test_df.reset_index(drop=True)
        train_pools[group_name] = train_df.reset_index(drop=True)

    return val_sets, test_sets, train_pools


def build_client_train_df(train_pools, composition, portion, N, flip_frac, seed):
    """
    Build the training DataFrame for one client.

    Parameters
    ----------
    train_pools : dict {group_name: df}   — overlap-free pools per group
    composition : dict {group_name: frac} — fraction of training budget per group
    portion     : float                   — fraction of N used as training budget
    N           : int                     — balanced group size
    flip_frac   : float                   — fraction of fitz_56 positives to flip
    seed        : int

    Returns
    -------
    train_df           : pd.DataFrame (with 'label_flipped' column)
    group_train_sizes  : dict {group_name: int}
    """
    target_total = int(N * portion)
    print(f"  Training budget: {target_total} (N={N}, portion={portion:.2f})")

    group_dfs_list = []
    group_train_sizes = {}

    for group_name, frac in composition.items():
        if frac == 0.0:
            group_train_sizes[group_name] = 0
            continue

        pool = train_pools[group_name]
        n_needed = int(round(target_total * frac))
        n_needed = min(n_needed, len(pool))

        # Prefer reliable samples
        reliable   = pool[pool["fitz_disagreement"] == False].copy().reset_index(drop=True)
        unreliable = pool[pool["fitz_disagreement"] == True].copy().reset_index(drop=True)

        n_rel  = min(len(reliable),   n_needed)
        n_unrl = min(len(unreliable), n_needed - n_rel)

        sampled_rel  = stratified_subsample(reliable,   LABEL, n_rel,  seed)
        sampled_unrl = stratified_subsample(unreliable, LABEL, n_unrl, seed)
        sampled      = pd.concat([sampled_rel, sampled_unrl], ignore_index=True)

        group_dfs_list.append(sampled)
        group_train_sizes[group_name] = len(sampled)
        print(f"  Group {group_name} (frac={frac:.3f}) → "
              f"sampled {len(sampled)} / {n_needed} (rel {len(sampled_rel)}, unrl {len(sampled_unrl)})")

    if not group_dfs_list:
        raise ValueError("No samples were drawn — check composition fractions.")

    train_df = pd.concat(group_dfs_list, ignore_index=True)
    train_df["label_flipped"] = False

    # ── Apply label flipping to fitz_56 rows only ─────────────────────────────
    if flip_frac > 0.0 and composition.get(FLIP_GROUP, 0.0) > 0.0:
        flip_mask   = train_df["fitzpatrick_scale"].isin(FITZPATRICK_GROUPS[FLIP_GROUP])
        flip_subset = train_df[flip_mask].copy()
        rest_subset = train_df[~flip_mask].copy()

        flip_subset = flip_positive_labels(flip_subset, LABEL, flip_frac, seed)
        train_df    = pd.concat([rest_subset, flip_subset], ignore_index=True)

        n_flipped = int(train_df["label_flipped"].sum())
        print(f"  Total flipped labels in training set: {n_flipped}")
    else:
        print(f"  No label flipping applied (flip_frac={flip_frac}, "
              f"fitz_56_frac={composition.get(FLIP_GROUP, 0.0):.3f})")

    return train_df.reset_index(drop=True), group_train_sizes


def make_weighted_sampler(df, label):
    counts         = df[label].value_counts().sort_index().values
    class_weights  = 1.0 / counts.astype(float)
    sample_weights = np.array([class_weights[int(y)] for y in df[label]])
    return WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(sample_weights),
        replacement = True,
    )


# =============================================================================
# GLOBAL DATA LOAD — called once at module import time so that every client
# subprocess shares the same preprocessing before carving its own slice.
# =============================================================================
def _load_and_preprocess() -> tuple:
    """
    Load the CSV, apply label and hash mappings, flag disagreements,
    and divide into per-group DataFrames.

    Returns (group_dfs, N, n_val, n_test)
    """
    df_full = pd.read_csv(CSV_PATH)

    df_full["high"]   = df_full["three_partition_label"].astype("category").cat.codes
    df_full["high"]   = df_full["high"].replace(0, 1)
    df_full["high"]   = df_full["high"].replace(2, 0)
    df_full["hasher"] = df_full["md5hash"]

    # Flag annotator disagreements
    def is_disagreement(row):
        g_scale   = get_group(row["fitzpatrick_scale"])
        g_centaur = get_group(row["fitzpatrick_centaur"])
        if g_scale is None or g_centaur is None:
            return True
        return g_scale != g_centaur

    df_full["fitz_disagreement"] = df_full.apply(is_disagreement, axis=1)

    group_dfs = {
        group_name: df_full[df_full["fitzpatrick_scale"].isin(fitz_values)].copy()
        for group_name, fitz_values in FITZPATRICK_GROUPS.items()
    }
    group_sizes = {name: len(gdf) for name, gdf in group_dfs.items()}
    N = min(group_sizes.values())

    n_val  = int(N * 0.15)
    n_test = N - int(N * 0.70) - n_val   # exact remainder so train+val+test == N

    print(f"Group sizes: {group_sizes}")
    print(f"N = {N}  →  n_val = {n_val}, n_test = {n_test}")

    return group_dfs, N, n_val, n_test


# =============================================================================
# POOL PARTITIONING
# =============================================================================
def partition_val_sets(val_sets: dict, N: int) -> dict:
    """
    Partition each group's val set into num_clients non-overlapping slices
    using the same cursor approach as partition_train_pools.

    Slice sizes are proportional to each client's composition fraction for
    that group, capped so their sum never exceeds the available val pool.
    This ensures two clients with the same composition still evaluate on
    different (non-overlapping) subsets of the val data.

    Returns
    -------
    client_vals : dict { cid: { group_name: df } }
    """
    client_vals = {
        cid: {g: pd.DataFrame(columns=val_df.columns)
              for g, val_df in val_sets.items()}
        for cid in range(num_clients)
    }

    for group_name, val_df in val_sets.items():
        # Shuffle once reproducibly (use seed=1 to differ from train shuffle)
        val_shuffled = val_df.sample(
            frac=1, random_state=1
        ).reset_index(drop=True)

        n_val_pool = len(val_shuffled)

        # Each client's slice is proportional to its composition fraction
        # for this group, scaled to n_val (= N * 0.15)
        slices = []
        for c in client_configs:
            frac     = c["composition"].get(group_name, 0.0)
            n_wanted = int(round(N * 0.15 * frac))
            slices.append(n_wanted)

        total_wanted = sum(slices)
        if total_wanted > n_val_pool:
            print(f"  [warn] Val group {group_name}: clients want {total_wanted} "
                  f"but pool has {n_val_pool}. Scaling down proportionally.")
            scale  = n_val_pool / total_wanted
            slices = [int(s * scale) for s in slices]

        cursor = 0
        for cid, n in enumerate(slices):
            client_vals[cid][group_name] = (
                val_shuffled.iloc[cursor: cursor + n].reset_index(drop=True)
            )
            cursor += n
            print(f"  Val partition — group {group_name}, "
                  f"client {cid}: rows [{cursor - n}, {cursor}) "
                  f"({n} samples)")

    return client_vals


def partition_train_pools(train_pools: dict, N: int) -> dict:
    """
    Partition each group's training pool into num_clients non-overlapping
    slices using a cursor, mirroring the gender_partition approach.

    The worst-case for fitz_56 is both clients at 50/50 composition, which
    together consume at most N*0.70 from fitz_56 — exactly the pool size —
    so portion=1.0 is safe.  fitz_14 is the larger group and is never the
    bottleneck.

    Each group's pool is shuffled once with the global seed, then sliced
    proportionally to each client's fitz_56 demand (their composition
    fraction for that group).  This guarantees:
        • No row appears in two clients' slices.
        • Each client receives exactly the rows it needs, up to pool size.

    Returns
    -------
    client_pools : dict { cid: { group_name: df } }
    """
    client_pools = {
        cid: {g: pd.DataFrame(columns=pool.columns)
              for g, pool in train_pools.items()}
        for cid in range(num_clients)
    }

    for group_name, pool_df in train_pools.items():
        # Shuffle pool once, reproducibly
        pool_shuffled = pool_df.sample(
            frac=1, random_state=0
        ).reset_index(drop=True)

        # Compute each client's slice size: portion * composition_frac * N
        # (capped so the sum never exceeds the pool)
        n_train_pool = len(pool_shuffled)
        slices = []
        for c in client_configs:
            frac     = c["composition"].get(group_name, 0.0)
            n_wanted = int(round(int(N * c["portion"]) * frac))
            slices.append(n_wanted)

        total_wanted = sum(slices)
        if total_wanted > n_train_pool:
            print(f"  [warn] Group {group_name}: clients want {total_wanted} "
                  f"but pool has {n_train_pool}. Scaling down proportionally.")
            scale  = n_train_pool / total_wanted
            slices = [int(s * scale) for s in slices]

        # Assign non-overlapping cursor-based slices
        cursor = 0
        for cid, n in enumerate(slices):
            client_pools[cid][group_name] = (
                pool_shuffled.iloc[cursor: cursor + n].reset_index(drop=True)
            )
            cursor += n
            print(f"  Pool partition — group {group_name}, "
                  f"client {cid}: rows [{cursor - n}, {cursor}) "
                  f"({n} samples)")

    return client_pools


# =============================================================================
# PER-CLIENT SETUP
# =============================================================================
def setup_for_client(cid: int, seed: int = 42):
    """
    Build and return (train_loader, val_loader, group_train_sizes)
    for the single client `cid` under the given seed.

    Val and test sets are carved out first (reproducibly by seed) so they
    are identical across all clients for the same seed.

    The training pool is partitioned across ALL clients upfront using a
    cursor (no overlap by construction), and each client then samples from
    its own exclusive slice using a per-client seed (seed + cid) so that
    different clients do not produce the same draw order.
    """
    cfg = client_configs[cid]

    # Per-client seed: val/test use the global seed (same split for all
    # clients); training sampling uses seed+cid (different per client).
    client_seed = seed + cid

    print(f"\n🧪 EXPERIMENT CONFIGURATION (building for client {cid})")
    for i, c in enumerate(client_configs):
        marker = " ◄ THIS CLIENT" if i == cid else ""
        print(f"  Client {i} | portion={c['portion']:.2f} | "
              f"composition={c['composition']} | flip={c['flip_frac']}{marker}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n📂 Loading data (seed={seed}, client_seed={client_seed})...")
    group_dfs, N, n_val, n_test = _load_and_preprocess()

    # ── Carve val + test (fixed across clients, same seed for all) ────────────
    print(f"\n📊 Carving val/test sets (seed={seed})...")
    val_sets, test_sets, train_pools = carve_val_test_per_group(
        group_dfs, LABEL, N, n_val, n_test, seed=seed
    )

    # ── Partition val sets across ALL clients (no overlap) ───────────────────
    # Each client gets a non-overlapping slice of each group's val pool,
    # sized proportionally to its composition fraction for that group.
    # Two clients with the same composition therefore evaluate on different
    # samples, preventing identical val losses.
    print(f"\n🔀 Partitioning val sets across {num_clients} clients...")
    client_vals = partition_val_sets(val_sets, N)
    combined_val = pd.concat(
        [client_vals[cid][g]
         for g, frac in cfg["composition"].items() if frac > 0.0],
        ignore_index=True
    )

    # ── Partition training pools across ALL clients (no overlap) ──────────────
    print(f"\n🔀 Partitioning training pools across {num_clients} clients...")
    client_pools = partition_train_pools(train_pools, N)

    # ── Build this client's training set from its exclusive slice ─────────────
    print(f"\n🗂️  Building training set for client {cid} (client_seed={client_seed})...")
    train_df, group_train_sizes = build_client_train_df(
        train_pools  = client_pools[cid],
        composition  = cfg["composition"],
        portion      = cfg["portion"],
        N            = N,
        flip_frac    = cfg["flip_frac"],
        seed         = client_seed,
    )

    print(f"\n  Train set size : {len(train_df)}")
    print(f"  Val   set size : {len(combined_val)}")
    print(f"  Train per-group: {group_train_sizes}")
    vc = train_df[LABEL].value_counts().sort_index()
    print(f"  Train label dist (after flip): neg={vc.get(0, 0)} pos={vc.get(1, 0)}")
    print(f"  Flipped labels in train: {int(train_df['label_flipped'].sum())}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    sampler = make_weighted_sampler(train_df, LABEL)
    train_ds = SkinDataset(train_df, root_dir=IMAGE_DIR, transform=train_transform)
    val_ds   = SkinDataset(combined_val, root_dir=IMAGE_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False,
        sampler=sampler, num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    print(f"✅ Client {cid} data ready | "
          f"train_batches={len(train_loader)} | val_batches={len(val_loader)}")

    return train_loader, val_loader, group_train_sizes

# NOTE: FitzpatrickClient (Flower NumPyClient) lives in src/client_fitz.py.
# That version includes GPU CPU-offload, per-GPU file locking for shared-GPU
# setups, and a persistent optimizer/scheduler across rounds.  Do not add a
# second definition here.
