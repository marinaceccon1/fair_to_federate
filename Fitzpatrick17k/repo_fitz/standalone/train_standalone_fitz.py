"""
train_standalone_fitz.py
────────────────────────
Trains a single EfficientNet-B2 model on client 0's data only (no federation).
Mirrors the FL training loop exactly:
  • same number of rounds (NUM_ROUNDS = 30)
  • same local optimiser / LR schedule
  • same val-loss-based best-model checkpointing
  • same config-tag filename convention

Usage (called by run_standalone_fitz.py):
    CONFIG_PATH=experiments_standalone/current_config_worker0.json \
    CUDA_VISIBLE_DEVICES=0 \
    python train_standalone_fitz.py
"""

import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from pathlib import Path
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Repo root anchored on __file__
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# =============================================================================
# LOAD CONFIGURATION
# =============================================================================
_default_config = str(REPO_ROOT / "experiments_standalone" / "current_config.json")
CONFIG_PATH = os.environ.get("CONFIG_PATH", _default_config)

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"❌ Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH) as f:
    experiment_config = json.load(f)   # list with ONE client-0 dict

assert len(experiment_config) == 1, (
    f"Standalone trainer expects exactly 1 client config, "
    f"got {len(experiment_config)}"
)
client_cfg = experiment_config[0]

print("\n🧪 STANDALONE TRAINER: Loaded experiment configuration")
print(f"  portion={client_cfg['portion']}, "
      f"composition={client_cfg['composition']}, "
      f"flip_frac={client_cfg['flip_frac']}")

# =============================================================================
# DATA SEED  (forwarded from run_standalone_fitz.py via env)
# =============================================================================
DATA_SEED = int(os.environ.get("DATA_SEED", "42"))

# =============================================================================
# IMPORT DATA HELPERS
# (data_setup_fitz.py reads CONFIG_PATH itself at import time, so the env var
#  must already be set before this import — guaranteed by the runner.)
# =============================================================================
from src.data_setup_fitz import (
    setup_for_client,
    create_model,
    LABEL,
)

# =============================================================================
# HYPER-PARAMETERS  (identical to the FL setup)
# =============================================================================
NUM_ROUNDS    = 30
LOCAL_EPOCHS  = 1          # one epoch per "round" — same as FL default
LR_FEATURES   = 1e-5
LR_HEAD       = 1e-4
MAX_GRAD_NORM = 1.0

SAVE_DIR = Path(os.environ.get("SAVE_DIR", str(REPO_ROOT / "best_models_fitz")))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}\n")

# =============================================================================
# CONFIG TAG  (matches server_fitz.py format_config_tag)
# =============================================================================
def format_config_tag(client_configs) -> str:
    parts = []
    for i, c in enumerate(client_configs):
        comp     = c["composition"]
        comp_str = (
            f"14x{int(comp['fitz_14'] * 100)}"
            f"_56x{int(comp['fitz_56'] * 100)}"
        )
        parts.append(
            f"C{i}"
            f"_p{c['portion']}"
            f"_{comp_str}"
            f"_flip{c['flip_frac']}"
        )
    return "__".join(parts)


config_tag = format_config_tag(experiment_config)
print(f"🏷  Model tag: {config_tag}\n")

# =============================================================================
# DATA
# =============================================================================
print("📂 Setting up data for client 0 ...")
train_loader, val_loader, group_train_sizes = setup_for_client(0, seed=DATA_SEED)

train_df = train_loader.dataset.df
vc       = train_df[LABEL].value_counts().sort_index()
n_neg    = int(vc.get(0, 1))
n_pos    = int(vc.get(1, 1))

# =============================================================================
# MODEL
# =============================================================================
model = create_model().to(device)

pos_weight_val = (n_neg / n_pos) ** 0.5 if n_pos > 0 else 1.0
pos_weight     = torch.tensor([pos_weight_val], dtype=torch.float).to(device)
criterion      = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam([
    {"params": model.features.parameters(),   "lr": LR_FEATURES},
    {"params": model.classifier.parameters(), "lr": LR_HEAD},
])

# CosineAnnealingLR: decays LR to eta_min over NUM_ROUNDS steps
cos_scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_ROUNDS, eta_min=1e-6)

print(f"✅ Model ready | pos_weight={pos_weight_val:.3f} | "
      f"train_batches={len(train_loader)} | val_batches={len(val_loader)}\n")

# =============================================================================
# TRAINING LOOP  (NUM_ROUNDS rounds, LOCAL_EPOCHS epochs per round)
# =============================================================================
best_val_loss = float("inf")
best_round    = 0
save_path = SAVE_DIR / f"best_standalone_model_fitz__{config_tag}.pt"

print("=" * 70)
print("🚀 STARTING TRAINING")
print("=" * 70)

for rnd in range(1, NUM_ROUNDS + 1):

    # ── Training ──────────────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0

    for epoch in range(LOCAL_EPOCHS):
        for batch in train_loader:
            imgs   = batch["image"].to(device)
            labels = batch[LABEL].to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(imgs.float())
            loss    = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

    cos_scheduler.step()

    avg_train_loss = running_loss / len(train_loader.dataset)
    print(f"Round {rnd:02d}/{NUM_ROUNDS} | train_loss={avg_train_loss:.4f}", end="")

    # ── Validation ────────────────────────────────────────────────────────────
    model.eval()
    val_loss, val_total, val_correct = 0.0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            imgs   = batch["image"].to(device)
            labels = batch[LABEL].to(device).float().view(-1, 1)
            out    = model(imgs.float())
            bl     = criterion(out, labels)
            val_loss    += bl.item() * imgs.size(0)
            val_total   += imgs.size(0)
            preds        = (torch.sigmoid(out) >= 0.5).float()
            val_correct += (preds == labels).sum().item()

    avg_val_loss = val_loss / val_total
    val_acc      = val_correct / val_total
    print(f" | val_loss={avg_val_loss:.4f} | val_acc={val_acc:.4f}", end="")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_round    = rnd
        torch.save(model.state_dict(), save_path)
        print(f" | ✅ NEW BEST  → {save_path}")
    else:
        print(f" | best so far: round {best_round} ({best_val_loss:.4f})")

    sys.stdout.flush()
    torch.cuda.empty_cache()

# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 70)
print("🏁 TRAINING COMPLETE")
print(f"   Best model : Round {best_round}, Val loss {best_val_loss:.4f}")
print(f"   Saved at   : {save_path}")
print("=" * 70)