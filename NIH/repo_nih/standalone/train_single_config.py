import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root: three levels up from this file (NIH/repo/standalone/train_single_config.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

SAVE_DIR = REPO_ROOT / "repo_nih" / "single_models"


def format_config_tag(config):
    """Format a single client config into a unique identifier."""
    return (f"p{config['portion']}_M{int(config['gender']['Male']*100)}"
            f"F{int(config['gender']['Female']*100)}_flip{config['flip_frac']}")


def train_single_config(config_dict, num_epochs=30):
    """
    Train a standalone model on a single client configuration.

    The target config is placed at position 0 in a two-client setup,
    with a minimal dummy client at position 1. This ensures data
    allocation is consistent with what the same config would receive
    during federated training.

    Args:
        config_dict: Single client config dict with keys 'portion',
                     'gender', 'flip_frac'.
        num_epochs:  Number of training epochs.
    """
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Models will be saved to: {SAVE_DIR}")

    dummy_config = {
        "portion":   0.03,
        "gender":    {"Male": 0.5, "Female": 0.5},
        "flip_frac": 0.0
    }
    client_configs = [config_dict, dummy_config]
    config_tag     = format_config_tag(config_dict)

    # Write config and point CONFIG_PATH to it before importing data_setup_nih
    config_path = REPO_ROOT / "repo_nih" / "experiments" / "current_config.json"
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(client_configs, f, indent=2)
    os.environ["CONFIG_PATH"] = str(config_path)

    from src.data_setup_nih import setup_for_client, device, target_pathologies
    from src.models.model import create_model

    print(f"\n{'='*70}")
    print(f"🎯 Training Configuration: {config_tag}")
    print(f"{'='*70}")
    print(f"  Portion:        {config_dict['portion']}")
    print(f"  Gender:         M={config_dict['gender']['Male']:.2f}, "
          f"F={config_dict['gender']['Female']:.2f}")
    print(f"  Flip fraction:  {config_dict['flip_frac']}")
    print(f"{'='*70}\n")

    train_loader, val_loader, flip_train, flip_val = setup_for_client(0)

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Flipped train: {len(flip_train)}")
    print(f"Flipped val:   {len(flip_val)}\n")

    model     = create_model().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    best_val_loss = float('inf')
    best_epoch    = 0

    for epoch in range(num_epochs):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            img          = batch[0].to(device)
            target_batch = batch[1].to(device)
            idx          = batch[2]

            new_target_batch = torch.zeros_like(target_batch)
            for j in range(len(new_target_batch)):
                idx_value = idx[j]
                if isinstance(idx_value, torch.Tensor):
                    idx_value = idx_value.item()
                if idx_value not in flip_train:
                    new_target_batch[j] = target_batch[j]

            output = model(img)
            loss   = criterion(output, new_target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                img          = batch[0].to(device)
                target_batch = batch[1].to(device)
                idx          = batch[2]

                new_target_batch = torch.zeros_like(target_batch)
                for j in range(len(new_target_batch)):
                    idx_value = idx[j]
                    if isinstance(idx_value, torch.Tensor):
                        idx_value = idx_value.item()
                    if idx_value not in flip_val:
                        new_target_batch[j] = target_batch[j]

                output   = model(img)
                loss     = criterion(output, new_target_batch)
                val_loss += loss.item() * img.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch    = epoch + 1
            filename      = SAVE_DIR / f"best_model__{config_tag}.pt"
            torch.save(model.state_dict(), filename)
            print(f"  ✅ New best! Saved to {filename}")

    print(f"\n{'='*70}")
    print(f"🏁 Training Complete")
    print(f"   Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")
    print(f"{'='*70}\n")

    return best_val_loss


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON file containing a single client config")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    train_single_config(config, args.epochs)
