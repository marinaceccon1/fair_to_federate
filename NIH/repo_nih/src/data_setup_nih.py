import sys
import os
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root: three levels up from this file (NIH/repo/src/data_setup_nih.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import flwr as fl
from collections import OrderedDict

from src.data.utils import import_nih_dfs, import_cxp_dfs
from src.federated.utils import gender_partition, select_women_to_flip, preprocess_nih_cxp
from src.data.dataset import CheXpertAndNIH
from src.models.model import create_model

# =============================================================================
# IMAGE DATA PATH
# Set the NIH_DATA_PATH environment variable to the root folder containing
# the NIH ChestX-ray14 images, e.g.:
#   export NIH_DATA_PATH=/path/to/your/images
# =============================================================================
path_image = os.environ.get("NIH_DATA_PATH")
if path_image is None:
    raise EnvironmentError(
        "❌ NIH_DATA_PATH environment variable is not set. "
        "Please set it to the root folder containing the NIH ChestX-ray14 images, e.g.:\n"
        "  export NIH_DATA_PATH=/path/to/your/images"
    )

# =============================================================================
# LOAD CONFIGURATION
# =============================================================================
CONFIG_PATH = os.environ.get("CONFIG_PATH", str(REPO_ROOT / "repo_nih" / "experiments" / "current_config.json"))

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"❌ Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH) as f:
    client_configs = json.load(f)

num_clients               = len(client_configs)
client_portions           = [c["portion"]   for c in client_configs]
client_gender_proportions = [c["gender"]    for c in client_configs]
client_flip_fractions     = [c["flip_frac"] for c in client_configs]

# =============================================================================
# SHARED CONSTANTS
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pathologies = [
    'Lung Opacity', 'Atelectasis', 'Cardiomegaly', 'Consolidation',
    'Edema', 'Effusion', 'Enlarged Cardiomediastinum', 'Fracture',
    'Lung Lesion', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
]

target_pathologies = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Pneumonia', 'Pneumothorax',
]

normalize_common = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform_common = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize_common
])

val_transform_common = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize_common
])


# =============================================================================
# PER-CLIENT SETUP
# =============================================================================
def setup_for_client(cid: int):
    """
    Load, partition, and return (train_loader, val_loader, flip_train, flip_val)
    for the single client `cid`.

    The full partition is computed so that every client's slice is identical
    to what it would be if all clients were built together (same seed, same
    portions). Everything except cid's slice is then immediately discarded.
    """
    print(f"\n🧪 EXPERIMENT CONFIGURATION (building for client {cid})")
    for i, c in enumerate(client_configs):
        marker = " ◄ THIS CLIENT" if i == cid else ""
        print(f"  Client {i} | Portion={c['portion']:.2f} | "
              f"Gender={c['gender']} | Flip={c['flip_frac']}{marker}")

    print(f"\n📂 Loading NIH and CXP dataframes...")
    train_df_nih, val_df_nih, _ = import_nih_dfs(str(REPO_ROOT))
    train_df_cxp, val_df_cxp, _ = import_cxp_dfs(str(REPO_ROOT))

    print(f"⚙️  Preprocessing...")
    train_df_nih_mod, _ = preprocess_nih_cxp(
        train_df_nih, train_df_cxp, pathologies, target_pathologies
    )
    val_df_nih_mod, _ = preprocess_nih_cxp(
        val_df_nih, val_df_cxp, pathologies, target_pathologies
    )
    del train_df_nih, val_df_nih, train_df_cxp, val_df_cxp

    print(f"📊 Partitioning — keeping only client {cid}'s slice...")
    nih_clients_train, _ = gender_partition(
        train_df_nih_mod,
        client_gender_proportions=client_gender_proportions,
        portions=client_portions,
        gender_col='Sex',
        seed=42
    )
    nih_clients_val, _ = gender_partition(
        val_df_nih_mod,
        client_gender_proportions=client_gender_proportions,
        portions=client_portions,
        gender_col='Sex',
        seed=42
    )

    df_train = nih_clients_train[cid]
    df_val   = nih_clients_val[cid]
    del nih_clients_train, nih_clients_val, train_df_nih_mod, val_df_nih_mod

    print(f"  Train rows: {len(df_train)}, Val rows: {len(df_val)}")

    print(f"🔀 Selecting women to flip...")
    flip_train = select_women_to_flip(
        [df_train], target_pathologies,
        fracs=[client_flip_fractions[cid]], seed=42
    ).get(0, set())

    flip_val = select_women_to_flip(
        [df_val], target_pathologies,
        fracs=[client_flip_fractions[cid]], seed=42
    ).get(0, set())

    print(f"  Flip train: {len(flip_train)}, Flip val: {len(flip_val)}")

    print(f"🗂️  Creating dataset and dataloader...")
    train_ds = CheXpertAndNIH(df_train, path_image=path_image,
                               transform=train_transform_common)
    val_ds   = CheXpertAndNIH(df_val,   path_image=path_image,
                               transform=val_transform_common)

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

    print(f"✅ Client {cid} data ready | "
          f"train_batches={len(train_loader)} | "
          f"val_batches={len(val_loader)}")

    return train_loader, val_loader, flip_train, flip_val


# =============================================================================
# FLOWER CLIENT
# =============================================================================
class NIHClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, train_loader, val_loader,
                 women_to_flip_train, women_to_flip_val):
        self.client_id           = client_id
        self.model               = create_model().to(device)
        self.train_loader        = train_loader
        self.val_loader          = val_loader
        self.criterion           = nn.BCELoss()
        self.women_to_flip_train = women_to_flip_train
        self.women_to_flip_val   = women_to_flip_val

        print(f"✅ Client {client_id} initialized | "
              f"device={next(self.model.parameters()).device} | "
              f"train_batches={len(train_loader)} | "
              f"val_batches={len(val_loader)} | "
              f"flip_train={len(women_to_flip_train)} | "
              f"flip_val={len(women_to_flip_val)}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        epochs    = config.get("local_epochs", 1)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch in self.train_loader:
                img          = batch[0].to(device)
                target_batch = batch[1].to(device)
                idx          = batch[2]

                new_target_batch = torch.zeros_like(target_batch)
                for j in range(len(new_target_batch)):
                    idx_value = idx[j]
                    if isinstance(idx_value, torch.Tensor):
                        idx_value = idx_value.item()
                    if idx_value not in self.women_to_flip_train:
                        new_target_batch[j] = target_batch[j]

                output_batch = self.model(img)
                loss         = self.criterion(output_batch, new_target_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"[Client {self.client_id}] Epoch {epoch+1}/{epochs} - "
                  f"Loss: {avg_loss:.4f}")

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        loss, total = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                img          = batch[0].to(device)
                target_batch = batch[1].to(device)
                idx          = batch[2]

                new_target_batch = torch.zeros_like(target_batch)
                for j in range(len(new_target_batch)):
                    idx_value = idx[j]
                    if isinstance(idx_value, torch.Tensor):
                        idx_value = idx_value.item()
                    if idx_value not in self.women_to_flip_val:
                        new_target_batch[j] = target_batch[j]

                output_batch = self.model(img)
                batch_loss   = self.criterion(output_batch, new_target_batch)

                loss  += batch_loss.item() * img.size(0)
                total += img.size(0)

        avg_loss = loss / total
        print(f"[Client {self.client_id}] Evaluation loss: {avg_loss:.4f}")
        return float(avg_loss), total, {"val_loss": float(avg_loss)}
