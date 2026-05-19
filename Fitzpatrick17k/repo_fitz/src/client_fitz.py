"""
client_fitz.py
──────────────
FL client mirroring train_standalone_fitz.py exactly:
  • EfficientNet-B2, same classifier head
  • Adam with two param groups (LR_FEATURES=1e-5, LR_HEAD=1e-4)
  • Optimizer and CosineAnnealingLR(T_max=30, eta_min=1e-6) initialised ONCE
    and kept alive across rounds — scheduler is stepped once per round
  • BCEWithLogitsLoss with pos_weight = (n_neg/n_pos)**0.5
  • Gradient clipping max_norm=1.0
  • 1 local epoch per round

GPU memory management
─────────────────────
When two clients share the same physical GPU (as in the 5-client / 3-GPU
layout), both EfficientNet-B2 models would normally occupy GPU memory
simultaneously, which may exceed the card's capacity.

To avoid this the client uses a two-part strategy:

  1. CPU offload — the model is kept on CPU between rounds.  It is only
     moved to GPU for the duration of fit() / evaluate(), then immediately
     moved back.  This means only one model per GPU needs to be resident
     at any given time.

  2. Per-GPU file lock — a fcntl.flock() exclusive lock on a per-GPU
     lock file (e.g. /tmp/gpu_lock_0) ensures that two clients assigned
     to the same GPU never execute their GPU work concurrently.  The
     second client blocks on the lock until the first has finished and
     moved its model back to CPU.

The locking is transparent to Flower: from the server's perspective every
client returns results at the end of the round as usual; clients sharing a
GPU simply run sequentially rather than in parallel.

This mechanism is activated only when CUDA is available.  On CPU-only
machines it is a no-op and behaviour is identical to the original client.
"""

import fcntl
import flwr as fl
import argparse
import os
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import contextmanager
from pathlib import Path
from torch.optim import lr_scheduler
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Repo root anchored on __file__
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# =============================================================================
# HYPER-PARAMETERS  — must match train_standalone_fitz.py exactly
# =============================================================================
NUM_ROUNDS    = int(os.environ.get("NUM_ROUNDS", "30"))  # T_max for cosine scheduler — must match server
LOCAL_EPOCHS  = 1
LR_FEATURES   = 1e-5
LR_HEAD       = 1e-4
MAX_GRAD_NORM = 1.0

# =============================================================================
# PER-GPU SERIALISATION LOCK
# =============================================================================
_GPU_LOCK_DIR = Path(os.environ.get("GPU_LOCK_DIR", "/tmp"))

@contextmanager
def _gpu_lock(gpu_index: int):
    """
    Acquire an exclusive advisory lock on /tmp/gpu_lock_<gpu_index> before
    doing any GPU work, and release it afterwards.

    Because CUDA_VISIBLE_DEVICES remaps device indices, we use the
    *physical* GPU index passed explicitly by the runner (GPU_PHYS_ID env
    var) rather than the logical cuda:0 index seen inside this process.
    This ensures clients on different physical GPUs use different lock files
    and never block each other unnecessarily.

    On CPU-only machines (gpu_index == -1) this is a no-op.
    """
    if gpu_index < 0:
        yield
        return

    lock_path = _GPU_LOCK_DIR / f"gpu_lock_{gpu_index}"
    fh = open(lock_path, "a")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX)   # blocks until the other client is done
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


# =============================================================================
# FL CLIENT
# =============================================================================
class FitzpatrickClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_loader, val_loader,
                 group_train_sizes, n_pos, n_neg, gpu_phys_id: int = -1):
        self.client_id         = client_id
        self.train_loader      = train_loader
        self.val_loader        = val_loader
        self.group_train_sizes = group_train_sizes
        self.gpu_phys_id       = gpu_phys_id   # physical GPU index for locking

        # The logical device seen by this process (always cuda:0 when
        # CUDA_VISIBLE_DEVICES is set to a single GPU by the runner).
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        from src.data_setup_fitz import create_model, LABEL
        self.LABEL = LABEL

        # Model is initialised on CPU; moved to GPU only during computation.
        self.model = create_model()   # stays on CPU at rest

        self.pos_weight_val = (n_neg / n_pos) ** 0.5 if n_pos > 0 else 1.0
        # criterion is recreated on the right device inside _to_gpu/_to_cpu
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight_val], dtype=torch.float)
        )

        # ── Optimizer and scheduler created ONCE and persisted across rounds ──
        self.optimizer = optim.Adam([
            {"params": self.model.features.parameters(),   "lr": LR_FEATURES},
            {"params": self.model.classifier.parameters(), "lr": LR_HEAD},
        ])
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=NUM_ROUNDS, eta_min=1e-6
        )

        print(f"✅ Client {client_id} initialised | "
              f"compute_device={self.device} | "
              f"gpu_phys_id={gpu_phys_id} | "
              f"train_batches={len(train_loader)} | "
              f"val_batches={len(val_loader)} | "
              f"pos_weight={self.pos_weight_val:.3f}")

    # ── Device helpers ────────────────────────────────────────────────────────

    def _to_gpu(self):
        """Move model and criterion to the compute device."""
        self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [self.pos_weight_val], dtype=torch.float, device=self.device
            )
        )

    def _to_cpu(self):
        """Move model back to CPU and fully release the GPU allocation."""
        self.model.to("cpu")
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight_val], dtype=torch.float)
        )
        if self.device.type == "cuda":
            # synchronize() blocks until all CUDA kernels have finished,
            # ensuring no GPU work is still in flight when we clear the cache.
            # Without this, empty_cache() may not reclaim all reserved blocks
            # and the next client to acquire the lock finds memory still held.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # ── Flower API ────────────────────────────────────────────────────────────

    def get_parameters(self, config):
        # Model may be on CPU here — that's fine, .cpu().numpy() always works.
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)   # update weights while still on CPU

        local_epochs = config.get("local_epochs", LOCAL_EPOCHS)
        running_loss = 0.0

        with _gpu_lock(self.gpu_phys_id):
            self._to_gpu()
            self.model.train()

            for epoch in range(local_epochs):
                for batch in self.train_loader:
                    imgs   = batch["image"].to(self.device)
                    labels = batch[self.LABEL].to(self.device).float().view(-1, 1)

                    self.optimizer.zero_grad()
                    outputs = self.model(imgs.float())
                    loss    = self.criterion(outputs, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=MAX_GRAD_NORM
                    )
                    self.optimizer.step()
                    running_loss += loss.item() * imgs.size(0)

            # Step scheduler and snapshot weights while still on GPU
            self.scheduler.step()
            result_params = self.get_parameters(config={})

            self._to_cpu()   # release GPU memory before unlocking

        avg_loss   = running_loss / len(self.train_loader.dataset)
        current_lr = self.scheduler.get_last_lr()
        print(f"[Client {self.client_id}] "
              f"Round done | train_loss={avg_loss:.4f} | lr={current_lr}")

        return (
            result_params,
            len(self.train_loader.dataset),
            {"group_train_sizes": json.dumps(self.group_train_sizes)},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)   # update weights while still on CPU

        loss, total, correct = 0.0, 0, 0

        with _gpu_lock(self.gpu_phys_id):
            self._to_gpu()
            self.model.eval()

            with torch.no_grad():
                for batch in self.val_loader:
                    imgs   = batch["image"].to(self.device)
                    labels = batch[self.LABEL].to(self.device).float().view(-1, 1)
                    out    = self.model(imgs.float())
                    bl     = self.criterion(out, labels)
                    loss   += bl.item() * imgs.size(0)
                    total  += imgs.size(0)
                    preds   = (torch.sigmoid(out) >= 0.5).float()
                    correct += (preds == labels).sum().item()

            self._to_cpu()   # release GPU memory before unlocking

        avg_loss = loss / total
        val_acc  = correct / total
        print(f"[Client {self.client_id}] Val loss: {avg_loss:.4f} | Val acc: {val_acc:.4f}")
        return float(avg_loss), total, {"val_loss": float(avg_loss), "val_acc": float(val_acc)}


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid",  type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    grpc_port   = int(os.environ.get("GRPC_PORT", "8080"))
    config_path = os.environ.get(
        "CONFIG_PATH",
        str(REPO_ROOT / "experiments_fitz" / "current_config.json")
    )

    # Physical GPU index forwarded by the runner via env var.
    # Used only for lock-file naming — the actual device seen by this process
    # is always cuda:0 because the runner sets CUDA_VISIBLE_DEVICES.
    gpu_phys_id = int(os.environ.get("GPU_PHYS_ID", "-1"))

    # SHARED_GPU=1 means another client is co-located on this physical GPU.
    # We rebuild the DataLoaders with pin_memory=False and fewer workers to
    # reduce pinned/reserved memory from idle loaders, giving the active
    # client enough room for its forward/backward pass.
    shared_gpu = os.environ.get("SHARED_GPU", "0") == "1"

    print(f"\n\U0001f680 Starting Fitzpatrick client {args.cid} "
          f"(port={grpc_port}, config={config_path}, "
          f"seed={args.seed}, gpu_phys_id={gpu_phys_id}, "
          f"shared_gpu={shared_gpu})...")

    from src.data_setup_fitz import num_clients, setup_for_client, LABEL
    import torch.utils.data as tud

    if args.cid >= num_clients:
        raise ValueError(
            f"\u274c Invalid client ID {args.cid}. "
            f"Config only defines {num_clients} clients."
        )

    train_loader, val_loader, group_train_sizes = setup_for_client(
        args.cid, seed=args.seed
    )

    if shared_gpu:
        # Rebuild loaders with memory-conservative settings for shared GPUs:
        #   - batch_size halved (16 vs 32): cuts peak activation + gradient
        #     memory roughly in half, giving the allocator enough headroom
        #     after the previous client's reserved blocks are released.
        #   - pin_memory=False: avoids reserving GPU-accessible host pages
        #     while this client is idle (waiting for the lock).
        #   - num_workers=2: fewer prefetch workers, less background memory.
        from src.data_setup_fitz import make_weighted_sampler, BATCH_SIZE
        shared_batch = max(1, BATCH_SIZE // 2)   # 16 when BATCH_SIZE=32
        train_ds = train_loader.dataset
        val_ds   = val_loader.dataset
        sampler  = make_weighted_sampler(train_ds.df, LABEL)
        train_loader = tud.DataLoader(
            train_ds, batch_size=shared_batch, shuffle=False,
            sampler=sampler, num_workers=2, pin_memory=False,
        )
        val_loader = tud.DataLoader(
            val_ds, batch_size=shared_batch, shuffle=False,
            num_workers=2, pin_memory=False,
        )
        print(f"  ℹ️  Shared-GPU mode: batch={shared_batch}, "
              f"pin_memory=False, num_workers=2")

    import pandas as pd
    train_df = train_loader.dataset.df
    vc    = train_df[LABEL].value_counts().sort_index()
    n_neg = int(vc.get(0, 1))
    n_pos = int(vc.get(1, 1))

    client = FitzpatrickClient(
        client_id         = args.cid,
        train_loader      = train_loader,
        val_loader        = val_loader,
        group_train_sizes = group_train_sizes,
        n_pos             = n_pos,
        n_neg             = n_neg,
        gpu_phys_id       = gpu_phys_id,
    )

    server_address = f"localhost:{grpc_port}"
    print(f"✅ Client {args.cid} connecting to {server_address}...\n")

    fl.client.start_numpy_client(
        server_address          = server_address,
        client                  = client,
        grpc_max_message_length = 1024 * 1024 * 1024,
    )