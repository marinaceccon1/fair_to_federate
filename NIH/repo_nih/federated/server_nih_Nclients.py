"""
server_nih_Nclients.py
=======================
Flower server for the N-client federated learning setup.

Reads NUM_CLIENTS from the environment to support 3, 5, 7, or any other
client count without code changes.  All paths are relative to the repo
root so the script works regardless of the working directory.

Environment variables
---------------------
NUM_CLIENTS     Number of federated clients (default: 7).
                Must match the number of entries in the config file.
CONFIG_PATH     Path to the per-experiment JSON config file.
                Default: <repo_root>/experiments/current_config_<N>clients_worker<W>.json
                (set automatically by run_experiments_cumulative_Nclients.py)
GRPC_PORT       gRPC port the server listens on (default: 8080).
                Each parallel worker should use a unique port.
NUM_ROUNDS      Number of federated learning rounds (default: 25).
CLIENT_FRACTION Fraction of clients sampled per round (default: 1.0).
MODELS_DIR      Directory where best models are saved.
                Default: <repo_root>/best_models_<N>clients

Usage (launched automatically by run_experiments_cumulative_Nclients.py):

    NUM_CLIENTS=5 GRPC_PORT=8081 CONFIG_PATH=experiments/current_config_5clients_worker1.json \\
        python federated/server_nih_Nclients.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import flwr as fl
import torch

# ---------------------------------------------------------------------------
# Repo root: one level up from this file  (repo/federated/server_nih_Nclients.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from src.models.model import create_model

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
NUM_CLIENTS     = int(os.environ.get("NUM_CLIENTS", "7"))
GRPC_PORT       = int(os.environ.get("GRPC_PORT",   "8080"))
NUM_ROUNDS      = int(os.environ.get("NUM_ROUNDS",  "25"))
CLIENT_FRACTION = float(os.environ.get("CLIENT_FRACTION", "1.0"))

# Config file: runner sets CONFIG_PATH before spawning the server
_default_config = (
    REPO_ROOT / "repo_nih" / "experiments"
    / f"current_config_{NUM_CLIENTS}clients_worker"
      f"{os.environ.get('WORKER_ID', '0')}.json"
)
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(_default_config)))

# Model save directory: relative to repo root unless overridden
_default_models_dir = REPO_ROOT / "repo_nih" / f"best_models_{NUM_CLIENTS}clients"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(_default_models_dir)))

# =============================================================================
# LOAD EXPERIMENT CONFIGURATION
# =============================================================================
if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"❌ Config file not found: {CONFIG_PATH}\n"
        f"   Set CONFIG_PATH or let run_experiments_cumulative_Nclients.py "
        f"create it automatically."
    )

with open(CONFIG_PATH) as f:
    experiment_config: List[Dict] = json.load(f)

if len(experiment_config) != NUM_CLIENTS:
    raise ValueError(
        f"❌ Config contains {len(experiment_config)} client(s), "
        f"but NUM_CLIENTS={NUM_CLIENTS}. "
        f"Update the NUM_CLIENTS environment variable or fix the config file."
    )

MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("\n🧪 SERVER: Loaded experiment configuration")
print(f"   NUM_CLIENTS  : {NUM_CLIENTS}")
print(f"   GRPC_PORT    : {GRPC_PORT}")
print(f"   NUM_ROUNDS   : {NUM_ROUNDS}")
print(f"   CONFIG_PATH  : {CONFIG_PATH}")
print(f"   MODELS_DIR   : {MODELS_DIR}")
for i, c in enumerate(experiment_config):
    print(
        f"   Client {i}: portion={c['portion']:.2f}, "
        f"gender={c['gender']}, flip={c['flip_frac']}"
    )


# =============================================================================
# FILENAME TAG
# =============================================================================
def format_config_tag(client_configs: List[Dict]) -> str:
    """
    Build a compact, filesystem-safe tag that encodes all client configs.

    Example (3 clients):
        C0_p0.12_M100F0_flip0.0__C1_p0.06_M50F50_flip0.15__C2_p0.03_M0F100_flip0.3
    """
    parts = []
    for i, c in enumerate(client_configs):
        parts.append(
            f"C{i}"
            f"_p{c['portion']}"
            f"_M{int(c['gender']['Male']   * 100)}"
            f"F{int(c['gender']['Female']  * 100)}"
            f"_flip{c['flip_frac']}"
        )
    return "__".join(parts)


# =============================================================================
# CUSTOM FEDAVG STRATEGY — saves the global model whenever val loss improves
# =============================================================================
class SaveBestModelStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg variant that tracks the per-round aggregated validation loss and
    saves the global model weights to disk whenever a new best is found.

    The saved filename embeds the full experiment config so that models from
    different runs never overwrite each other:

        best_global_model_nih_<N>clients__<config_tag>.pt
    """

    def __init__(self, num_clients: int, config_tag: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_clients          = num_clients
        self.config_tag           = config_tag
        self.best_val_loss        = float("inf")
        self.best_round           = 0
        self.current_round_params = None
        self.current_round        = 0

    # -------------------------------------------------------------------------
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(
            rnd, results, failures
        )
        if aggregated_parameters is not None:
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            model    = create_model()
            keys     = list(model.state_dict().keys())
            self.current_round_params = {
                k: torch.tensor(v) for k, v in zip(keys, ndarrays)
            }
            self.current_round = rnd
        return aggregated_parameters, metrics

    # -------------------------------------------------------------------------
    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)

        if aggregated_result is not None and results:
            total_examples  = sum(r.num_examples for _, r in results)
            weighted_losses = [r.loss * r.num_examples for _, r in results]
            avg_val_loss    = sum(weighted_losses) / total_examples

            print(f"\n📊 Round {rnd}/{NUM_ROUNDS}  |  Avg val loss: {avg_val_loss:.4f}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_round    = rnd

                filename = (
                    MODELS_DIR
                    / f"best_global_model_nih_{self.num_clients}clients"
                      f"__{self.config_tag}.pt"
                )
                torch.save(self.current_round_params, filename)

                print(f"✅ NEW BEST  — Round {rnd}  Loss {avg_val_loss:.4f}")
                print(f"   Saved → {filename}")
            else:
                print(
                    f"   Best so far: Round {self.best_round}, "
                    f"Loss {self.best_val_loss:.4f}"
                )

        return aggregated_result


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    config_tag = format_config_tag(experiment_config)
    print(f"\n🏷  Model tag : {config_tag}\n")

    strategy = SaveBestModelStrategy(
        num_clients=NUM_CLIENTS,
        config_tag=config_tag,
        fraction_fit=CLIENT_FRACTION,
        fraction_evaluate=CLIENT_FRACTION,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"local_epochs": 1},
    )

    server_address = f"0.0.0.0:{GRPC_PORT}"
    print(f"✅ Starting Flower server on {server_address}\n")

    try:
        fl.server.start_server(
            server_address=server_address,
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        )
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🏁 TRAINING COMPLETE")
    print(
        f"   Best model : Round {strategy.best_round}, "
        f"Loss {strategy.best_val_loss:.4f}"
    )
    print(f"   Saved in   : {MODELS_DIR}")
    print("=" * 70)
