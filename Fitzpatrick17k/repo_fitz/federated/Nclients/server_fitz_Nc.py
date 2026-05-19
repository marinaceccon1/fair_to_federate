"""
server_fitz_Nc.py
──────────────────
Flower server for N-client (N > 2) Fitzpatrick FL setups.

Expects NUM_CLIENTS clients (passed via --num-clients).
Saves best models with an Nc prefix in the filename so they never collide
with artefacts produced by runs with different N.

Usage:
    python server_fitz_Nc.py --num-clients 3
    python server_fitz_Nc.py --num-clients 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import flwr as fl
import torch
from typing import List, Dict

# Repo root: two levels up from this file (repo/federated/Nclients/)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_setup_fitz import create_model

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--num-clients", type=int, required=True,
                    help="Number of FL clients expected by this server")
args = parser.parse_args()

NUM_CLIENTS = args.num_clients

# =============================================================================
# LOAD CONFIGURATION
# =============================================================================
_default_config = str(_REPO_ROOT / f"experiments_fitz_{NUM_CLIENTS}c" /
                      "current_config.json")
CONFIG_PATH = os.environ.get("CONFIG_PATH", _default_config)
GRPC_PORT   = int(os.environ.get("GRPC_PORT", "8080"))
MODELS_DIR  = Path(os.environ.get(
    "MODELS_DIR", str(_REPO_ROOT / f"best_models_fitz_{NUM_CLIENTS}c")
))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"❌ Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH) as f:
    experiment_config = json.load(f)

print(f"\n🧪 SERVER ({NUM_CLIENTS}c): Loaded experiment configuration")
for i, c in enumerate(experiment_config):
    print(f"  Client {i}: portion={c['portion']}, "
          f"composition={c['composition']}, flip={c['flip_frac']}")

# =============================================================================
# CONFIG TAG
# =============================================================================
def format_config_tag(client_configs: List[Dict]) -> str:
    n = len(client_configs)
    parts = []
    for i, c in enumerate(client_configs):
        comp     = c["composition"]
        comp_str = (
            f"14x{int(comp['fitz_14'] * 100)}"
            f"_56x{int(comp['fitz_56'] * 100)}"
        )
        parts.append(f"C{i}_p{c['portion']}_{comp_str}_flip{c['flip_frac']}")
    # Prefix with Nc to distinguish models across different federation sizes
    return f"{n}c__" + "__".join(parts)


# =============================================================================
# FEDERATED PARAMETERS
# =============================================================================
NUM_ROUNDS      = int(os.environ.get("NUM_ROUNDS", "30"))
CLIENT_FRACTION = 1.0

# =============================================================================
# STRATEGY
# =============================================================================
class SaveBestModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, config_tag: str, models_dir: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_val_loss        = float("inf")
        self.best_round           = 0
        self.current_round_params = None
        self.current_round        = 0
        self.config_tag           = config_tag
        self.models_dir           = models_dir

    def configure_fit(self, server_round, parameters, client_manager):
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )
        return [
            (client, fl.common.FitIns(fit_ins.parameters,
                                      {**fit_ins.config,
                                       "current_round": server_round}))
            for client, fit_ins in client_instructions
        ]

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

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)

        if aggregated_result is not None and results:
            total_examples  = sum(r.num_examples for _, r in results)
            weighted_losses = [r.loss * r.num_examples for _, r in results]
            avg_val_loss    = sum(weighted_losses) / total_examples

            print(f"\n📊 Round {rnd}")
            print(f"   Avg validation loss: {avg_val_loss:.4f}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_round    = rnd
                filename = self.models_dir / \
                           f"best_global_model_fitz__{self.config_tag}.pt"
                torch.save(self.current_round_params, filename)
                print(f"✅ NEW BEST MODEL ({NUM_CLIENTS}c)")
                print(f"   Round: {rnd}")
                print(f"   Loss:  {avg_val_loss:.4f}")
                print(f"   Saved: {filename}")
            else:
                print(f"   Best so far: Round {self.best_round}, "
                      f"Loss {self.best_val_loss:.4f}")

        return aggregated_result


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    config_tag = format_config_tag(experiment_config)
    print(f"\n🏷  Model tag ({NUM_CLIENTS}c): {config_tag}\n")

    strategy = SaveBestModelStrategy(
        config_tag            = config_tag,
        models_dir            = MODELS_DIR,
        fraction_fit          = CLIENT_FRACTION,
        fraction_evaluate     = CLIENT_FRACTION,
        min_fit_clients       = NUM_CLIENTS,
        min_evaluate_clients  = NUM_CLIENTS,
        min_available_clients = NUM_CLIENTS,
    )

    print(f"✅ Starting Flower server ({NUM_CLIENTS}c) on 0.0.0.0:{GRPC_PORT}\n")

    try:
        fl.server.start_server(
            server_address = f"0.0.0.0:{GRPC_PORT}",
            strategy       = strategy,
            config         = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        )
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\n" + "=" * 70)
    print(f"🏁 TRAINING COMPLETE ({NUM_CLIENTS}c)")
    print(f"Best model: Round {strategy.best_round}, "
          f"Loss {strategy.best_val_loss:.4f}")
    print("=" * 70)
