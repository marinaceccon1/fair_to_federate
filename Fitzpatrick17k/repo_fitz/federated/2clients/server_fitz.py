import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root anchored on __file__
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flwr as fl
import torch
from typing import List, Dict
from src.data_setup_fitz import create_model

# =============================================================================
# LOAD CONFIGURATION
# =============================================================================
_default_config = str(REPO_ROOT / "experiments_fitz" / "current_config.json")
CONFIG_PATH = os.environ.get("CONFIG_PATH", _default_config)
GRPC_PORT   = int(os.environ.get("GRPC_PORT", "8080"))

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"❌ Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH) as f:
    experiment_config = json.load(f)

print("\n🧪 SERVER: Loaded experiment configuration")
for i, c in enumerate(experiment_config):
    print(f"  Client {i}: portion={c['portion']}, "
          f"composition={c['composition']}, flip={c['flip_frac']}")

# =============================================================================
# CONFIG TAG
# =============================================================================
def format_config_tag(client_configs: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(client_configs):
        comp = c["composition"]
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

# =============================================================================
# FEDERATED PARAMETERS  — mirrors train_standalone_fitz.py
# =============================================================================
# Directory where best models are saved — override via MODELS_DIR env var
_default_models_dir = str(REPO_ROOT / "best_models_fitz")
MODELS_DIR = Path(os.environ.get("MODELS_DIR", _default_models_dir))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FEDERATED PARAMETERS
# =============================================================================
NUM_ROUNDS      = int(os.environ.get("NUM_ROUNDS", "30"))
CLIENT_FRACTION = float(os.environ.get("CLIENT_FRACTION", "1.0"))

# =============================================================================
# STRATEGY
# =============================================================================
class SaveBestModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, config_tag: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_val_loss        = float("inf")
        self.best_round           = 0
        self.current_round_params = None
        self.current_round        = 0
        self.config_tag           = config_tag

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Forward the current round number to clients so they can step their
        CosineAnnealingLR scheduler at the right position.
        """
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )
        # Inject current_round into every client's fit config
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
            total_examples  = sum(fit_res.num_examples for _, fit_res in results)
            weighted_losses = [
                fit_res.loss * fit_res.num_examples for _, fit_res in results
            ]
            avg_val_loss = sum(weighted_losses) / total_examples

            print(f"\n📊 Round {rnd}")
            print(f"   Avg validation loss: {avg_val_loss:.4f}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_round    = rnd
                filename = MODELS_DIR / f"best_global_model_fitz__{self.config_tag}.pt"
                torch.save(self.current_round_params, filename)
                print("✅ NEW BEST MODEL")
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
    print(f"\n🏷  Model tag: {config_tag}\n")

    strategy = SaveBestModelStrategy(
        config_tag             = config_tag,
        fraction_fit           = CLIENT_FRACTION,
        fraction_evaluate      = CLIENT_FRACTION,
        min_fit_clients        = len(experiment_config),
        min_evaluate_clients   = len(experiment_config),
        min_available_clients  = len(experiment_config),
    )

    print(f"✅ Starting Flower server on 0.0.0.0:{GRPC_PORT}\n")

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
    print("🏁 TRAINING COMPLETE")
    print(f"Best model: Round {strategy.best_round}, "
          f"Loss {strategy.best_val_loss:.4f}")
    print("=" * 70)