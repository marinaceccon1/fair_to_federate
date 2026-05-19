"""
client_nih_2clients.py
=======================
Flower client entry point for the 2-client federated learning setup.

Usage (launched automatically by run_experiments_2clients.py):

    GRPC_PORT=8080 CONFIG_PATH=experiments/current_config_2clients_worker0.json \\
    NIH_DATA_PATH=/path/to/images \\
        python client_nih_2clients.py --cid 0
"""

import argparse
import os
import sys
from pathlib import Path

import flwr as fl

# ---------------------------------------------------------------------------
# Repo root: three levels up from this file
# (NIH/repo/federated_2clients/client_nih_2clients.py)
# Also add federated/ so data_setup_nih_Nclients can be imported directly.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "repo_nih" / "federated"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID (0-based)")
    args = parser.parse_args()
    cid = args.cid

    grpc_port   = int(os.environ.get("GRPC_PORT", "8080"))
    config_path = os.environ.get(
        "CONFIG_PATH",
        str(REPO_ROOT / "repo_nih" / "experiments" / "current_config_2clients_worker0.json")
    )

    print(f"\n🚀 Starting client {cid} (port={grpc_port}, config={config_path})...")

    # Import reads CONFIG_PATH and NIH_DATA_PATH at import time
    from data_setup_nih_Nclients import NIHClient, num_clients, setup_for_client

    if cid >= num_clients:
        raise ValueError(
            f"❌ Invalid client ID {cid}. "
            f"Config defines {num_clients} clients (0–{num_clients - 1})."
        )

    train_loader, val_loader, flip_train, flip_val = setup_for_client(cid)

    client = NIHClient(
        client_id=cid,
        train_loader=train_loader,
        val_loader=val_loader,
        women_to_flip_train=flip_train,
        women_to_flip_val=flip_val,
    )

    server_address = f"localhost:{grpc_port}"
    print(f"✅ Client {cid} connecting to {server_address}...\n")

    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
        grpc_max_message_length=1024 * 1024 * 1024,
    )
