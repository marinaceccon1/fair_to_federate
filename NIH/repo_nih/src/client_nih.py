import flwr as fl
import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root: three levels up from this file (NIH/repo/src/client_nih.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID index")
    args = parser.parse_args()
    cid = args.cid

    grpc_port   = int(os.environ.get("GRPC_PORT", "8080"))
    config_path = os.environ.get(
        "CONFIG_PATH",
        str(REPO_ROOT / "repo_nih" / "experiments" / "current_config.json")
    )

    print(f"\n🚀 Starting client {cid} (port={grpc_port}, config={config_path})...")

    from src.data_setup_nih import NIHClient, num_clients, setup_for_client

    if cid >= num_clients:
        raise ValueError(
            f"❌ Invalid client ID {cid}. "
            f"Config only defines {num_clients} clients."
        )

    train_loader, val_loader, flip_train, flip_val = setup_for_client(cid)

    client = NIHClient(
        client_id=cid,
        train_loader=train_loader,
        val_loader=val_loader,
        women_to_flip_train=flip_train,
        women_to_flip_val=flip_val
    )

    server_address = f"localhost:{grpc_port}"
    print(f"✅ Client {cid} connecting to {server_address}...\n")

    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
        grpc_max_message_length=1024 * 1024 * 1024,
    )
