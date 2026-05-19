"""
run_experiments_fitz_Nc.py
───────────────────────────
Experiment runner for N-client (N > 2) Fitzpatrick FL setups.

GPU layout must be specified via --client-gpus. Examples:
    3 clients across 3 GPUs:  --client-gpus 0 1 2
    5 clients across 3 GPUs:  --client-gpus 0 0 1 1 2

The server can optionally share a GPU with a client (--server-gpu) or run
CPU-only when --server-gpu is omitted.

Usage:
    python run_experiments_fitz_Nc.py --num-clients 3 --client-gpus 0 1 2
    python run_experiments_fitz_Nc.py --num-clients 5 --client-gpus 0 0 1 1 2 \\
                                      --worker-id 1 --server-cpu-only
"""

import argparse
import json
import subprocess
import time
import sys
import signal
import os
from pathlib import Path
from collections import Counter

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(
    description="Fitzpatrick FL N-client experiment runner"
)
parser.add_argument("--num-clients", type=int, required=True,
                    help="Total number of FL clients (must be > 2)")
parser.add_argument("--client-gpus", type=int, nargs="+", required=True,
                    help="GPU index for each client, e.g. --client-gpus 0 1 2")
parser.add_argument("--server-gpu", type=int, default=None,
                    help="GPU index for the server. Omit for CPU-only server.")
parser.add_argument("--worker-id", type=int, default=0,
                    help="Unique integer ID for this runner process (default: 0)")
parser.add_argument("--base-port", type=int, default=8080,
                    help="Base gRPC port; worker i uses base-port + i (default: 8080)")
parser.add_argument("--data-seed", type=int, default=42,
                    help="Seed for reproducible val/test splits (default: 42)")
parser.add_argument("--random-order", action="store_true", default=True,
                    help="Stratified random sampling (default: True)")
args = parser.parse_args()

NUM_CLIENTS  = args.num_clients
CLIENT_GPUS  = args.client_gpus
SERVER_GPU   = args.server_gpu
WORKER_ID    = args.worker_id
BASE_PORT    = args.base_port
DATA_SEED    = args.data_seed
RANDOM_ORDER = args.random_order

if NUM_CLIENTS < 3:
    parser.error("--num-clients must be >= 3")
if len(CLIENT_GPUS) != NUM_CLIENTS:
    parser.error(f"--client-gpus must have exactly {NUM_CLIENTS} entries, "
                 f"got {len(CLIENT_GPUS)}")

from experiment_manager_fitz_Nc import (
    sample_unused_config,
    mark_config_used,
    release_reserved_config,
    mark_config_failed,
    num_total_configs,
    num_used_configs,
    TOTAL_BUDGET,
)

GRPC_PORT   = BASE_PORT + WORKER_ID
CONFIG_PATH = Path(f"experiments_fitz_{NUM_CLIENTS}c") / \
              f"current_config_worker{WORKER_ID}.json"

WORKER_ENV = {
    **os.environ,
    "GRPC_PORT":   str(GRPC_PORT),
    "CONFIG_PATH": str(CONFIG_PATH),
}

# Repo root: two levels up from this file (repo/federated/Nclients/)
_REPO_ROOT = Path(__file__).resolve().parents[2]

print("\n" + "=" * 70)
print(f"🔬 FITZPATRICK EXPERIMENT RUNNER  [{NUM_CLIENTS} CLIENTS]")
print("=" * 70)
print(f"Worker ID     : {WORKER_ID}")
print(f"gRPC port     : {GRPC_PORT}")
print(f"Config file   : {CONFIG_PATH}")
print(f"Sampling mode : {'RANDOM (stratified)' if RANDOM_ORDER else 'SEQUENTIAL'}")
print(f"Num clients   : {NUM_CLIENTS}")
print(f"Data seed     : {DATA_SEED}")
if SERVER_GPU is not None:
    print(f"Server GPU    : cuda:{SERVER_GPU}")
else:
    print(f"Server GPU    : none  (CPU-only, CUDA_VISIBLE_DEVICES=\"\")")
for i, g in enumerate(CLIENT_GPUS):
    print(f"Client {i} GPU  : cuda:{g}")
print(f"Total configs : {num_total_configs(NUM_CLIENTS)}")
print(f"Completed     : {num_used_configs(NUM_CLIENTS)}")
print(f"Remaining     : {num_total_configs(NUM_CLIENTS) - num_used_configs(NUM_CLIENTS)}")
print("=" * 70 + "\n")

experiment_count = 0


# =============================================================================
# HELPERS
# =============================================================================
def cleanup_processes(server, clients):
    print("\n🧹 Cleaning up processes...")
    for i, client in enumerate(clients):
        if client.poll() is None:
            try:
                client.terminate()
                client.wait(timeout=3)
            except subprocess.TimeoutExpired:
                print(f"  ⚠️  Force killing client {i}")
                client.kill()
                client.wait()
    if server and server.poll() is None:
        try:
            server.terminate()
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print("  ⚠️  Force killing server")
            server.kill()
            server.wait()
    print("✅ Cleanup complete")


# =============================================================================
# MAIN LOOP
# =============================================================================
while True:
    experiment_count += 1

    try:
        config = sample_unused_config(NUM_CLIENTS, random_order=RANDOM_ORDER)
    except RuntimeError:
        print("\n✅ ALL CONFIGURATIONS COMPLETED!")
        break

    print(f"\n{'=' * 70}")
    print(f"🧪 EXPERIMENT {experiment_count}  [worker {WORKER_ID}] [{NUM_CLIENTS}c]")
    print(f"{'=' * 70}")
    for i, c in enumerate(config):
        print(f"  Client {i}: portion={c['portion']:.2f}, "
              f"composition={c['composition']}, flip={c['flip_frac']}")
    print("=" * 70 + "\n")

    CONFIG_PATH.parent.mkdir(exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📝 Config written to {CONFIG_PATH}")

    server  = None
    clients = []
    success = False

    try:
        # ── Start server ──────────────────────────────────────────────────────
        server_cuda = "" if SERVER_GPU is None else str(SERVER_GPU)
        server_desc = "CPU-only" if SERVER_GPU is None else f"cuda:{SERVER_GPU}"
        print(f"🖥️  Starting server on port {GRPC_PORT} ({server_desc})...")
        print("─" * 70)
        server = subprocess.Popen(
            [sys.executable,
             str(_REPO_ROOT / "federated" / "Nclients" / "server_fitz_Nc.py"),
             "--num-clients", str(NUM_CLIENTS)],
            env={**WORKER_ENV, "CUDA_VISIBLE_DEVICES": server_cuda},
            stdout=None,
            stderr=None,
        )

        print("⏳ Waiting 15s for server initialisation...")
        time.sleep(15)

        if server.poll() is not None:
            print(f"❌ Server died during initialisation "
                  f"(exit code: {server.returncode})")
            raise RuntimeError("Server failed to start")

        # ── Start clients ─────────────────────────────────────────────────────
        gpu_counts = Counter(CLIENT_GPUS)
        print(f"👥 Starting {NUM_CLIENTS} clients...")
        print("─" * 70)
        for i in range(NUM_CLIENTS):
            gpu_id    = CLIENT_GPUS[i]
            is_shared = "1" if gpu_counts[gpu_id] > 1 else "0"
            print(f"  Client {i} → CUDA_VISIBLE_DEVICES={gpu_id} "
                  f"(SHARED_GPU={is_shared})")
            client = subprocess.Popen(
                [sys.executable,
                 str(_REPO_ROOT / "src" / "client_fitz.py"),
                 "--cid",  str(i),
                 "--seed", str(DATA_SEED)],
                env={**WORKER_ENV,
                     "CUDA_VISIBLE_DEVICES":    str(gpu_id),
                     "GPU_PHYS_ID":             str(gpu_id),
                     "SHARED_GPU":              is_shared,
                     "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"},
                stdout=None,
                stderr=None,
            )
            clients.append(client)
            time.sleep(2)

        print("\n⏳ Training in progress...\n")
        print("─" * 70)
        sys.stdout.flush()

        print("⏳ Waiting for server to finish...")
        sys.stdout.flush()
        try:
            server_return = server.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            print("⚠️  Server exceeded 1-hour timeout, killing it")
            server.kill()
            server_return = server.wait()

        print(f"Server exited with code {server_return}")
        sys.stdout.flush()
        time.sleep(1)

        print("\n🛑 Server finished — force stopping clients...")
        sys.stdout.flush()
        for i, c in enumerate(clients):
            if c.poll() is None:
                print(f"  🔪 Killing client {i} (PID: {c.pid})...")
                try:
                    os.kill(c.pid, signal.SIGKILL)
                except ProcessLookupError:
                    print(f"  ℹ️  Client {i} already stopped")
                except Exception as e:
                    print(f"  ⚠️  Error killing client {i}: {e}")

        time.sleep(1)
        for i, c in enumerate(clients):
            try:
                c.wait(timeout=1)
                print(f"  ✅ Client {i} stopped")
            except subprocess.TimeoutExpired:
                print(f"  ⚠️  Client {i} still running after SIGKILL")
            except Exception:
                pass
        sys.stdout.flush()

        success = (server_return == 0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Keyboard interrupt detected!")
        cleanup_processes(server, clients)
        release_reserved_config(NUM_CLIENTS, config)
        print("🛑 Stopping experiment runner")
        break

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        cleanup_processes(server, clients)
        success = False

    # ── Handle result ─────────────────────────────────────────────────────────
    if success:
        print("\n✅ Training completed successfully!")
        mark_config_used(NUM_CLIENTS, config)
        completed = num_used_configs(NUM_CLIENTS)
        print(f"📊 Configuration confirmed ({completed}/{TOTAL_BUDGET})\n")
        if completed >= TOTAL_BUDGET:
            print(f"\n🎯 TARGET REACHED: {TOTAL_BUDGET} experiments completed!")
            break
    else:
        print(f"\n❌ Training failed")
        mark_config_failed(NUM_CLIENTS, config)
        release_reserved_config(NUM_CLIENTS, config)
        print("🔁 Config released — will be retried by this or another worker\n")

    print(f"\n{'=' * 70}")
    print(f"[Worker {WORKER_ID}] Progress: "
          f"{num_used_configs(NUM_CLIENTS)}/{num_total_configs(NUM_CLIENTS)} "
          f"({100 * num_used_configs(NUM_CLIENTS) / num_total_configs(NUM_CLIENTS):.1f}%)")
    print(f"{'=' * 70}\n")

    time.sleep(2)

print("\n" + "=" * 70)
print(f"🏁 EXPERIMENT RUNNER FINISHED ({NUM_CLIENTS}c)  [worker {WORKER_ID}]")
print(f"Completed: {num_used_configs(NUM_CLIENTS)}/{num_total_configs(NUM_CLIENTS)}")
print("=" * 70)
