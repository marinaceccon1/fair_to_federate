import argparse
import json
import subprocess
import time
import sys
import signal
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root anchored on __file__
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR  = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_manager_fitz import (
    sample_unused_config,
    mark_config_used,
    release_reserved_config,
    mark_config_failed,
    num_total_configs,
    num_used_configs,
)

# =============================================================================
# SETTINGS
# =============================================================================
NUM_CLIENTS  = 2
BASE_PORT    = 8080     # worker 0 → 8080, worker 1 → 8081, …
RANDOM_ORDER = True     # set False for sequential iteration

# Seed for reproducible val/test splits (identical across all workers/clients)
DATA_SEED = 42

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(description="Fitzpatrick FL experiment runner")
parser.add_argument(
    "--worker-id", type=int, default=0,
    help="Unique integer ID for this runner process (0, 1, 2, …). "
         "Controls the gRPC port and per-worker config file so multiple "
         "terminals can run in parallel without collision."
)
args = parser.parse_args()

WORKER_ID   = args.worker_id
GRPC_PORT   = BASE_PORT + WORKER_ID
CONFIG_PATH = _REPO_ROOT / "experiments_fitz" / f"current_config_worker{WORKER_ID}.json"

# Environment variables forwarded to every subprocess spawned by this worker
WORKER_ENV = {
    **os.environ,
    "GRPC_PORT":   str(GRPC_PORT),
    "CONFIG_PATH": str(CONFIG_PATH),
}

print("\n" + "=" * 70)
print("🔬 FITZPATRICK EXPERIMENT RUNNER")
print("=" * 70)
print(f"Worker ID     : {WORKER_ID}")
print(f"gRPC port     : {GRPC_PORT}")
print(f"Config file   : {CONFIG_PATH}")
print(f"Sampling mode : {'RANDOM' if RANDOM_ORDER else 'SEQUENTIAL'}")
print(f"Num clients   : {NUM_CLIENTS}")
print(f"Data seed     : {DATA_SEED}")
print(f"Total configs : {num_total_configs(NUM_CLIENTS)}")
print(f"Completed     : {num_used_configs()}")
print(f"Remaining     : {num_total_configs(NUM_CLIENTS) - num_used_configs()}")
print("=" * 70 + "\n")

experiment_count = 0


# =============================================================================
# HELPERS
# =============================================================================
def cleanup_processes(server, clients):
    """Forcefully terminate all processes belonging to this worker."""
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

    # ── Sample + reserve next config (atomic, lock-protected) ─────────────────
    try:
        config = sample_unused_config(NUM_CLIENTS, random_order=RANDOM_ORDER)
    except RuntimeError:
        print("\n✅ ALL CONFIGURATIONS COMPLETED!")
        break

    print(f"\n{'=' * 70}")
    print(f"🧪 EXPERIMENT {experiment_count}  [worker {WORKER_ID}]")
    print(f"{'=' * 70}")
    for i, c in enumerate(config):
        print(f"  Client {i}: portion={c['portion']:.2f}, "
              f"composition={c['composition']}, flip={c['flip_frac']}")
    print("=" * 70 + "\n")

    # ── Write per-worker config file ──────────────────────────────────────────
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📝 Config written to {CONFIG_PATH}")

    server  = None
    clients = []
    success = False

    try:
        # ── Start server (cuda:0) ─────────────────────────────────────────────
        print(f"🖥️  Starting server on port {GRPC_PORT} (CUDA_VISIBLE_DEVICES=0)...")
        print("─" * 70)
        server = subprocess.Popen(
            [sys.executable, str(_THIS_DIR / "server_fitz.py")],
            env={**WORKER_ENV,
                 "CUDA_VISIBLE_DEVICES": "0",
                 "MODELS_DIR": str(_REPO_ROOT / "best_models_fitz")},
            cwd=str(_REPO_ROOT),
        )

        print("⏳ Waiting 15s for server initialisation...")
        time.sleep(15)

        if server.poll() is not None:
            print(f"❌ Server died during initialisation "
                  f"(exit code: {server.returncode})")
            raise RuntimeError("Server failed to start")

        # ── Start clients (cuda:1, cuda:2, …) ────────────────────────────────
        # Client i gets GPU i+1 so it never shares a device with the server.
        print(f"👥 Starting {NUM_CLIENTS} clients...")
        print("─" * 70)
        for i in range(NUM_CLIENTS):
            gpu_id      = i + 1          # client 0 → cuda:1, client 1 → cuda:2
            client_seed = DATA_SEED + i  # distinct seed per client
            print(f"  Client {i} → CUDA_VISIBLE_DEVICES={gpu_id}, seed={client_seed}")
            client = subprocess.Popen(
                [sys.executable, str(_REPO_ROOT / "src" / "client_fitz.py"),
                 "--cid",  str(i),
                 "--seed", str(DATA_SEED)],
                env={**WORKER_ENV, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
                cwd=str(_REPO_ROOT),
            )
            clients.append(client)
            time.sleep(2)   # stagger starts

        print("\n⏳ Training in progress...\n")
        print("─" * 70)
        sys.stdout.flush()

        # ── Wait for server ───────────────────────────────────────────────────
        print("⏳ Waiting for server to finish...")
        sys.stdout.flush()
        try:
            server_return = server.wait(timeout=3600)   # 1-hour max
        except subprocess.TimeoutExpired:
            print("⚠️  Server exceeded 1-hour timeout, killing it")
            server.kill()
            server_return = server.wait()

        print(f"Server exited with code {server_return}")
        sys.stdout.flush()
        time.sleep(1)

        # ── Kill clients ──────────────────────────────────────────────────────
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
        # Release so another worker (or a future restart) can retry
        release_reserved_config(config)
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
        mark_config_used(config)          # confirmation log (key already reserved)
        print("📊 Configuration confirmed as completed\n")
    else:
        print(f"\n❌ Training failed")
        mark_config_failed(config)        # log it for debugging
        release_reserved_config(config)   # put it back in the pool for retry
        print("🔁 Config released — will be retried by this or another worker\n")

    print(f"\n{'=' * 70}")
    print(f"[Worker {WORKER_ID}] Progress: "
          f"{num_used_configs()}/{num_total_configs(NUM_CLIENTS)} "
          f"({100 * num_used_configs() / num_total_configs(NUM_CLIENTS):.1f}%)")
    print(f"{'=' * 70}\n")

    time.sleep(2)

print("\n" + "=" * 70)
print(f"🏁 EXPERIMENT RUNNER FINISHED  [worker {WORKER_ID}]")
print(f"Completed: {num_used_configs()}/{num_total_configs(NUM_CLIENTS)}")
print("=" * 70)