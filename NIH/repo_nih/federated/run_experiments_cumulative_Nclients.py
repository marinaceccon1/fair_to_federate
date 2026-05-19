"""
run_experiments_cumulative_Nclients.py
=======================================
Experiment runner for the N-client cumulative federated learning setup.

Environment variables
---------------------
NUM_CLIENTS     Number of federated clients (default: 7).
WORKER_ID       Worker index for parallel runs (default: 0).
                Each worker uses a unique gRPC port and config file.
GPU_DEVICE      CUDA device string passed to clients (default: "cuda").
MAX_SAMPLES     Experiment budget — stop after this many completed runs
                (default: 200). Can also be set in the script constants below.

Usage
-----
Single worker:
    export NUM_CLIENTS=5
    python run_experiments_cumulative_Nclients.py

Multiple parallel workers (each in its own terminal):
    NUM_CLIENTS=7 WORKER_ID=0 python run_experiments_cumulative_Nclients.py
    NUM_CLIENTS=7 WORKER_ID=1 python run_experiments_cumulative_Nclients.py
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root: one level up from this file (repo/federated/run_experiments…)
# Add both repo root (for src.*) and federated/ (for sibling bare imports).
# ---------------------------------------------------------------------------
_FEDERATED_DIR = Path(__file__).resolve().parent
_REPO_ROOT     = _FEDERATED_DIR.parent.parent
import sys as _sys
if str(_REPO_ROOT)     not in _sys.path: _sys.path.insert(0, str(_REPO_ROOT))
if str(_FEDERATED_DIR) not in _sys.path: _sys.path.insert(0, str(_FEDERATED_DIR))

from experiment_manager_cumulative_Nclients import (
    sample_unused_config,
    expand_config_to_individual_clients,
    mark_config_used,
    get_progress_summary,
    NUM_CLIENTS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
MAX_SAMPLES  = int(os.environ.get("MAX_SAMPLES", "200"))
EXPERIMENT_SEED = 42

WORKER_ID = int(os.environ.get("WORKER_ID", "0"))
BASE_PORT  = 8080
GRPC_PORT  = BASE_PORT + WORKER_ID
CONFIG_PATH = Path(
    f"experiments/current_config_{NUM_CLIENTS}clients_worker{WORKER_ID}.json"
)

MAX_CONSECUTIVE_FAILURES = 5

# =============================================================================
# STARTUP BANNER
# =============================================================================
print("\n" + "=" * 70)
print(f"🔬 {NUM_CLIENTS}-CLIENT CUMULATIVE FEDERATED LEARNING EXPERIMENT RUNNER")
print("=" * 70)
print(f"Worker ID        : {WORKER_ID}")
print(f"gRPC port        : {GRPC_PORT}")
print(f"Config path      : {CONFIG_PATH}")
print(f"Num clients      : {NUM_CLIENTS}")
print(f"GPU device       : {os.environ.get('GPU_DEVICE', 'cuda')}")
print(f"Sampling strategy: STRATIFIED UNIFORM (client 0 strata)")

summary = get_progress_summary(NUM_CLIENTS, MAX_SAMPLES)
print(f"\nConfiguration space:")
print(f"  Total possible   : {summary['total']:,}")
print(f"  Target to train  : {MAX_SAMPLES:,}  ({MAX_SAMPLES / max(summary['total'], 1) * 100:.1f}%)")
print(f"  Already completed: {summary['completed']:,}")
print(f"  Remaining        : {summary['remaining']:,}")
print(f"  Progress         : {summary['percent_complete']:.1f}%")
print("=" * 70 + "\n")

experiment_count     = 0
consecutive_failures = 0


# =============================================================================
# PROCESS CLEANUP
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
# MAIN EXPERIMENT LOOP
# =============================================================================
while True:
    experiment_count += 1

    # ── Sample next config ───────────────────────────────────────────────────
    try:
        cumulative_config = sample_unused_config(
            NUM_CLIENTS,
            max_samples=MAX_SAMPLES,
            seed=EXPERIMENT_SEED + experiment_count + WORKER_ID * 10_000,
        )
    except RuntimeError as e:
        print(f"\n{e}")
        print("\n✅ EXPERIMENT RUNNER FINISHED!")
        break

    # ── Expand cumulative config → one dict per client ───────────────────────
    config_seed = EXPERIMENT_SEED + experiment_count + WORKER_ID * 10_000
    full_config = expand_config_to_individual_clients(
        cumulative_config, seed=config_seed
    )

    # ── Display ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"🧪 WORKER {WORKER_ID} — EXPERIMENT {experiment_count}")
    print(f"{'='*70}")
    print("Client 0 (explicit):")
    print(
        f"  portion={cumulative_config['client_0']['portion']:.2f}, "
        f"gender={cumulative_config['client_0']['gender']}, "
        f"flip={cumulative_config['client_0']['flip_frac']}"
    )
    print("\nOther clients (cumulative):")
    print(f"  portion={cumulative_config['cumulative_metrics']['portion']:.4f}")
    print(f"  male   ={cumulative_config['cumulative_metrics']['male']:.4f}")
    print(f"  flip   ={cumulative_config['cumulative_metrics']['flip']:.4f}")
    print("\nExpanded to individual clients:")
    for i, c in enumerate(full_config):
        print(
            f"  Client {i}: portion={c['portion']:.2f}, "
            f"gender={c['gender']}, flip={c['flip_frac']}"
        )
    print("=" * 70 + "\n")

    # ── Write per-worker config file ─────────────────────────────────────────
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(full_config, f, indent=2)
    print(f"📄 Config written to {CONFIG_PATH}")

    worker_env = os.environ.copy()
    worker_env["GRPC_PORT"]      = str(GRPC_PORT)
    worker_env["CONFIG_PATH"]    = str(CONFIG_PATH)
    worker_env["NUM_CLIENTS"]    = str(NUM_CLIENTS)
    worker_env["EXPERIMENT_DIR"] = "experiments"

    server  = None
    clients = []

    try:
        # ── Server ───────────────────────────────────────────────────────────
        print("🖥️  Starting server...")
        print("─" * 70)
        server = subprocess.Popen(
            [sys.executable, str(_FEDERATED_DIR / "server_nih_Nclients.py")],
            env=worker_env,
            cwd=str(_FEDERATED_DIR),
        )

        print("⏳ Waiting 15 s for server initialisation...")
        time.sleep(15)

        if server.poll() is not None:
            print(f"❌ Server died during init (code: {server.returncode})")
            print("⚠️  Config NOT marked as used — will be retried")
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n🛑 {MAX_CONSECUTIVE_FAILURES} consecutive failures — stopping.")
                break
            print(f"↩️  Continuing (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})...")
            time.sleep(2)
            continue

        # ── Clients ──────────────────────────────────────────────────────────
        # Stagger client starts to reduce simultaneous GPU lock contention
        print(f"👥 Starting {NUM_CLIENTS} clients (staggered start)...")
        print("─" * 70)
        for i in range(NUM_CLIENTS):
            client = subprocess.Popen(
                [sys.executable, str(_FEDERATED_DIR / "client_nih_Nclients.py"), "--cid", str(i)],
                env=worker_env,
                cwd=str(_FEDERATED_DIR),
            )
            clients.append(client)
            time.sleep(3)

        print("\n⏳ Training in progress...\n")
        print("─" * 70)
        sys.stdout.flush()

        # ── Wait for server ──────────────────────────────────────────────────
        print("⏳ Waiting for server to finish (timeout 2 h)...")
        sys.stdout.flush()
        try:
            server_return = server.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            print("⚠️  Server exceeded 2-hour timeout — killing")
            server.kill()
            server_return = server.wait()

        print(f"✅ Server exited with code {server_return}")
        sys.stdout.flush()

        time.sleep(1)

        # ── Stop clients ─────────────────────────────────────────────────────
        print("\n🛑 Server finished — stopping clients...")
        sys.stdout.flush()
        for i, c in enumerate(clients):
            if c.poll() is None:
                try:
                    os.kill(c.pid, signal.SIGKILL)
                    print(f"  🔪 Killed client {i} (PID {c.pid})")
                except ProcessLookupError:
                    print(f"  ℹ️  Client {i} already stopped")
                except Exception as exc:
                    print(f"  ⚠️  Error killing client {i}: {exc}")

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

        # ── Outcome ──────────────────────────────────────────────────────────
        if server_return == 0:
            print("\n✅ Training completed successfully!")
            mark_config_used(cumulative_config)
            print("📊 Configuration marked as completed\n")
            consecutive_failures = 0
        else:
            print(f"\n❌ Training failed (return code {server_return})")
            print("⚠️  Config NOT marked as used — will be retried")
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n🛑 {MAX_CONSECUTIVE_FAILURES} consecutive failures — stopping.")
                break
            print(f"↩️  Continuing (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})...")

    except KeyboardInterrupt:
        print("\n\n⚠️  Keyboard interrupt!")
        cleanup_processes(server, clients)
        print("🛑 Stopping experiment runner")
        break

    except Exception as exc:
        print(f"\n❌ Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
        cleanup_processes(server, clients)
        consecutive_failures += 1
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"\n🛑 {MAX_CONSECUTIVE_FAILURES} consecutive failures — stopping.")
            break
        print(f"↩️  Continuing (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})...")

    # ── Progress ─────────────────────────────────────────────────────────────
    summary = get_progress_summary(NUM_CLIENTS, MAX_SAMPLES)
    print(f"\n{'='*70}")
    print(
        f"[Worker {WORKER_ID}] Progress: "
        f"{summary['completed']}/{MAX_SAMPLES} "
        f"({summary['percent_complete']:.1f}%)"
    )
    print(f"{'='*70}\n")
    time.sleep(2)


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print(f"🏁 WORKER {WORKER_ID} FINISHED")
summary = get_progress_summary(NUM_CLIENTS, MAX_SAMPLES)
print(
    f"Completed: {summary['completed']}/{MAX_SAMPLES} "
    f"({summary['percent_complete']:.1f}%)"
)
print("=" * 70)
