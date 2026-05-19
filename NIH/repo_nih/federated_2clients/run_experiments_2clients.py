"""
run_experiments_2clients.py
============================
Experiment runner for the 2-client federated learning setup.

Iterates over all 729 configurations in the full Cartesian-product grid
and runs each one exactly once.  Unlike the N-client cumulative runner,
no sampling is needed — the complete enumeration is tractable.

Supports parallel workers: each worker picks a different config atomically
via file-locked reservation (experiment_manager_2clients.py).

Environment variables
---------------------
WORKER_ID       Worker index for parallel runs (default: 0).
                Each worker uses a unique gRPC port (8080 + WORKER_ID)
                and a unique config file.
GPU_DEVICE      CUDA device string passed to clients (default: "cuda").

Usage
-----
Single worker:
    python federated_2clients/run_experiments_2clients.py

Multiple parallel workers (each in its own terminal):
    WORKER_ID=0 python federated_2clients/run_experiments_2clients.py
    WORKER_ID=1 python federated_2clients/run_experiments_2clients.py
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_2CLIENTS_DIR = Path(__file__).resolve().parent          # federated_2clients/
_REPO_ROOT    = _2CLIENTS_DIR.parent.parent              # NIH/repo/ → NIH/

if str(_REPO_ROOT)    not in sys.path: sys.path.insert(0, str(_REPO_ROOT))
if str(_2CLIENTS_DIR) not in sys.path: sys.path.insert(0, str(_2CLIENTS_DIR))

from experiment_manager_2clients import (
    sample_unused_config,
    mark_config_used,
    mark_config_failed,
    release_reservation,
    get_progress_summary,
    num_total_configs,
    num_used_configs,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_CLIENTS = 2

WORKER_ID  = int(os.environ.get("WORKER_ID", "0"))
BASE_PORT  = 8080
GRPC_PORT  = BASE_PORT + WORKER_ID
CONFIG_PATH = (
    _REPO_ROOT / "repo_nih" / "experiments"
    / f"current_config_2clients_worker{WORKER_ID}.json"
)

MAX_CONSECUTIVE_FAILURES = 5

# =============================================================================
# STARTUP BANNER
# =============================================================================
print("\n" + "=" * 70)
print("🔬 2-CLIENT FEDERATED LEARNING EXPERIMENT RUNNER")
print("=" * 70)
print(f"Worker ID   : {WORKER_ID}")
print(f"gRPC port   : {GRPC_PORT}")
print(f"Config path : {CONFIG_PATH}")
print(f"GPU device  : {os.environ.get('GPU_DEVICE', 'cuda')}")

summary = get_progress_summary()
print(f"\nConfiguration space:")
print(f"  Total    : {summary['total']}")
print(f"  Done     : {summary['completed']}")
print(f"  Reserved : {summary['reserved']}")
print(f"  Remaining: {summary['remaining']}")
print(f"  Progress : {summary['percent_complete']:.1f}%")
print("=" * 70 + "\n")

experiment_count     = 0
consecutive_failures = 0


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
# MAIN EXPERIMENT LOOP
# =============================================================================
while True:
    experiment_count += 1

    # ── Sample next config ───────────────────────────────────────────────────
    try:
        config = sample_unused_config()
    except RuntimeError as e:
        print(f"\n{e}")
        print("\n✅ ALL CONFIGURATIONS COMPLETED!")
        break

    # ── Display ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"🧪 WORKER {WORKER_ID} — EXPERIMENT {experiment_count}")
    print(f"{'='*70}")
    for i, c in enumerate(config):
        print(
            f"  Client {i}: portion={c['portion']:.2f}, "
            f"gender={c['gender']}, flip={c['flip_frac']}"
        )
    print("=" * 70 + "\n")

    # ── Write config file ────────────────────────────────────────────────────
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📄 Config written to {CONFIG_PATH}")

    worker_env = os.environ.copy()
    worker_env["GRPC_PORT"]   = str(GRPC_PORT)
    worker_env["CONFIG_PATH"] = str(CONFIG_PATH)
    worker_env["NUM_CLIENTS"] = str(NUM_CLIENTS)
    worker_env["WORKER_ID"]   = str(WORKER_ID)

    server  = None
    clients = []

    try:
        # ── Server ───────────────────────────────────────────────────────────
        print("🖥️  Starting server...")
        print("─" * 70)
        server = subprocess.Popen(
            [sys.executable, str(_2CLIENTS_DIR / "server_nih_2clients.py")],
            env=worker_env,
            cwd=str(_2CLIENTS_DIR),
        )

        print("⏳ Waiting 15 s for server initialisation...")
        time.sleep(15)

        if server.poll() is not None:
            print(f"❌ Server died during init (code: {server.returncode})")
            print("⚠️  Config NOT marked as used — releasing reservation")
            release_reservation(config)
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n🛑 {MAX_CONSECUTIVE_FAILURES} consecutive failures — stopping.")
                break
            print(f"↩️  Continuing (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})...")
            time.sleep(2)
            continue

        # ── Clients ──────────────────────────────────────────────────────────
        print(f"👥 Starting {NUM_CLIENTS} clients...")
        print("─" * 70)
        for i in range(NUM_CLIENTS):
            client = subprocess.Popen(
                [sys.executable, str(_2CLIENTS_DIR / "client_nih_2clients.py"),
                 "--cid", str(i)],
                env=worker_env,
                cwd=str(_2CLIENTS_DIR),
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
            mark_config_used(config)
            print("📊 Configuration marked as completed\n")
            consecutive_failures = 0
        else:
            print(f"\n❌ Training failed (return code {server_return})")
            print("⚠️  Config NOT marked as used — releasing reservation")
            mark_config_failed(config)
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n🛑 {MAX_CONSECUTIVE_FAILURES} consecutive failures — stopping.")
                break
            print(f"↩️  Continuing (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})...")

    except KeyboardInterrupt:
        print("\n\n⚠️  Keyboard interrupt!")
        cleanup_processes(server, clients)
        release_reservation(config)
        print("🛑 Stopping experiment runner")
        break

    except Exception as exc:
        print(f"\n❌ Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
        cleanup_processes(server, clients)
        release_reservation(config)
        consecutive_failures += 1
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"\n🛑 {MAX_CONSECUTIVE_FAILURES} consecutive failures — stopping.")
            break
        print(f"↩️  Continuing (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})...")

    # ── Progress ─────────────────────────────────────────────────────────────
    summary = get_progress_summary()
    print(f"\n{'='*70}")
    print(
        f"[Worker {WORKER_ID}] Progress: "
        f"{summary['completed']}/{summary['total']} "
        f"({summary['percent_complete']:.1f}%)"
    )
    print(f"{'='*70}\n")
    time.sleep(2)


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print(f"🏁 WORKER {WORKER_ID} FINISHED")
summary = get_progress_summary()
print(
    f"Completed: {summary['completed']}/{summary['total']} "
    f"({summary['percent_complete']:.1f}%)"
)
print("=" * 70)
