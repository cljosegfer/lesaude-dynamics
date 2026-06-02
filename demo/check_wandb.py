"""
wandb connection check
======================
Run this before any training script to verify wandb is reachable.

Usage
-----
    python demo/check_wandb.py                    # uses WANDB_PROJECT env or 'test'
    python demo/check_wandb.py --project my-proj  # explicit project
    python demo/check_wandb.py --offline          # skip actual init, just check API key + network
"""

import argparse
import os
import socket
import sys
import time


WANDB_HOST = "api.wandb.ai"
WANDB_PORT = 443
CONNECT_TIMEOUT = 5  # seconds


def check_tcp(host: str = WANDB_HOST, port: int = WANDB_PORT, timeout: float = CONNECT_TIMEOUT) -> bool:
    """Low-level TCP reachability check — no wandb import needed."""
    print(f"  TCP {host}:{port} ... ", end="", flush=True)
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        print("✓ reachable")
        return True
    except OSError as exc:
        print(f"✗ failed ({exc})")
        return False


def check_api_key() -> str | None:
    """Return the active API key (from env or netrc), or None."""
    key = os.environ.get("WANDB_API_KEY", "")
    if key:
        masked = key[:4] + "…" + key[-4:]
        print(f"  API key (env)     ... ✓ {masked}")
        return key
    # wandb falls back to ~/.netrc
    import netrc as netrc_mod
    try:
        nrc = netrc_mod.netrc()
        auth = nrc.authenticators("api.wandb.ai")
        if auth:
            key = auth[2]
            masked = key[:4] + "…" + key[-4:]
            print(f"  API key (~/.netrc) ... ✓ {masked}")
            return key
    except FileNotFoundError:
        pass
    print("  API key           ... ✗ not found (set WANDB_API_KEY or run `wandb login`)")
    return None


def check_init(project: str, timeout: float = 30.0) -> bool:
    """Try a real wandb.init and immediately finish it."""
    import wandb  # imported late so TCP/key checks print first

    print(f"  wandb.init (project={project!r}, timeout={timeout}s) ... ", end="", flush=True)
    t0 = time.monotonic()
    try:
        run = wandb.init(
            project=project,
            name="connection-check",
            tags=["check"],
            notes="Automated connectivity check — safe to delete.",
            settings=wandb.Settings(init_timeout=timeout),
            mode="online",
        )
        elapsed = time.monotonic() - t0
        print(f"✓  ({elapsed:.1f}s)  run={run.id}")
        run.finish()
        return True
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"✗  ({elapsed:.1f}s)\n    {type(exc).__name__}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify wandb connectivity before training.")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "connection-check"))
    parser.add_argument("--offline", action="store_true", help="Skip wandb.init; only check TCP + API key.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for wandb.init (default 30).")
    args = parser.parse_args()

    print("\n── wandb connection check ─────────────────────────────────────")

    ok = True

    print("\n[1/3] Network reachability")
    ok &= check_tcp()

    print("\n[2/3] Credentials")
    key = check_api_key()
    ok &= key is not None

    if args.offline:
        print("\n[3/3] wandb.init  (skipped — --offline)")
    else:
        print("\n[3/3] wandb.init")
        if not ok:
            print("  Skipped — fix errors above first.")
        else:
            ok &= check_init(args.project, timeout=args.timeout)

    print()
    if ok:
        print("✓ All checks passed — wandb is ready.\n")
        sys.exit(0)
    else:
        print("✗ One or more checks failed — see above.\n")
        print("Tips:")
        print("  • Set WANDB_API_KEY or run `wandb login`")
        print("  • Check proxy/firewall for api.wandb.ai:443")
        print("  • To train offline: set WANDB_MODE=offline or use_wandb=false")
        print("  • Increase timeout: python scripts/supervised.py ++wandb_init_timeout=120\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
