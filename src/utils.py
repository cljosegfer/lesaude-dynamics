
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
