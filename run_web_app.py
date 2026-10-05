"""
run_web_app.py - ConSQL Real-Time Web Application Runner
Launches the FastAPI backend and serves the Next-Gen Text-to-SQL Studio UI.

Usage:
    python run_web_app.py
Then open:
    http://localhost:8000
"""

import sys
import os
import socket
import logging
from pathlib import Path

# Add project root to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_web_app")


def find_open_port(base_port: int = 8000, max_port: int = 8030) -> int:
    """Finds an open TCP port on localhost."""
    for port in range(base_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise OSError(f"No open port found in range {base_port}-{max_port}")


def get_local_ip() -> str:
    """Detects primary outbound local network IPv4 address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    import uvicorn
    from web_app.server import app, get_pipeline

    port = int(os.environ.get("CONSQL_PORT", "8000"))
    port = find_open_port(port)
    local_ip = get_local_ip()

    print("\n" + "=" * 65)
    print("  ConSQL Next-Gen Real-Time Text-to-SQL Studio")
    print("=" * 65)
    print(f"  Working Directory: {CURRENT_DIR}")
    print(f"  Port:              {port}")
    print(f"  This PC (Local):   http://localhost:{port}")
    print(f"  All Devices (LAN): http://{local_ip}:{port}")
    print("=" * 65 + "\n")

    # Warm up pipeline components
    print("Warming up ConSQL pipeline components...")
    get_pipeline()
    print("[OK] Pipeline loaded and ready.")

    print(f"\nStarting Uvicorn web server on http://localhost:{port} ...\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
