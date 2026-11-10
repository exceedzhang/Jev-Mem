"""Run the Jev-Mem sidecar (8700) and the MCP server (8701) in one container.

Both services are separate processes on separate ports. The MCP client
configuration (JEV_MEM_MCP_BASE_URL / JEV_MEM_MCP_API_KEY) is filled in
from the sidecar's own settings when the two are co-located, so a single
.env file serves both; explicit JEV_MEM_MCP_* values always win.

`docker stop` sends SIGTERM; both processes are shut down together and the
container exits 0.
"""

import os
import signal
import subprocess
import sys
import time

SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", "..", ".."))


def load_service_env() -> None:
    """Same .env precedence as app.py: service-local first, then repo root."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(SERVICE_DIR, ".env"), override=False)
    load_dotenv(os.path.join(APP_ROOT, ".env"), override=False)


def main() -> int:
    load_service_env()
    # Co-located defaults: the MCP client reaches the sidecar over loopback.
    os.environ.setdefault("JEV_MEM_MCP_BASE_URL", "http://127.0.0.1:8700")
    os.environ.setdefault("JEV_MEM_MCP_API_KEY", os.environ.get("JEV_MEM_API_KEY", ""))

    sidecar = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8700"],
        cwd=SERVICE_DIR,
    )
    mcp = subprocess.Popen(
        [sys.executable, os.path.join(APP_ROOT, "integrations", "mcp", "server.py")],
        cwd=APP_ROOT,
    )
    children = (sidecar, mcp)

    def _shutdown(_signum, _frame) -> None:
        for proc in children:
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Stay up until one service stops, then stop the other.
    while True:
        if any(proc.poll() is not None for proc in children):
            break
        time.sleep(0.5)

    for proc in children:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    codes = (sidecar.returncode, mcp.returncode)
    return 0 if all(code in (0, -15) for code in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
