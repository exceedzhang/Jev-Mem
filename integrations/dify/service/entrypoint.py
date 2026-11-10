"""Run the Jev-Mem sidecar (8700) and the MCP server (8701) in one container.

Both services are separate processes on separate ports. The MCP client
configuration (JEV_MEM_MCP_BASE_URL / JEV_MEM_MCP_API_KEY) is filled in
from the sidecar's own settings when the two are co-located, so a single
.env file serves both; explicit JEV_MEM_MCP_* values always win.

The sidecar is the primary service: it keeps the container alive. The MCP
server is supervised alongside it — if it fails it is reported but does not
take the sidecar down, and when JEV_MEM_API_KEY is empty (sidecar auth
disabled) MCP access cannot be authenticated, so it is not started at all.

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


def start_services() -> tuple[subprocess.Popen, subprocess.Popen | None]:
    """Start the sidecar and, when it can be authenticated, the MCP server."""
    sidecar = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8700"],
        cwd=SERVICE_DIR,
    )
    if not os.environ.get("JEV_MEM_MCP_API_KEY", "").strip():
        print(
            "[entrypoint] JEV_MEM_API_KEY is empty: MCP server not started "
            "(agent access on 8701 requires an API key).",
            flush=True,
        )
        return sidecar, None
    mcp = subprocess.Popen(
        [sys.executable, os.path.join(APP_ROOT, "integrations", "mcp", "server.py")],
        cwd=APP_ROOT,
    )
    return sidecar, mcp


def _stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> int:
    load_service_env()
    # Co-located defaults: the MCP client reaches the sidecar over loopback.
    os.environ.setdefault("JEV_MEM_MCP_BASE_URL", "http://127.0.0.1:8700")
    os.environ.setdefault("JEV_MEM_MCP_API_KEY", os.environ.get("JEV_MEM_API_KEY", ""))

    sidecar, mcp = start_services()

    def _shutdown(_signum, _frame) -> None:
        for proc in (sidecar, mcp):
            if proc is not None and proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Stay up while the sidecar lives; report an MCP failure but keep serving.
    reported = False
    while sidecar.poll() is None:
        if mcp is not None and mcp.poll() is not None and not reported:
            reported = True
            print(
                f"[entrypoint] MCP server exited with code {mcp.returncode}; "
                "the Dify sidecar keeps running — check JEV_MEM_MCP_* settings.",
                flush=True,
            )
        time.sleep(0.5)

    _stop(mcp)
    _stop(sidecar)
    return 0 if sidecar.returncode in (0, -15) else 1


if __name__ == "__main__":
    sys.exit(main())