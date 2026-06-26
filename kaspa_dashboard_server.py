"""Local server for the KASPA Random Forest dashboard update workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from update_kaspa_data import DEFAULT_OUTPUT as DATA_CSV
from update_kaspa_data import get_data_status, run_update


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DASHBOARD = OUTPUTS / "kaspa_rf_dashboard.html"
METRICS = OUTPUTS / "kaspa_rf_metrics.json"
UPDATE_LOCK = threading.Lock()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    return {
        "command": args,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def pipeline_status() -> dict[str, Any]:
    status = get_data_status()
    metrics = read_json(METRICS)
    status.update(
        {
            "dashboard_file": str(DASHBOARD),
            "dashboard_exists": DASHBOARD.exists(),
            "metrics_exists": METRICS.exists(),
            "served_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    if metrics:
        status["model_data_end"] = metrics.get("data_end")
        status["backtest_end"] = metrics.get("backtest_end")
        status["latest_signal"] = metrics.get("latest_signal")
    return status


def run_pipeline() -> dict[str, Any]:
    if not UPDATE_LOCK.acquire(blocking=False):
        return {
            "ok": False,
            "message": "Update already running.",
            "status": pipeline_status(),
        }
    try:
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        data_status = run_update()
        model = run_command(
            [
                sys.executable,
                str(ROOT / "random_forest_kaspa.py"),
                "--input",
                str(DATA_CSV),
            ]
        )
        if model["returncode"] != 0:
            return {
                "ok": False,
                "message": "Data update finished, but model retrain failed.",
                "started_at_utc": started,
                "data_status": data_status,
                "model": model,
                "status": pipeline_status(),
            }

        dashboard = run_command([sys.executable, str(ROOT / "build_dashboard.py")])
        ok = dashboard["returncode"] == 0
        return {
            "ok": ok,
            "message": "Dashboard updated." if ok else "Model retrained, but dashboard rebuild failed.",
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_status": data_status,
            "model": model,
            "dashboard": dashboard,
            "status": pipeline_status(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "message": str(exc),
            "status": pipeline_status(),
        }
    finally:
        UPDATE_LOCK.release()


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(OUTPUTS), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in {"/", "/dashboard"}:
            self.path = "/kaspa_rf_dashboard.html"
            return super().do_GET()
        if self.path == "/api/status":
            return self.write_json(pipeline_status())
        return super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/api/update":
            return self.write_json(run_pipeline())
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def write_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Serving KASPA dashboard at http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
