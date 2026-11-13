"""Loopback-only demo with native local inference or an explicitly configured TGI endpoint."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

from .backends import GenerationConfig, backend_from_config
from .evaluation import grade_once
from .io import utc_now, write_json
from .prompts import make_prompt


def serve(backend_config, port=8765, results="experiments/mac-validation"):
    backend = backend_from_config(backend_config)
    lock = threading.Lock()
    result_path = Path(results)
    demo = json.loads(Path("examples/evaluation.json").read_text())

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value, content_type="application/json"):
            data = (
                value
                if isinstance(value, bytes)
                else json.dumps(value, ensure_ascii=False).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self.reply(
                    200,
                    files("prometheus_eval").joinpath("web/index.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            elif self.path == "/api/health":
                self.reply(
                    200,
                    {
                        "ready": True,
                        "backend": backend.identity,
                        "local": backend.identity["backend"] in ("mlx", "torch"),
                    },
                )
            elif self.path == "/api/example":
                self.reply(200, demo)
            elif self.path == "/api/results":
                path = result_path / "summary.json"
                self.reply(200, json.loads(path.read_text()) if path.exists() else {})
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/evaluate":
                self.reply(404, {"error": "Not found"})
                return
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                self.reply(403, {"error": "Only this local page may submit evaluations"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 256 * 1024:
                    raise ValueError("Expected a JSON input smaller than 256 KiB")
                data = json.loads(self.rfile.read(size))
                prompt = make_prompt(
                    data["instruction"],
                    data["response"],
                    data.get("reference_answer", ""),
                    data["rubric"],
                )
            except (ValueError, KeyError, TypeError) as e:
                self.reply(400, {"error": str(e)})
                return
            if not lock.acquire(blocking=False):
                self.reply(409, {"error": "An evaluation is running; please wait"})
                return
            try:
                result = grade_once(backend, prompt, GenerationConfig(max_tokens=384), attempts=2)
                record = {
                    "input": data,
                    "backend": backend.identity,
                    "timestamp_utc": utc_now(),
                    **result,
                }
                write_json("artifacts/demo/live_result.json", record)
                self.reply(200 if result["status"] == "ok" else 422, result)
            finally:
                lock.release()

    location = (
        "local inference" if backend.identity["backend"] != "tgi" else "configured TGI endpoint"
    )
    print(f"Prometheus ready at http://127.0.0.1:{port} — {location}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
