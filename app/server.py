from __future__ import annotations

import base64
import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.lenet_infer import MnistDigitsLenet86


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):
        static_dir = Path(__file__).resolve().parent / "frontend"
        super().__init__(*args, directory=str(static_dir), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            self.send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Expected application/json")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(HTTPStatus.LENGTH_REQUIRED)
            return

        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            data_url = payload["data_url"]
            if not isinstance(data_url, str) or "," not in data_url:
                raise ValueError("Invalid data_url")
            b64 = data_url.split(",", 1)[1]
            img_bytes = base64.b64decode(b64, validate=True)
            img = Image.open(BytesIO(img_bytes)).convert("L")
        except Exception as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, f"Bad request: {exc}")
            return

        try:
            result = self.server.model.predict(img)  # type: ignore[attr-defined]
        except Exception as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Inference failed: {exc}")
            return

        out = json.dumps(result).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    weights_env = os.environ.get("LENET_WEIGHTS")
    if weights_env:
        weights_path = Path(weights_env)
    else:
        weights_path = repo_root / "weights.bin"
        if not weights_path.exists():
            legacy = repo_root / "weights-86.bin"
            if legacy.exists():
                weights_path = legacy
                print(f"Using legacy weights file: {weights_path} (set LENET_WEIGHTS to override)")
            else:
                raise SystemExit(f"Missing weights file: {weights_path} (legacy: {legacy})")

    host = os.environ.get("LENET_DEMO_HOST", "127.0.0.1")
    port = int(os.environ.get("LENET_DEMO_PORT", "8000"))

    httpd = ThreadingHTTPServer((host, port), DemoHandler)
    httpd.model = MnistDigitsLenet86(weights_path)  # type: ignore[attr-defined]

    print(f"LeNet demo running at http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


__all__ = ["main"]
