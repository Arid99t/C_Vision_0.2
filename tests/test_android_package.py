"""Verify the built APK's offline assets on the host, without an Android device.

Set CHEMVISION_ASSIST_APK to inspect another build. The browser smoke serves
only the APK zip contents and explicit test fixtures: it does not use FastAPI
or source-tree assets. This checks packaging and single-thread WASM workers,
not Android WebView, device permissions, camera hardware, or native TTS.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import threading
from urllib.parse import urlsplit
from zipfile import ZipFile

import pytest

from tools.fetch_assist_models import FILES as ASSIST_FILES, VOCAB_SHA256
from tools.fetch_web_runtime import FILES as RUNTIME_FILES

ROOT = Path(__file__).resolve().parents[1]
APK = Path(os.environ.get("CHEMVISION_ASSIST_APK", ROOT / "dist" / "C-Vision-Assist-0.2.0-debug.apk"))
PREFIX = "assets/web/"


@pytest.fixture
def package():
    if not APK.is_file():
        pytest.skip("Android APK not built; see clients/glasses/android/README.md")
    with ZipFile(APK) as archive:
        yield archive


def _sha(archive, path):
    digest = hashlib.sha256()
    with archive.open(PREFIX + path) as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_apk_contains_complete_verified_offline_assets(package):
    names = set(package.namelist())
    required = {
        "assist.html", "static/js/assist/app.js", "static/js/assist/audio-tap.js",
        "static/js/assist/asr.worker.js", "static/js/assist/ocr.worker.js",
        "static/data/safety.json", "static/data/products.json", "static/data/parts.json", "static/data/chem3d.json",
        "static/models/resistor.json", "static/models/resistor.onnx",
        "static/models/formula_rec.json", "static/models/formula_rec.onnx", "THIRD_PARTY_NOTICES.txt",
    }
    assert {PREFIX + path for path in required} <= names
    assert any(name.startswith(PREFIX + "licenses/") for name in names)
    for file, expected in RUNTIME_FILES.items():
        assert _sha(package, "static/vendor/ort/" + file) == expected, file
    for file, (_, _, expected, _) in ASSIST_FILES.items():
        if file != "asr_tokenizer.json":  # runtime uses only the derived vocabulary
            assert _sha(package, "static/models/assist/" + file) == expected, file
    assert _sha(package, "static/models/assist/asr_vocab.json") == VOCAB_SHA256

    # Follow app-owned JS imports and worker scripts, including dynamic imports.
    dependency = re.compile(r"(?:\bfrom\s*|\bimport\s*\(?\s*|\bworkerUrl\(\s*|\bnew URL\(\s*)['\"]([^'\"]+\.(?:js|mjs))['\"]")
    for name in sorted(names):
        if not name.startswith(PREFIX + "static/js/") or not name.endswith(".js"):
            continue
        source = package.read(name).decode("utf-8")
        for target in dependency.findall(source):
            if target.startswith("/"):
                wanted = PREFIX + target.lstrip("/")
            else:
                parts = list(PurePosixPath(name).parent.parts)
                for part in PurePosixPath(target).parts:
                    if part == "..":
                        parts.pop()
                    elif part != ".":
                        parts.append(part)
                wanted = "/".join(parts)
            assert wanted in names, f"Missing dependency {target} from {name}"

    assert not any("/static/test/" in name or name.endswith((".pt", ".npz", ".env")) for name in names)
    for file in ("resistor", "formula_rec"):
        meta = json.loads(package.read(PREFIX + f"static/models/{file}.json"))
        assert PREFIX + "static/models/" + meta.get("file", file + ".onnx") in names


@contextmanager
def _serve_package(archive, fixtures):
    """Trusted localhost replaces Android's intercepted HTTPS asset origin.

    No cross-origin isolation headers intentionally: exercise the single-thread
    fallback expected on WebView, where SharedArrayBuffer may be unavailable.
    """
    missing = []
    # Exercise exactly the native responder's CSP, including WASM compilation
    # and module/blob workers. Isolation headers are omitted as explained above.
    native = (ROOT / "clients/glasses/android/app/src/main/java/org/cvision/assist/MainActivity.java").read_text(encoding="utf-8")
    expression = re.search(r'headers\.put\("Content-Security-Policy",\s*(.*?)\);', native, re.S)
    assert expression, "Native response must declare its CSP"
    csp = "".join(json.loads('"' + part + '"') for part in re.findall(r'"((?:[^"\\]|\\.)*)"', expression.group(1)))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            asset = "assist.html" if path in ("/", "/assist", "/assist.html") else path.lstrip("/")
            try:
                data = fixtures[path] if path in fixtures else archive.read(PREFIX + asset)
            except KeyError:
                missing.append(path)
                self.send_error(404)
                return
            mime = {".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm",
                    ".onnx": "application/octet-stream"}.get(Path(asset).suffix)
            self.send_response(200)
            self.send_header("Content-Type", mime or mimetypes.guess_type(asset)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Security-Policy", csp)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", missing
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_packaged_models_work_offline_in_single_thread_browser(package):
    playwright = pytest.importorskip("playwright.sync_api")
    from tests import assist_scenes as scenes

    if not scenes.fonts_available():
        pytest.skip("Test scene fonts are unavailable")
    resistor = scenes.resistor_frame(["yellow", "violet", "red", "gold"], seed=3, name="apk_resistor.png")
    labels = scenes.packages(
        ((30, 90, 200), [("ULTRA", 60, True), ("BLEACH", 120, True)]),
        ((250, 250, 250), [("Glass Cleaner", 76, True), ("with AMMONIA", 64, True)], (20, 60, 160)),
        name="apk_labels.png")
    fixtures = {
        "/resistor.png": (ROOT / "backend" / resistor.lstrip("/")).read_bytes(),
        "/labels.png": (ROOT / "backend" / labels.lstrip("/")).read_bytes(),
        "/speech.wav": (ROOT / "tests/fixtures/speech_sample.wav").read_bytes(),
    }
    with _serve_package(package, fixtures) as (base, missing), playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                                           "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        context = browser.new_context(viewport={"width": 640, "height": 360}, service_workers="block")
        external, problems = [], []

        def local_only(route):
            if route.request.url.startswith(base + "/"):
                route.continue_()
            else:
                external.append(route.request.url)
                route.abort()

        context.route("**/*", local_only)
        page = context.new_page()
        page.on("pageerror", lambda error: problems.append(str(error)))
        page.on("console", lambda message: problems.append(message.text) if message.type == "error" else None)
        try:
            page.goto(base + "/assist?threads=1&mode=safety&src=/labels.png")
            page.wait_for_function("() => window.assist?.mode?.formulas && assist.state().detail?.verdict?.level === 'danger'", timeout=180000)
            assert page.evaluate("() => self.crossOriginIsolated") is False
            assert page.evaluate("async () => (await import('/static/js/ortenv.js')).threads") == 1

            page.goto(base + "/assist?threads=1&mode=electronics&sub=resistor&src=/resistor.png")
            page.wait_for_function("() => window.assist?.state().detail?.resistor?.ohms === 4700", timeout=120000)

            page.goto(base + "/assist?threads=1&mode=access&sub=captions&src=/labels.png&audio=/speech.wav")
            page.wait_for_function("() => window.assist?.state().detail?.captions?.finals?.length >= 2", timeout=180000)
            heard = page.evaluate("() => assist.state().detail.captions.finals.join(' ').toLowerCase()")
            assert all(word in heard for word in ("plants", "glucose", "carbon dioxide", "water", "sunlight")), heard
            page.evaluate("() => assist.stop()")
            assert page.evaluate("() => !assist.running && !assist.modes.get('access').worker")
            assert not external, external
            assert not missing, missing
            assert not problems, problems
        finally:
            browser.close()
