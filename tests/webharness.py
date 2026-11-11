"""Shared helpers for the browser tests: run the backend on a free port and
drive a headless Chromium (Playwright) against it.

    from tests.webharness import browser_page
    with browser_page() as (page, base):
        page.evaluate("async () => { const m = await import('/static/js/chem.js'); ... }")
"""
from __future__ import annotations

import base64
import contextlib
import os
import socket
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def skip_unavailable(reason):
    """Skip a standalone script or a pytest module without aborting collection."""
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(reason, allow_module_level=True)
    print("SKIP: " + reason)
    raise SystemExit(0)


def playwright_available() -> bool:
    try:
        import playwright.sync_api  # noqa: F401
        import uvicorn  # noqa: F401
        return True
    except ImportError:
        return False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def backend_server():
    """Serve backend.app on 127.0.0.1:<free port>; yields the base URL."""
    import uvicorn
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config("backend.app:app", host="127.0.0.1", port=port,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("backend did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@contextlib.contextmanager
def browser_page(path: str = "/static/blank.html", args: list | None = None,
                 viewport: dict | None = None, wait: str = "load"):
    """Headless Chromium page opened on the local backend. Yields (page, base_url).
    Console errors and failed requests are collected in page.problems."""
    from playwright.sync_api import sync_playwright
    with backend_server() as base, sync_playwright() as pw:
        # WebGL through SwiftShader (software): the tests then never compete with
        # a training run for the graphics card, and behave the same on any PC.
        browser = pw.chromium.launch(args=["--use-fake-ui-for-media-stream",
                                           "--use-fake-device-for-media-stream",
                                           "--use-angle=swiftshader",
                                           "--enable-unsafe-swiftshader", *(args or [])])
        context = browser.new_context(viewport=viewport or {"width": 412, "height": 915},
                                      device_scale_factor=1)
        page = context.new_page()
        page.problems = []
        page.on("pageerror", lambda e: page.problems.append(f"pageerror: {e}"))
        page.on("console", lambda m: page.problems.append(f"console.{m.type}: {m.text}")
                if m.type == "error" else None)
        page.on("requestfailed", lambda r: page.problems.append(f"requestfailed: {r.url}"))
        page.goto(base + path, wait_until=wait)
        try:
            yield page, base
        finally:
            browser.close()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def unb64(text: str) -> bytes:
    return base64.b64decode(text)


# JavaScript helpers injected into evaluate() calls
JS_B64 = """
const fromB64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const toB64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) {
  s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); } return btoa(s); };
"""
