"""Download the WebAssembly runtime the phone web app uses to run its models.

    python tools/fetch_web_runtime.py

Fetches ONNX Runtime Web (MIT licence), pinned to one version and checked
against known SHA-256 hashes, into backend/static/vendor/ort/. The files are
served by the backend itself, so the phone needs no CDN and the app also works
on a LAN without internet.

They are not kept in git (the .wasm is 15 MB); run this once after cloning.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

VERSION = "1.31.1"
BASE = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VERSION}/dist/"
FILES = {
    "ort.wasm.min.mjs": "c21d9c326d453a4a61398270c804d5979e118540597cb0c07f7b3d64e6098234",
    "ort-wasm-simd-threaded.mjs": "30ed0b5cd1182ede931fea448afb567d4dc705ec957af09f108a2be5f770dac3",
    "ort-wasm-simd-threaded.wasm": "4b884c203b0fa48d057be3332c65513ff85aa3c5bddd6946acc0fab7ecda001d",
}
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "backend", "static", "vendor", "ort")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    os.makedirs(DEST, exist_ok=True)
    for name, want in FILES.items():
        path = os.path.join(DEST, name)
        if os.path.exists(path) and sha256(path) == want:
            print(f"ok       {name}")
            continue
        print(f"fetching {name} ...")
        tmp = path + ".part"
        urllib.request.urlretrieve(BASE + name, tmp)
        got = sha256(tmp)
        if got != want:
            os.remove(tmp)
            print(f"  hash mismatch for {name}: got {got}; refusing to install it")
            return 1
        os.replace(tmp, path)
    print(f"ONNX Runtime Web {VERSION} is in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
