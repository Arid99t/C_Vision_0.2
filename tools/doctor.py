"""Check a local installation before opening /ar. Run with the server's Python."""
from pathlib import Path
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fetch_web_runtime import FILES, sha256


def main():
    failures = []
    for module in ("fastapi", "uvicorn", "multipart", "chempy", "numpy", "PIL", "cv2", "onnxruntime"):
        if importlib.util.find_spec(module) is None:
            failures.append(f"Missing {module}: install requirements-core.txt")
    static = ROOT / "backend" / "static"
    for relative in ("ar.html", "vendor/three.module.js", "models/formula_rec.onnx",
                     "models/formula_rec.json", "data/chem3d.json"):
        if not (static / relative).is_file():
            failures.append(f"Missing backend/static/{relative}")
    for name, expected in FILES.items():
        path = static / "vendor" / "ort" / name
        if not path.is_file() or sha256(str(path)) != expected:
            failures.append(f"Missing or damaged {name}: run tools/fetch_web_runtime.py")
    if not failures:
        try:
            from backend.recognition.infer import get_recognizer
            from backend.chemistry import react
            reader = get_recognizer()
            assert react(["CH4", "O2"])["ok"]
            bundle = json.loads((static / "data" / "chem3d.json").read_text(encoding="utf-8"))
            print(f"Ready: {reader.kind} recognizer; {len(bundle['molecules'])} structures; "
                  f"{len(bundle['reactions'])} reaction plans")
        except Exception as error:
            failures.append(f"Runtime check failed: {error}")
    objects = all((static / "models" / f"objects.{suffix}").is_file() for suffix in ("onnx", "json"))
    print("Optional object labels: " + ("installed" if objects else "not installed (formulas still work)"))
    for problem in failures:
        print("FAIL: " + problem)
    if failures:
        return 1
    print("Start: python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000")
    print("Open http://localhost:8000/ar; use an HTTPS address for a phone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
