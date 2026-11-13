"""Check an installation. Add --assist to require all Assist features as well."""
from pathlib import Path
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fetch_web_runtime import FILES, sha256
from tools.fetch_assist_models import FILES as ASSIST_FILES, VOCAB_SHA256


def main(argv=None):
    require_assist = "--assist" in (sys.argv[1:] if argv is None else argv)
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
    # the Assist app (/assist): each part is optional and reported separately
    assist = static / "models" / "assist"
    parts = {
        "label / chip / page reader": all((assist / n).is_file() for n in ("ocr_det.onnx", "ocr_rec.onnx")),
        "live captions": all((assist / n).is_file() for n in ("asr_encoder.onnx", "asr_decoder.onnx", "asr_vocab.json")),
        "resistor reader": all((static / "models" / n).is_file() for n in ("resistor.onnx", "resistor.json")),
        "safety, products and parts lists": all((static / "data" / n).is_file() for n in ("safety.json", "products.json", "parts.json")),
        "app page": (static / "assist.html").is_file(),
    }
    for name, present in parts.items():
        print(f"Assist {name}: " + ("installed" if present else "MISSING"))
        if require_assist and not present:
            failures.append(f"Assist {name} is missing")
    # Missing optional models are allowed for the chemistry-only app. A present
    # but corrupted model is always an error, and --assist requires all models.
    expected = {name: spec[2] for name, spec in ASSIST_FILES.items() if name != "asr_tokenizer.json"}
    expected["asr_vocab.json"] = VOCAB_SHA256
    for name, want in expected.items():
        path = assist / name
        if path.is_file() and sha256(str(path)) != want:
            failures.append(f"Damaged Assist {name}: run tools/fetch_assist_models.py")
    if not all(parts.values()):
        print("  -> python tools/fetch_assist_models.py   (text and speech models)")
        print("  -> python -m backend.resistor.export     (resistor reader, needs the trained weights)")
    for problem in failures:
        print("FAIL: " + problem)
    if failures:
        return 1
    print("Start: python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000")
    print("Open http://localhost:8000/ar (chemistry AR) or http://localhost:8000/assist (hands-free Assist app).")
    print("Use an HTTPS address for a phone or smart glasses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
