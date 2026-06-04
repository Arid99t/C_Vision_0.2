"""API smoke tests using FastAPI's TestClient. Run: python tests/test_api.py"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402
from backend.app import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health").json()
    assert r["status"] == "ok"
    assert "recognizer_available" in r


def test_react_steam_reforming():
    r = client.post("/react", json={"reactants": ["CH4", "H2O"]}).json()
    assert r["ok"] is True
    assert r["products"] == {"CO": 1, "H2": 3}
    assert "→" in r["equation"]


def test_describe_zero_oh():
    r = client.post("/describe", json={"formula": "H20"}).json()
    assert r["formula"] == "H2O"
    assert r["name"] == "Water"


def test_balance():
    r = client.post("/balance", json={"reactants": ["Fe", "O2"],
                                      "products": ["Fe2O3"]}).json()
    assert r["ok"] is True
    assert r["reactants"] == {"Fe": 4, "O2": 3}
    assert r["products"] == {"Fe2O3": 2}


def test_react_unknown():
    r = client.post("/react", json={"reactants": ["He", "Ne"]}).json()
    assert r["ok"] is False


def test_recognize_unavailable_or_bad():
    # without a trained model -> 503; if a model exists, a junk image -> 400
    code = client.post("/recognize",
                       files={"image": ("x.png", b"x", "image/png")}).status_code
    assert code in (400, 503)


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in funcs:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {exc!r}")
        else:
            passed += 1
            print(f"ok   {fn.__name__}")
    print(f"\n{passed}/{len(funcs)} passed")
    sys.exit(0 if passed == len(funcs) else 1)
