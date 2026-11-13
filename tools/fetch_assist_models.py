"""Download the pretrained models the Assist app (/assist) reuses.

    python tools/fetch_assist_models.py            # text reader + speech (about 60 MB)
    python tools/fetch_assist_models.py --no-speech   # text reader only (about 11 MB)

Two things in the Assist app are not trained in this repo, because good small
pretrained models already exist and training them needs far more data than a
laptop can handle:

  reading printed text   PP-OCRv3 English detector + recognizer (PaddleOCR,
                         Apache-2.0), ONNX conversions published by RapidOCR
  live captions          Moonshine "tiny" speech-to-text (Useful Sensors, MIT),
                         ONNX conversion published by onnx-community

Every file is pinned to a SHA-256 hash and refused if it does not match. The
files go to backend/static/models/assist/ and are served by the backend, so the
glasses/phone need no internet. They are not kept in git; run this once after
cloning. (The resistor reader IS trained here: see backend/resistor/.)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request

HF = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
# Immutable upstream revisions; the hashes below also verify every installed file.
REVISIONS = {
    "SWHL/RapidOCR": "1cfba2e90fc938db55889873735088de210cc173",
    "onnx-community/moonshine-tiny-ONNX": "a6da1241cd305dcd64eab1edbd615f2bb9aabb95",
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "backend", "static", "models", "assist")

# name on disk -> (repo, path in repo, sha256, group)
FILES = {
    "ocr_det.onnx": ("SWHL/RapidOCR", "PP-OCRv4/en_PP-OCRv3_det_infer.onnx",
                     "f139598bc2af4e4b6fe98dec11574e30edfdd91fc94ac1425c18ace3bd5a866b", "text"),
    "ocr_rec.onnx": ("SWHL/RapidOCR", "PP-OCRv3/en_PP-OCRv3_rec_infer.onnx",
                     "ef7abd8bd3629ae57ea2c28b425c1bd258a871b93fd2fe7c433946ade9b5d9ea", "text"),
    "asr_encoder.onnx": ("onnx-community/moonshine-tiny-ONNX", "onnx/encoder_model.onnx",
                         "cbbf580f703b2af2137e0f6d14cd87f31cc67bd858bfd8715403a9489982d1a5", "speech"),
    "asr_decoder.onnx": ("onnx-community/moonshine-tiny-ONNX", "onnx/decoder_model_merged_quantized.onnx",
                         "eed87831c3a6103534aae7d47a5d485025c659a1323901513961c39fe8a1a367", "speech"),
    "asr_tokenizer.json": ("onnx-community/moonshine-tiny-ONNX", "tokenizer.json",
                           "7b913404bdd039af4756783218af4440bc07fb7d6d8258d677e34f95b3ec416f", "speech"),
}
VOCAB_SHA256 = "1b6f89eebdecd5bb5d8d5c91eec21c90e12b94ea35fa3d4aee92b6a3995f912e"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, repo: str, path: str, want: str) -> bool:
    out = os.path.join(DEST, name)
    if os.path.exists(out) and sha256(out) == want:
        print(f"ok       {name}")
        return True
    print(f"fetching {name}  ({repo}) ...")
    tmp = out + ".part"
    request = urllib.request.Request(HF.format(repo=repo, revision=REVISIONS[repo], path=path),
                                     headers={"User-Agent": "assist-model-fetch"})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as f:
        for block in iter(lambda: response.read(1 << 20), b""):
            f.write(block)
    if sha256(tmp) != want:
        got = sha256(tmp)
        os.remove(tmp)
        print(f"  hash mismatch for {name}: got {got}; refusing to install it")
        return False
    os.replace(tmp, out)
    return True


def write_vocab() -> None:
    """The app only needs token id -> text to print captions, not the 3.7 MB
    tokenizer description. Keep just that list."""
    src = os.path.join(DEST, "asr_tokenizer.json")
    if sha256(src) != FILES["asr_tokenizer.json"][2]:
        raise ValueError("Tokenizer checksum mismatch; refusing to replace the vocabulary")
    with open(src, encoding="utf-8") as f:
        tok = json.load(f)
    vocab = tok["model"]["vocab"]
    size = max(vocab.values()) + 1
    for added in tok.get("added_tokens", []):
        size = max(size, added["id"] + 1)
    table = [""] * size
    for text, idx in vocab.items():
        table[idx] = text
    special = sorted(a["id"] for a in tok.get("added_tokens", []) if a.get("special"))
    out = {"tokens": table, "special": special, "bos": 1, "eos": 2, "sample_rate": 16000,
           "tokens_per_second": 6.5, "layers": 6}
    data = json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(data).hexdigest() != VOCAB_SHA256:
        raise ValueError("Derived vocabulary checksum mismatch; refusing to install it")
    target = os.path.join(DEST, "asr_vocab.json")
    with open(target + ".part", "wb") as f:
        f.write(data)
    os.replace(target + ".part", target)
    os.remove(src)


def main(argv: list[str]) -> int:
    os.makedirs(DEST, exist_ok=True)
    speech = "--no-speech" not in argv
    ok = True
    for name, (repo, path, want, group) in FILES.items():
        if group == "speech" and not speech:
            continue
        vocab = os.path.join(DEST, "asr_vocab.json")
        if name == "asr_tokenizer.json" and os.path.isfile(vocab) and sha256(vocab) == VOCAB_SHA256:
            print("ok       asr_vocab.json")
            continue
        ok = fetch(name, repo, path, want) and ok
    if ok and speech and os.path.exists(os.path.join(DEST, "asr_tokenizer.json")):
        write_vocab()
        print("wrote    asr_vocab.json")
    print(("Assist models are in " if ok else "Some files failed; see above. Target: ") + DEST)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
