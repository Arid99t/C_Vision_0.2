"""Model installation must reject damage and recover a corrupted speech vocabulary."""
import hashlib
import io
import json

from tools import fetch_assist_models as fetcher


def digest(data):
    return hashlib.sha256(data).hexdigest()


def test_failed_download_preserves_installed_model(tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher, "DEST", str(tmp_path))
    target = tmp_path / "model.onnx"
    target.write_bytes(b"existing version")
    requests = []

    def download(request, timeout):
        requests.append(request.full_url)
        return io.BytesIO(b"corrupted download")

    monkeypatch.setattr(fetcher.urllib.request, "urlopen", download)
    repo = "onnx-community/moonshine-tiny-ONNX"
    assert not fetcher.fetch("model.onnx", repo, "onnx/model.onnx", digest(b"wanted model"))
    assert target.read_bytes() == b"existing version"
    assert not (tmp_path / "model.onnx.part").exists()
    assert f"/resolve/{fetcher.REVISIONS[repo]}/" in requests[0]


def test_corrupted_cached_vocabulary_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher, "DEST", str(tmp_path))
    tokenizer = json.dumps({"model": {"vocab": {"hello": 0}}, "added_tokens": [
        {"id": 1, "special": True}, {"id": 2, "special": True}]}).encode()
    wanted = {"tokens": ["hello", "", ""], "special": [1, 2], "bos": 1, "eos": 2,
              "sample_rate": 16000, "tokens_per_second": 6.5, "layers": 6}
    wanted_bytes = json.dumps(wanted, ensure_ascii=False, separators=(",", ":")).encode()
    monkeypatch.setattr(fetcher, "FILES", {"asr_tokenizer.json": (
        "onnx-community/moonshine-tiny-ONNX", "tokenizer.json", digest(tokenizer), "speech")})
    monkeypatch.setattr(fetcher, "VOCAB_SHA256", digest(wanted_bytes))
    target = tmp_path / "asr_vocab.json"
    target.write_text('{"tokens": ["wrong"]}', encoding="utf-8")
    downloads = []

    def download(request, timeout):
        downloads.append(request.full_url)
        return io.BytesIO(tokenizer)

    monkeypatch.setattr(fetcher.urllib.request, "urlopen", download)
    assert fetcher.main([]) == 0
    assert json.loads(target.read_text(encoding="utf-8")) == wanted
    assert not (tmp_path / "asr_tokenizer.json").exists()
    assert len(downloads) == 1
    # The verified compact vocabulary makes subsequent installs work offline.
    assert fetcher.main([]) == 0
    assert len(downloads) == 1
