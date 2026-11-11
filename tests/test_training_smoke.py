"""Exercise real-crop fine-tuning and export with stand-in images, in temp files.

This checks the workflow, not real handwriting accuracy. Production weights
and training.log are never changed.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw


def test_finetune_and_export(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from backend.recognition import train, preprocess
    from backend.recognition.export import export_onnx
    from backend.recognition.config import MOBILE_WEIGHTS_PATH
    if not Path(MOBILE_WEIGHTS_PATH).is_file():
        pytest.skip("Training checkpoint is not included in a runtime-only checkout")
    images, labels = [], []
    for i in range(16):
        label = ["H2O", "CO2", "NaCl", "CH4"][i % 4]
        picture = Image.new("L", (160, 48), 255)
        ImageDraw.Draw(picture).text((15, 8), label, fill=0, font_size=28)
        picture.save(tmp_path / f"crop{i}.png")
        images.append(preprocess.prepare_crop(np.asarray(picture)))
        labels.append(label)
    base = str(tmp_path / "samples")
    Path(base + ".bin").write_bytes(b"".join(im.tobytes() for im in images))
    np.savez(base + ".npz", widths=np.array([im.shape[1] for im in images], np.uint16),
             labels="\n".join(labels), modes="f" * len(labels))
    label_file = tmp_path / "labels.tsv"
    label_file.write_text("".join(f"crop{i}.png\t{lab}\n" for i, lab in enumerate(labels)), encoding="utf-8")
    monkeypatch.setattr(train, "LOG_PATH", str(tmp_path / "train.log"))
    output = tmp_path / "fine.pt"
    args = train.build_argparser().parse_args([
        "--init", MOBILE_WEIGHTS_PATH, "--out", str(output), "--data", base,
        "--eval", base, "--epochs", "1", "--steps", "2", "--batch", "4", "--device", "cpu",
        "--real-dir", str(tmp_path), "--real-labels", str(label_file), "--real-frac", "0.5"])
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        train.train(args)
        checkpoint = torch.load(output, map_location="cpu")
        assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())
        stats = export_onnx(str(output), str(tmp_path / "fine.onnx"))
        assert stats["max_abs_diff_vs_torch"] < 1e-3
    finally:
        torch.set_num_threads(previous)
