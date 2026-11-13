"""APK packaging must fail closed and include only its explicit asset list."""
import hashlib
import zipfile

import pytest

from tools import prepare_assist_assets as pack


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    assets = {'backend/static/assist.html': 'web/assist.html',
              'backend/static/models/reader.onnx': 'web/static/models/reader.onnx'}
    for source in assets:
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.encode())
    model = 'backend/static/models/reader.onnx'
    monkeypatch.setattr(pack, 'ASSETS', assets)
    monkeypatch.setattr(pack, 'PINNED', {model: hashlib.sha256(model.encode()).hexdigest()})
    return tmp_path


def test_allowlist_and_apk_integrity(checkout):
    private = checkout / 'backend/static/test/private-photo.jpg'
    private.parent.mkdir(parents=True)
    private.write_bytes(b'not for distribution')
    manifest = pack.prepare_assets(checkout)
    output = checkout / pack.OUTPUT
    assert set(manifest['files']) == set(pack.ASSETS.values())
    assert not list(output.rglob('*private*'))
    apk = checkout / 'app.apk'
    with zipfile.ZipFile(apk, 'w') as archive:
        for path in output.rglob('*'):
            if path.is_file():
                archive.write(path, 'assets/' + path.relative_to(output).as_posix())
    assert pack.verify_apk(apk) == manifest
    with zipfile.ZipFile(apk, 'a') as archive:
        archive.writestr('assets/private-photo.jpg', b'unexpected')
    with pytest.raises(ValueError, match='allowlist'):
        pack.verify_apk(apk)


def test_checksum_failure_preserves_previous_bundle(checkout):
    pack.prepare_assets(checkout)
    output = checkout / pack.OUTPUT
    previous = (output / 'asset-manifest.json').read_bytes()
    (checkout / 'backend/static/models/reader.onnx').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        pack.prepare_assets(checkout)
    assert (output / 'asset-manifest.json').read_bytes() == previous


def test_stale_generated_assets_are_removed(checkout):
    pack.prepare_assets(checkout)
    stale = checkout / pack.OUTPUT / 'web/static/old-private-model.onnx'
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b'old')
    pack.prepare_assets(checkout)
    assert not stale.exists()
