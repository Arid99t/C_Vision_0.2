"""Stage the allowlisted offline Assist bundle for the Android APK.

Run from any directory: python tools/prepare_assist_assets.py
Downloaded runtimes/models and the two release model exports are checked
against pinned hashes. Source JS/data can change normally; their exact staged
hashes are recorded in asset-manifest.json inside the APK. No test fixtures,
object models, Python code, training files or private photos are packaged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fetch_assist_models import FILES as ASSIST_FILES, VOCAB_SHA256
from tools.fetch_web_runtime import FILES as RUNTIME_FILES

OUTPUT = Path('clients/glasses/android/app/build/generated/assistAssets')
BASE_JS = ('chem', 'decode', 'detect', 'ortenv', 'preprocess', 'recognizer', 'track')
ASSIST_JS = ('access', 'app', 'asr.worker', 'audio-tap', 'barcode', 'electronics',
             'formulas', 'icons', 'kb', 'ocr', 'ocr.worker', 'resistor', 'safety')
STATIC_FILES = (
    *(f'js/{name}.js' for name in BASE_JS),
    *(f'js/assist/{name}.js' for name in ASSIST_JS),
    *(f'data/{name}.json' for name in ('chem3d', 'parts', 'products', 'safety')),
    'models/formula_rec.onnx', 'models/formula_rec.json',
    'models/resistor.onnx', 'models/resistor.json',
    *(f'models/assist/{name}' for name in ASSIST_FILES if name != 'asr_tokenizer.json'),
    'models/assist/asr_vocab.json',
    *(f'vendor/ort/{name}' for name in RUNTIME_FILES),
)
ASSETS = {'backend/static/assist.html': 'web/assist.html',
          **{f'backend/static/{name}': f'web/static/{name}' for name in STATIC_FILES},
          'THIRD_PARTY_NOTICES.txt': 'web/THIRD_PARTY_NOTICES.txt',
          'backend/static/vendor/ort.LICENSE.txt': 'web/licenses/ONNX-Runtime.LICENSE.txt',
          'backend/static/licenses/ASSIST_MODEL_SOURCES.txt': 'web/licenses/ASSIST_MODEL_SOURCES.txt',
          **{f'backend/static/licenses/{name}.LICENSE.txt': f'web/licenses/{name}.LICENSE.txt'
             for name in ('PaddleOCR', 'RapidOCR', 'Moonshine')}}
PINNED = {
    **{f'backend/static/vendor/ort/{name}': digest for name, digest in RUNTIME_FILES.items()},
    **{f'backend/static/models/assist/{name}': details[2]
       for name, details in ASSIST_FILES.items() if name != 'asr_tokenizer.json'},
    'backend/static/models/assist/asr_vocab.json': VOCAB_SHA256,
    # Update deliberately after a newly exported model passes its regression tests.
    'backend/static/models/formula_rec.onnx': 'c5baa65e7adb59b0d72eee30f8964d07ed90d37dc8575ef800c40ff7355f842b',
    'backend/static/models/resistor.onnx': '304e51c894890266fff85a750ff9167e8690ee4d23d74f86e62a1ce8a7d19239',
}


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare_assets(root: Path = ROOT) -> dict:
    root = root.resolve()
    output = root / OUTPUT
    if output.is_symlink() or not output.resolve().is_relative_to(root):
        raise ValueError('Generated assets directory must stay inside the checkout')
    sources = []
    # Finish validation before replacing a previously usable generated bundle.
    for name, target in ASSETS.items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f'Missing or unsafe required asset: {name}')
        if source.stat().st_size == 0:
            raise ValueError(f'Empty required asset: {name}')
        if name in PINNED and sha256(source) != PINNED[name]:
            raise ValueError(f'Checksum mismatch: {name}; restore or fetch the pinned asset before building')
        if Path(target).is_absolute() or '..' in Path(target).parts:
            raise ValueError(f'Unsafe packaged path: {target}')
        sources.append((name, source, target))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='assist-stage-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'assets'
        stage.mkdir()
        manifest = {'schemaVersion': 1, 'version': '0.2.0', 'files': {}}
        for name, source, target in sources:
            dest = stage / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            digest = sha256(dest)
            if name in PINNED and digest != PINNED[name]:
                raise ValueError(f'Asset changed while staging: {name}')
            manifest['files'][target] = {'bytes': dest.stat().st_size, 'sha256': digest}
        (stage / 'asset-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        # OUTPUT is a fixed generated directory, resolved and constrained above.
        if output.exists():
            shutil.rmtree(output)
        stage.replace(output)
    return manifest


def verify_apk(path: Path) -> dict:
    """Confirm every bundled asset matches the generated manifest and allowlist."""
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.startswith('assets/') and not name.endswith('/')]
        if len(names) != len(set(names)):
            raise ValueError('APK contains duplicate asset paths')
        expected = {'assets/' + target for target in ASSETS.values()} | {'assets/asset-manifest.json'}
        if set(names) != expected:
            raise ValueError('APK asset list differs from the Assist allowlist')
        manifest = json.loads(archive.read('assets/asset-manifest.json'))
        if set(manifest['files']) != set(ASSETS.values()):
            raise ValueError('APK manifest differs from the Assist allowlist')
        reverse = {target: source for source, target in ASSETS.items()}
        for target, record in manifest['files'].items():
            data = archive.read('assets/' + target)
            digest = hashlib.sha256(data).hexdigest()
            if digest != record['sha256'] or len(data) != record['bytes']:
                raise ValueError(f'APK asset checksum mismatch: {target}')
            source = reverse[target]
            if source in PINNED and digest != PINNED[source]:
                raise ValueError(f'APK pinned asset checksum mismatch: {target}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-apk', type=Path, help='verify an already built APK instead of staging files')
    args = parser.parse_args()
    try:
        info = verify_apk(args.verify_apk) if args.verify_apk else prepare_assets()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise SystemExit(str(error)) from error
    size = sum(entry['bytes'] for entry in info['files'].values())
    destination = args.verify_apk or ROOT / OUTPUT
    print(f"Verified {len(info['files'])} bundled files ({size / 1024 / 1024:.1f} MiB): {destination}")
