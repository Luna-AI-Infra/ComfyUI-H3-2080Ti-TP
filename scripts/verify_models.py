"""Verify downloaded model bytes against the tested manifest; never downloads weights."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comfyui', type=Path, required=True)
    args = ap.parse_args()
    models = json.loads((Path(__file__).resolve().parents[1] / 'configs/models.json').read_text(encoding='utf-8'))
    failed = False
    for item in models:
        path = args.comfyui / item['comfyui_path']
        if not path.exists():
            print(item['file'], 'not installed (FL2VA and Ref2VA are alternatives)')
            continue
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(16 * 1024**2), b''):
                h.update(chunk)
        ok = h.hexdigest() == item['sha256'] and path.stat().st_size == item['bytes']
        print(item['file'], 'OK' if ok else 'MISMATCH')
        failed |= not ok
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
