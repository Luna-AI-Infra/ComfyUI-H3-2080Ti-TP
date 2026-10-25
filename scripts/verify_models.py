"""Verify downloaded model bytes against the tested manifest; never downloads weights."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comfyui', type=Path, required=True)
    ap.add_argument('--task', choices=('fl2va', 'ref2va'), required=True,
                    help='Require all five assets for this task, including the example Turbo LoRA')
    args = ap.parse_args()
    models = json.loads((Path(__file__).resolve().parents[1] / 'configs/models.json').read_text(encoding='utf-8'))
    failed = False
    for item in models:
        if item['role'] == 'diffusion_model' and args.task not in item['file']:
            continue
        path = args.comfyui / item['comfyui_path']
        if not path.exists():
            print(item['file'], 'MISSING')
            failed = True
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
