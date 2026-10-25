"""Install checksum-pinned Linux x86_64 SM75 kernels; no compiler or administrator access."""
import argparse
import ctypes
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, help='Use an already downloaded release archive')
    ap.add_argument('--replace', action='store_true', help='Replace existing libraries with different hashes')
    args = ap.parse_args()
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        ap.error('This prebuilt is for Linux x86_64 only; build from source on other platforms')
    meta = json.loads((ROOT / 'configs/prebuilt.json').read_text(encoding='utf-8'))
    for name, expected in meta['source_sha256'].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise RuntimeError('Kernel sources differ from this prebuilt: ' + name + '; build from source')
    if args.archive:
        data = args.archive.read_bytes()
    else:
        print('Downloading pinned SM75 kernel release', flush=True)
        with urllib.request.urlopen(meta['url'], timeout=120) as response:
            data = response.read()
    if digest(data) != meta['sha256']:
        raise RuntimeError('Archive SHA256 mismatch; no libraries installed')
    out = ROOT / 'kernels/build'
    with tempfile.TemporaryDirectory(prefix='h3-sm75-') as folder:
        staging = Path(folder)
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
            members = archive.getmembers()
            if len({m.name for m in members}) != len(members):
                raise RuntimeError('Duplicate archive member')
            for member in members:
                name = Path(member.name)
                if not member.isfile() or name.is_absolute() or '..' in name.parts:
                    raise RuntimeError('Unsafe archive member')
                if name.parts[:2] != ('kernels', 'build'):
                    raise RuntimeError('Unexpected archive layout')
                dest = staging / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as f:
                    dest.write_bytes(f.read())
        for name, expected in meta['libraries'].items():
            src = staging / 'kernels/build' / name
            if digest(src.read_bytes()) != expected:
                raise RuntimeError('Library SHA256 mismatch: ' + name)
            target = out / name
            if target.exists() and digest(target.read_bytes()) != expected and not args.replace:
                raise RuntimeError('Existing library differs: ' + name + '; back it up or use --replace')
        # Check ELF dependencies before installing any files; no GPU kernels execute.
        try:
            for name in meta['libraries']:
                ctypes.CDLL(str(staging / 'kernels/build' / name))
        except OSError as e:
            raise RuntimeError('Prebuilt dependency cannot load. CUDA 12 cuBLASLt and compatible '
                               'glibc/libstdc++ are required; see docs/PREBUILT.md. ' + str(e)) from e
        for src in (staging / 'kernels/build').rglob('*'):
            if src.is_file():
                dest = out / src.relative_to(staging / 'kernels/build')
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
    print('Installed and ELF-loaded all six SM75 libraries; run doctor and GPU smoke checks separately')


if __name__ == '__main__':
    main()
