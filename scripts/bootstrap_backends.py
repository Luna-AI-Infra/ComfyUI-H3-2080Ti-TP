"""Fetch the pinned, separately attributed Sparse Kitchen backend into third_party."""
import argparse
from pathlib import Path
import subprocess
import io
import tarfile
import urllib.request

REVISION = 'c6bc3cfc58c18c3a6220aeea09116e1eebd91669'
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', action='store_true', help='Use the pinned GitHub source archive instead of Git transport')
    args = ap.parse_args()
    target = ROOT / 'third_party/H3-Optimizations'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        marker = target / '.h3-source-revision'
        if marker.is_file() and marker.read_text().strip() == REVISION:
            print('Pinned Sparse Kitchen source archive already exists')
            return
        head = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if head != REVISION:
            raise RuntimeError('Existing Sparse Kitchen checkout differs; select it explicitly with H3_SPARSE_KITCHEN_SOURCE')
        print('Pinned Sparse Kitchen checkout already exists')
        return
    if args.archive:
        url = f'https://codeload.github.com/Zironic/H3-Optimizations/tar.gz/{REVISION}'
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        staging = target.with_name(target.name + '.archive-staging')
        staging.mkdir(exist_ok=False)
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts[1:]
                if not parts:
                    continue
                if member.issym() or member.islnk() or '..' in parts or any(Path(p).is_absolute() for p in parts):
                    raise RuntimeError('Unsafe archive member rejected')
                path = staging.joinpath(*parts)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, path.open('wb') as dst:
                        dst.write(src.read())
        if not (staging / 'h3_optimizations/native/int8_attention.py').is_file():
            raise RuntimeError('Downloaded archive does not contain the expected backend')
        (staging / '.h3-source-revision').write_text(REVISION + '\n')
        staging.rename(target)
        print('Pinned Sparse Kitchen source archive ready')
        return
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/Zironic/H3-Optimizations.git', str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '--detach', REVISION], check=True)
    print('Sparse Kitchen ready; native code may build on first use. See docs/INSTALL.md.')


if __name__ == '__main__':
    main()
