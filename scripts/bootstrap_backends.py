"""Fetch the pinned, separately attributed Sparse Kitchen backend into third_party."""
import argparse
from pathlib import Path
import subprocess

REVISION = 'c6bc3cfc58c18c3a6220aeea09116e1eebd91669'
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    target = ROOT / 'third_party/H3-Optimizations'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        head = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if head != REVISION:
            raise RuntimeError('Existing Sparse Kitchen checkout differs; select it explicitly with H3_SPARSE_KITCHEN_SOURCE')
        print('Pinned Sparse Kitchen checkout already exists')
        return
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/Zironic/H3-Optimizations.git', str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '--detach', REVISION], check=True)
    print('Sparse Kitchen ready; native code may build on first use. See docs/INSTALL.md.')


if __name__ == '__main__':
    main()
