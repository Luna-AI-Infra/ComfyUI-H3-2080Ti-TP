"""Locate optional native kernels without depending on a developer's filesystem."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def library_path(name, override=None):
    value = os.environ.get(override) if override else None
    path = Path(value).expanduser() if value else ROOT / 'kernels' / 'build' / f'lib{name}.so'
    if not path.is_file():
        raise OSError(
            f'H3 native kernel {name!r} is not built. Run scripts/build_kernels.py '
            f'--only {name} with the ComfyUI Python environment; '
            f'{override or "the library override"} may select an existing compatible build.'
        )
    return str(path)
