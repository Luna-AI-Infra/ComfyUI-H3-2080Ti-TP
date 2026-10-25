"""Portable opt-in launcher. RAM model/report paths resolve relative to ComfyUI."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comfyui', required=True, type=Path)
    ap.add_argument('--config', required=True, type=Path)
    args, remaining = ap.parse_known_args()
    if remaining[:1] == ['--']:
        remaining = remaining[1:]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    for name, value in config.get('environment', {}).items():
        if not name.startswith('H3_'):
            raise ValueError('Only H3_ environment settings are accepted')
        os.environ[name] = str(value)
    root = args.comfyui.resolve()
    if not (root / 'main.py').is_file():
        raise ValueError('--comfyui must point to a ComfyUI checkout')
    os.chdir(root)
    sys.path.insert(0, str(root))
    # ComfyUI imports can parse argv during RAM-loader installation.
    sys.argv = ['main.py'] + config.get('comfyui_args', []) + remaining
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, 'allow_fp16_accumulation'):
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
    # Opt-in only: this loader was used in the RAM-resident experiments.
    ram = config.get('ram_weights')
    if ram:
        import resource
        paths = [root / p for p in ram['paths']]
        required = sum(p.stat().st_size for p in paths)
        soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if soft != resource.RLIM_INFINITY and soft < required:
            raise RuntimeError('RAM weights need a sufficient memlock limit; see docs/INSTALL.md')
        from ram_weights import preload, install
        report = Path(ram.get('report', 'h3-ram-report.json'))
        preload(paths, report)
        import comfy.options
        comfy.options.enable_args_parsing()
        install()
    runpy.run_path(str(root / 'main.py'), run_name='__main__')


if __name__ == '__main__':
    main()
