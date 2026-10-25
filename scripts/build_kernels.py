"""Build the original SM75 kernels. No model weights or administrator access required."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    'fc2_int8': 'fc2_int8/kernel.cu',
    'swiglu': 'swiglu/kernel.cu',
    'kitchen_shared_kv': 'kitchen_shared_kv/launcher.cu',
    'kitchen_offsets': 'kitchen_offsets/launcher.cu',
    'vae_swiglu': 'vae_swiglu/kernel.cu',
    'vae_half': 'vae_half/kernel.cu',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', nargs='+', choices=list(TARGETS), default=list(TARGETS))
    ap.add_argument('--nvcc', default=os.environ.get('NVCC', 'nvcc'))
    ap.add_argument('--release', action='store_true',
                    help='Omit profiling line information and compile with relative source paths for redistribution')
    ap.add_argument('--cutlass-include', type=Path, default=os.environ.get('CUTLASS_INCLUDE_DIR'))
    args = ap.parse_args()
    nvcc = shutil.which(args.nvcc)
    if not nvcc:
        ap.error('nvcc not found; install a CUDA toolkit and select it with --nvcc or NVCC')
    if 'fc2_int8' in args.only:
        if not args.cutlass_include or not (args.cutlass_include / 'cutlass/cutlass.h').is_file():
            ap.error('fc2_int8 needs --cutlass-include PATH_TO_CUTLASS/include; see docs/INSTALL.md')
    out = ROOT / 'kernels/build'
    out.mkdir(exist_ok=True)
    evidence = []
    for name in args.only:
        source = ROOT / 'kernels' / TARGETS[name]
        target = out / f'lib{name}.so'
        cmd = [nvcc, '-arch=sm_75', '-O3', '-std=c++17', '--shared', '-Xcompiler=-fPIC']
        if not args.release:
            cmd += ['-lineinfo']
        if name == 'swiglu':
            cmd += ['--fmad=false']
        if name == 'fc2_int8':
            include = args.cutlass_include.absolute()
            try:
                include = include.relative_to(ROOT)
            except ValueError:
                pass
            cmd += ['-I' + str(include)]
        cmd += [str(source.relative_to(ROOT)), '-o', str(target.relative_to(ROOT))]
        if name == 'vae_half':
            cmd += ['-lcublasLt', '-lcublas']
        print(f'Building {name}', flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
        evidence.append(dict(name=name, source=source.relative_to(ROOT).as_posix(),
                             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                             binary_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    # Do not record machine paths or compiler command lines in distributable manifests.
    (out / 'manifest.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print('Built:', ', '.join(args.only))


if __name__ == '__main__':
    main()
