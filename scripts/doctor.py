"""Report runtime capabilities without loading model weights or launching CUDA kernels."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comfyui', type=Path)
    args = ap.parse_args()
    out = dict(python=platform.python_version(), platform=platform.system(), packages={}, native_kernels={})
    for name in ('torch', 'comfy-kitchen', 'comfy-aimdo', 'gguf', 'comfyui-frontend-package'):
        try:
            out['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out['packages'][name] = None
    for name in ('fc2_int8', 'swiglu', 'kitchen_shared_kv', 'kitchen_offsets', 'vae_swiglu', 'vae_half'):
        out['native_kernels'][name] = (ROOT / 'kernels/build' / f'lib{name}.so').is_file()
    out['sparse_kitchen_checkout'] = (ROOT / 'third_party/H3-Optimizations/h3_optimizations').is_dir()
    if importlib.util.find_spec('torch'):
        import torch
        out['cuda_available'] = torch.cuda.is_available()
        out['cuda_runtime'] = torch.version.cuda
        out['devices'] = [dict(index=i, name=torch.cuda.get_device_name(i),
                               capability=list(torch.cuda.get_device_capability(i)))
                          for i in range(torch.cuda.device_count())]
        out['p2p_0_1'] = torch.cuda.device_count() > 1 and torch.cuda.can_device_access_peer(0, 1)
    if args.comfyui:
        out['comfyui_main_exists'] = (args.comfyui / 'main.py').is_file()
        out['gguf_plugin_present'] = (args.comfyui / 'custom_nodes/ComfyUI-GGUF').is_dir()
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
