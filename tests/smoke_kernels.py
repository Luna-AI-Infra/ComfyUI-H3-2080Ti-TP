"""Small GPU correctness/finite checks for packaged kernels, not generation benchmarks."""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    package = '_h3_kernel_check'
    if package not in sys.modules:
        p = types.ModuleType(package)
        p.__path__ = [str(ROOT / 'multistream')]
        sys.modules[package] = p
    spec = importlib.util.spec_from_file_location(package + '.' + name, ROOT / 'multistream' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, 'allow_fp16_accumulation'):
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
    native = module('native_libs')
    for name in ('fc2_int8', 'swiglu', 'kitchen_shared_kv', 'kitchen_offsets', 'vae_swiglu', 'vae_half'):
        ctypes.CDLL(native.library_path(name))
    swiglu = module('swiglu_fusion')
    vae = module('vae_ffn_accum')
    vector = module('vae_swiglu_kernel')
    rows = []
    with torch.inference_mode():
        for rank in range(torch.cuda.device_count()):
            if torch.cuda.get_device_capability(rank) != (7, 5):
                continue
            with torch.cuda.device(rank):
                torch.manual_seed(123)
                for dtype in (torch.float16, torch.float32):
                    x = torch.randn(32, 128, device=rank, dtype=dtype) * .5
                    g, v = x.chunk(2, -1)
                    expected = F.silu(g) * v
                    y, z = swiglu.fused(x)
                    tolerance = 2e-3 if dtype == torch.float16 else 2e-6
                    assert torch.allclose(y, expected, atol=tolerance, rtol=tolerance)
                    assert torch.isfinite(z).all()
                    rows.append(dict(rank=rank, case='swiglu', dtype=str(dtype),
                                     max_abs=float((y.float() - expected.float()).abs().max()), passed=True))
                    if dtype == torch.float16:
                        y = vector.fused(x)
                        assert torch.allclose(y, expected, atol=2e-3, rtol=2e-3)
                        rows.append(dict(rank=rank, case='vae_vector', passed=True,
                                         max_abs=float((y.float() - expected.float()).abs().max())))
                os.environ['H3_VAE_FFN_ACCUM'] = 'fp16'
                x = torch.randn(32, 64, device=rank, dtype=torch.float16) * .1
                w = torch.randn(64, 64, device=rank, dtype=torch.float16) * .1
                y = vae.linear(x, w, output_fp32=True)
                expected = x.float() @ w.float().T
                assert torch.isfinite(y).all()
                rows.append(dict(rank=rank, case='vae_half', passed=True,
                                 relative_rmse=float(((y-expected).square().mean()/expected.square().mean()).sqrt())))
                a = torch.randint(-8, 8, (128, 14336), device=rank, dtype=torch.int8)
                w = torch.randint(-8, 8, (2688, 14336), device=rank, dtype=torch.int8)
                out = torch.empty(128, 2688, device=rank, dtype=torch.int32)
                lib = ctypes.CDLL(native.library_path('fc2_int8'))
                lib.launch.argtypes = [ctypes.c_int]*5 + [ctypes.c_void_p]*5
                lib.launch.restype = ctypes.c_int
                code = lib.launch(2, 128, 2688, 14336, 1, a.data_ptr(), w.data_ptr(), out.data_ptr(), None,
                                  torch.cuda.current_stream().cuda_stream)
                assert code == 0
                expected = torch._int_mm(a, w.T)
                torch.cuda.synchronize(rank)
                assert torch.equal(out, expected)
                rows.append(dict(rank=rank, case='fc2_int8', passed=True, full_int32_equal=True))
    assert rows, 'No SM75 device was tested'
    result = dict(scope='Small local kernel smoke checks; no full-block/VAE/model/quality claim', cases=rows)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
