# Installation and integration / 安装与接入

## Tested host application

- ComfyUI **0.42.0**, commit `8b219284fd56145c21fa7522c9ebb85125b80d08` (deployment-documentation commit).
- The two local deployment commits on that checkout changed documentation only. The immediately preceding upstream base is [`46aad33e4d4c0696582611bf450dd93742fa9090`](https://github.com/Comfy-Org/ComfyUI/commit/46aad33e4d4c0696582611bf450dd93742fa9090); no functional ComfyUI-core patch is bundled here. If a deployment-only commit is unavailable from upstream, use the corresponding upstream code revision rather than expecting a private Git object to exist remotely.
- ComfyUI-GGUF commit [`373048b8403a7820620065210a691263d4da0a61`](https://github.com/city96/ComfyUI-GGUF/commit/373048b8403a7820620065210a691263d4da0a61).
- Python 3.11.16; PyTorch 2.11.0+cu130; comfy-kitchen 0.2.39; comfy-aimdo 0.5.6; gguf 0.19.0; frontend 1.57.1.

The `requires-comfyui` metadata is a minimum declaration, not an assertion that every newer release works. Internal H3/model-patcher/execution APIs are used. Preserve your working environment before changing ComfyUI or PyTorch. Linux/SM75 is the measured target; Windows and other GPU architectures have not been validated for the added TP2 paths.

## 1. Install the custom nodes

In the Python environment that already runs ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Luna-AI-Infra/ComfyUI-H3-TP.git
git clone https://github.com/city96/ComfyUI-GGUF.git
git -C ComfyUI-GGUF checkout 373048b8403a7820620065210a691263d4da0a61
```

Do not load both this pack and the upstream MultiStream pack: they register the same node names. Follow the GGUF plugin's own dependency instructions. `dependencies=[]` in this pack does not mean CUDA toolchains, GGUF and optional attention backends are provided automatically. Do not install a different PyTorch wheel blindly into an existing ComfyUI environment.

## 2. Install the attention backend you select

The inherited Sage backend requires [SageAttention](https://github.com/thu-ml/SageAttention); the optional Flash backend requires [FlashAttention for Turing](https://github.com/WeetHet/flash-attention-turing). They are separate projects with their own build requirements. `pytorch_fp32` / `pytorch_fp16` can be used as explicit diagnostic alternatives, but are not the published sparse-performance configuration.

For the benchmarked Sparse Kitchen path:

```bash
cd ComfyUI/custom_nodes/ComfyUI-H3-TP
python scripts/bootstrap_backends.py
```

This fetches [Zironic/H3-Optimizations](https://github.com/Zironic/H3-Optimizations) at `c6bc3cfc58c18c3a6220aeea09116e1eebd91669` into `third_party/H3-Optimizations`. Its native backend may compile on first use. You can instead set `H3_SPARSE_KITCHEN_SOURCE` to a compatible checkout. The backend source is not duplicated into this release.

If Git's TLS transport fails on the host, `python scripts/bootstrap_backends.py --archive` downloads the same pinned revision as a source archive and records its revision marker. It does not disable HTTPS certificate checks or substitute an unpinned branch. A failed/unknown pre-existing directory is not silently overwritten.

## 3. Build the optional performance kernels

Use a CUDA toolkit with SM75 support. CUDA 12.8 was used for the packaged C-ABI libraries; this is distinct from the CUDA 13.0 runtime in the tested PyTorch wheel.

Tuned FC2 additionally needs CUTLASS 3.8.0 headers. The tested CUTLASS commit is `df18f5e4f5de76bed8be1de8e4c245f2f5ec3020`:

```bash
git clone https://github.com/NVIDIA/cutlass.git third_party/cutlass
git -C third_party/cutlass checkout df18f5e4f5de76bed8be1de8e4c245f2f5ec3020
python scripts/build_kernels.py --cutlass-include third_party/cutlass/include
```

Select `--nvcc` or `NVCC` if `nvcc` is not on PATH. The script builds into `kernels/build`; binaries are ignored by Git. For example, `--only swiglu vae_swiglu vae_half` builds only those options without CUTLASS. `kitchen_shared_kv` and `kitchen_offsets` retain their attributed Apache-2.0 source headers. Do not add blanket fast-math options: the original SwiGLU build deliberately uses `--fmad=false`.

Run:

```bash
python scripts/doctor.py --comfyui ../..
```

The separate experimental `sparse_kitchen_sage_int8` consumer also has `python scripts/build_kitchen_sage.py`; this uses PyTorch's extension builder and bundled attributed source. It is not needed for the benchmarked ordinary `sparse_kitchen_int8` preset and is not part of the six-library portability smoke result.

Source defaults enable joint SwiGLU, so build `swiglu` even if you do not select every performance option, or explicitly select `H3_TP_SWIGLU_FUSION=off`. Missing tuned FC2 can fall back to vendor, while other selected native options fail explicitly; check the doctor output rather than assuming all optimizations are active.

## 4. Download models and start

Use [MODELS.md](MODELS.md) and `configs/models.json`. Then:

```bash
python scripts/verify_models.py --comfyui ../..
python scripts/launch_comfyui.py --comfyui ../.. --config configs/conservative.json
```

After kernels/backends are ready, select `configs/performance.json`. The scripts assume they run with the ComfyUI Python environment. Paths above are relative examples, not a required disk layout. The launcher is opt-in; ordinary ComfyUI startup still registers all nodes.

The launcher's `--reserve-vram 15` is the process-level reserve used in the test environment. The TP node's `reserve_vram_gb=4` is a separate workspace/planner input. Neither is a universal recommended allocation for other GPUs.

Import an editor graph from `workflows/ui` or submit a graph from `workflows/api`. Review backend selection, model filenames, GPU indices and precision modes. Keep `H3LoadPipeline.precision_mode` and `H3TPVAEDecode.precision_mode` identical when staging is enabled.

## Optional RAM-resident loader

`configs/performance-ram-ref2va.json` shows the tested five-weight Ref2VA set. The portable launcher resolves its model paths relative to the selected ComfyUI root and installs the RAM loader before normal weight loading. For FL2VA, replace the Ref2VA entry with the FL2VA file; do not preload both unless you need both and have budgeted the additional RAM.

The selected files total about 54.8 GiB before CPU packed copies, model objects and activations. The test machine has 256 GB RAM; this is not a 55 GB system-memory requirement. Linux `mlock` must be permitted for the selected bytes. Configure the service/user's memlock limit appropriately; the launcher checks it and does not silently elevate privileges or change system policy. Optional `numactl --interleave=all` was used on the dual-socket host. Preload can take many minutes and is excluded from request timing.

The loader wraps specific ComfyUI/GGUF internals, and rejects unknown model files while active. It is experimental and should be disabled for unrelated workflows. Start a new process without `ram_weights` to return to ordinary loading. Existing experiments used environment-driven private launch scripts; the public JSON launcher is a packaging adaptation, not a new end-to-end performance result.

## Queueing

Submit multiple `/prompt` requests to one normal ComfyUI server. They share the single worker and execute serially; the TP2 task uses both cards. Do not run two ComfyUI processes against these GPUs expecting cache or memory isolation. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) before running a long unattended queue.
