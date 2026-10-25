# Prebuilt SM75 operators / 预编译算子

The [v0.1.1-sm75 release](https://github.com/Luna-AI-Infra/ComfyUI-H3-2080Ti-TP/releases/tag/v0.1.1-sm75) provides six libraries for the measured **Linux x86_64 / dual RTX 2080 Ti 22 GB (SM75)** setup:

| Library | Purpose |
|---|---|
| `libfc2_int8.so` | Tuned INT8→INT32 FC2 GEMM |
| `libswiglu.so` | Joint main/LoRA SwiGLU |
| `libkitchen_shared_kv.so` | Shared-KV attention path |
| `libkitchen_offsets.so` | Offset attention routing path |
| `libvae_swiglu.so` | Vector VAE SwiGLU |
| `libvae_half.so` | Opt-in local FP16 VAE FFN accumulation |

From the installed plugin directory, with the ComfyUI Python:

```bash
python scripts/install_prebuilt.py
python scripts/doctor.py --comfyui ../..
```

The installer checks the pinned archive SHA256, individual library hashes and local kernel-source hashes, then checks ELF dependency loading before copying files into `kernels/build`. It does not run GPU computations or modify system dependencies. Existing different libraries require an explicit `--replace`; back up your local build first. It does not enable any GPU unlock.

For an offline/slow-network host, download the release asset elsewhere and copy it to the host:

```bash
python scripts/install_prebuilt.py --archive h3-sm75-linux-x86_64-cuda12.8-v0.1.1-sm75.tar.gz
```

## Compatibility and dependencies

- Built with **CUDA Toolkit 12.8**, pinned **CUTLASS 3.8.0**, targeting `sm_75`; Linux x86_64, Ubuntu22.04/glibc2.35 build environment.
- `libvae_half.so` requires **CUDA 12 `libcublasLt.so.12`**. A PyTorch CUDA13 wheel alone does not supply that SONAME. Install a compatible CUDA12 cuBLAS runtime through the appropriate NVIDIA/system package channel, or use the matching existing toolkit runtime. NVIDIA runtime libraries are not bundled.
- Ensure the CUDA12 library directory is visible to the dynamic linker. For an existing default CUDA12.8 toolkit, a per-process example is `export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}`. Use your installation's actual library directory; do not replace unrelated services' environments.
- Other dependencies include compatible glibc/libstdc++, a compatible NVIDIA driver and the ComfyUI/PyTorch environment described in [INSTALL.md](INSTALL.md). These are C-ABI libraries, not Python extension wheels; they do not embed model weights or the full Sparse Kitchen backend.
- Run `python scripts/bootstrap_backends.py` separately for the pinned attention backend. Prebuilt custom operators do not install GGUF, ComfyUI or model assets.
- Windows, non-x86 hosts, other GPU architectures and stock 11GB 2080 Ti memory capacity are not validated by this release. Use source builds and separate validation rather than assuming compatibility.

The archive omits profiling line information and includes source/build metadata, hashes, GPL and relevant CUTLASS/Apache notices. Corresponding sources are in the same release tag under `kernels/`; the pinned CUTLASS revision is recorded. [Attribution](ATTRIBUTION.md).

## Build instead

The default inference choice remains the performance preset; a source build is an alternative way to obtain its operators:

```bash
git clone https://github.com/NVIDIA/cutlass.git third_party/cutlass
git -C third_party/cutlass checkout df18f5e4f5de76bed8be1de8e4c245f2f5ec3020
python scripts/build_kernels.py --release --cutlass-include third_party/cutlass/include
```

Select CUDA12.8 `nvcc` through `--nvcc` or `NVCC` if it is not on PATH. Omit `--release` for a local profiling build with line information; do not distribute such a build without reviewing embedded machine paths.

## 中文使用说明

Release 提供上述六份 `.so`，专门面向 **Linux x86_64、2080 Ti / SM75**，不是 Windows 算子包。在插件目录执行 `python scripts/install_prebuilt.py` 即可下载并校验，通常无需安装编译器。算子包之外仍需要 ComfyUI、GGUF、权重、Sparse Kitchen 和 **CUDA12 的 cuBLASLt 运行库**；只有 PyTorch CUDA13 不够。

安装器会核对压缩包、每个库和本地算子源码的 SHA256，先检查动态库依赖再写入 `kernels/build`。已有不同库时需要显式 `--replace`，建议先备份；脚本不会自动更改系统 CUDA 或服务配置。离线安装用 `--archive` 指向下载的同一压缩包，校验不变。

发布构建不带 profiling 行号信息，附带许可证与对应源码版本。预编译算子的小型双卡校验不等于完整视频、所有输入或长期队列已经验证；完整生成仍按 README 的实验范围理解。
