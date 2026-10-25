# Attribution and provenance

This is an experimental derivative of [Márton Sági's Comfy-H3-MultiStream](https://github.com/martonsagi/Comfy-H3-MultiStream). Its GPL-3.0-only license and source copyright headers are retained. The snapshot used for public preparation was H3 source commit `4369debd37e4f5a426796417fdcdeb5d2cf0e5c7` in the development tree. A fresh public history is intentional: it avoids distributing private author-email history and machine-specific deployment files.

The fork adds TP2 H3 runtime work, communication and streaming/caching changes, Qwen/decoder/reference-encoder paths, and benchmarked SM75 kernel adapters described in the settings and experiment paper. It also packages native sources and replaces developer-specific library paths; packaging does not establish new end-to-end benchmark results.

Dependencies and adapted sources:

| Project / author | Role |
|---|---|
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | Host, H3 model/sampling/VAE implementation, model management and node API. |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | GGUF encoder loading. |
| [H3-Optimizations](https://github.com/Zironic/H3-Optimizations) | Sparse Kitchen router, layout and native attention backend; pinned external checkout. |
| [SageAttention](https://github.com/thu-ml/SageAttention) | Optional Sage backend and attributed SM75 source dependencies. |
| [FlashAttention for Turing](https://github.com/WeetHet/flash-attention-turing) | Optional dense FP16 backend. |
| [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) | Template implementation behind the tuned INT8 GEMM. |
| [Larry MiniMax H3 Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) | Optional distilled checkpoint used in8-step experiments. |
| MiniMax / Comfy-Org / GGUF checkpoint authors | Model assets, distributed separately at the links in MODELS.md. |

Attribution/license files embedded in `multistream/kitchen_sage_src` and the Kitchen kernel source directories remain in place. See [UPSTREAM-NOTICE.txt](UPSTREAM-NOTICE.txt) for the upstream notice verbatim. Any authorization described there belongs to its named holder and must not be represented as authorization granted to this fork or its users.

The prebuilt operator release retains the same source terms and includes the relevant notices. Tuned FC2 uses the pinned CUTLASS 3.8.0 headers; [its BSD license](licenses/CUTLASS-LICENSE.txt) accompanies the binary archive. Corresponding kernel source is available at the binary release tag. CUDA/cuBLAS runtimes and model weights are not redistributed. See [prebuilt compatibility and installation](PREBUILT.md).

The three public videos are selected generated experimental outputs, remuxed without private container metadata. Historical input reference images are not distributed. Reports are selected redacted copies; raw machine logs, giant profiler captures, SSH/admin helpers, checkpoints, original private author history and private input assets are intentionally excluded.
