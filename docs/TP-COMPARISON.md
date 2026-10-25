# What TP accelerates / 张量并行的对照范围

All measurements below used the project's dual RTX 2080 Ti 22 GB/NVLink host. A speedup belongs to one input, algorithm, timing boundary and historical implementation. The current performance preset has not had a new single-GPU/TP2 whole-block or full-generation A/B test.

## E22 configuration audit / 1.348倍的配置复核

Original normal-run JSONs and runner scripts confirm that the single-GPU and TP2 cases shared:

| Item | Both cases |
|---|---|
| Implementation commit | `6ae7805bfd26e4f5b10b4e3fc7e842163f98945e` |
| Input | Same captured first-step block25 trajectory, x=[17007,5376], from 896×512 / 124 internal frames |
| Model math | Full INT8 ConvRot linears, original AdaLN, same original compute dtype |
| Attention | Kitchen keep0.5, shared-KV mode1; pinned backend `c6bc3cfc58c18c3a6220aeea09116e1eebd91669` |
| Chunk / residency | chunk2048; only block25 resident; weight ballast0; no CPU offload |
| Timing | 3 warmups +12 normal repetitions; host wall-clock block completion; excludes loading/input clone/output checking |

The variable was one full-row single-GPU block versus two output-row shards with direct event-P2P exchange. Single-GPU median **238.039ms**, TP2 **176.618ms**, **1.348×**. The complete block outputs were elementwise equal (max_abs0, relative RMSE0). This single-GPU baseline was extracted from the project's same-operator implementation, not unmodified upstream ComfyUI.

**This is a valid early block comparison, not a measurement of today's speed preset.** It predates chunk4096, tuned FC2, joint SwiGLU and offset routing. It also excludes the whole-model weight streaming/loading that can matter in end-to-end generation.

A separate profiled capture showed attention core52.882ms→about26.8–27.6ms and FC1 36.869ms→about18.8–19.0ms, but FC2/SwiGLU43.581ms→about38.7–39.0ms. The FFN does not halve as a whole in this output-row design, and each rank exchanges activations. Copy intervals overlap compute and cannot be added as net latency. These observations explain why this historical block's total speedup was lower than the attention-only result; they do not predict today's result. [Sanitized E22 report](experiments/E22.md).

中文结论：这组单卡/双卡的输入、代码版本、精度、稀疏率、chunk、预热和计时口径已对齐，1.348倍没有算错。但它是早期同算子的单block实验，不是当前默认档，也不是“整模型只加速1.348倍”。当前配置的纯TP整体收益尚需新的同配置A/B，不能拿旧单卡时间除以新双卡时间。

## Other same-input TP results / 约1.8倍及更高的记录

| Scope | Single GPU | TP2 | Ratio | Boundary |
|---|---:|---:|---:|---|
| E21 optimized Kitchen attention routing+quantization+kernel | 60.307ms | 30.981ms | 1.947× | CUDA events; same full QKV, 56 heads versus28+28; no projections/FFN/gather |
| E29 full VAE FP32 decode | 105.816s | 57.684s | 1.834× | Same latent; second warm decode; no weight loading |
| E29 full VAE FP16 decode | 25.982s | 18.316s | 1.419× | Same dtype, but split-GEMM/reduction rounding differs |
| E36 warm Qwen streaming encode | 5.305s | 2.570s | 2.064× | Same prompt/GGUF type and FP32 computation; both CPU-pack cache hits; includes preparation/release |

E21's matching backend outputs were elementwise equal at this attention scope. E29 FP32 TP versus single FP32 relative RMSE was2.36e-7; FP16 TP versus single-FP32 reference PSNR was52.45dB (single-FP1670.88dB). E36 TP2 conditioning differed from the original path (relative L2≈2.04e-6); its full videos changed despite small conditioning differences. None is a claim of lossless full-video generation. See [E21](experiments/E21.md), [E29](experiments/E29.md), [E36](experiments/E36.md).

1.834倍来自同FP32的完整VAE，约1.95倍来自Attention阶段，2.064倍来自热缓存Qwen编码。三者都不能代替整个Transformer block或完整视频的TP加速比；不同精度之间的提升还包含精度改变。

## NVLink's role / NVLink的作用

TP divides layer work across both cards, then exchanges activations/partial results. The implementation uses direct peer copies and CUDA events. The measured `NV2` connection provides the inter-GPU transport; no CPU staging is required for these transfers. There is no measured NVLink-on/off A/B here, so its isolated speedup is unknown. A two-card setup without verified P2P is not an established substitute for this environment.
