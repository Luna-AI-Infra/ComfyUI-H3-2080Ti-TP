# MiniMax H3 on Dual Turing GPUs

## Tensor Parallelism, Reusable Host Packs and Explicit Precision Trade-offs

Luna-AI-Infra | Engineering experiment report | 25 October 2026

## Abstract

This report documents an experimental ComfyUI implementation for MiniMax H3 on two RTX 2080 Ti GPUs with 22 GB VRAM each. The work combines TP2 execution, event-driven peer exchange, grouped weight streaming, canonical CPU pack reuse, Qwen streaming, staged VAE decode, and optional SM75 kernels. A matched combined configuration reduced a five-second video's complete request time from 115.734 to 110.617 seconds. In a separate length-changing sequence, canonical caching reduced preparation from 62.234 seconds to approximately 0.5 seconds while avoiding repacking. Successful outputs include five-, ten- and fifteen-second 480p videos. These observations are not additive speedups or guarantees of unchanged quality. Precision changes have local error measurements, some faster candidates were withdrawn, and repeated-job CUDA failures remain unresolved.

## 1. Scope and contributions

The objective is practical execution on an older dual-socket DDR3 host with limited per-GPU memory. The contribution is a set of integrated engineering changes and documented trade-offs, rather than a new video model or a claim of state-of-the-art inference speed.

- A dedicated two-GPU H3 runtime with integer linear layers, chunked peer exchange and bounded grouped weight residency.
- Canonical immutable host packs that can serve multiple GPU groupings without repacking the model after a length change.
- Separate optimizations for the Qwen encoder, reference encoders and video VAE, exposed through ComfyUI nodes and explicit process settings.
- Attributed CUDA sources, portable build/launch helpers and a record retaining unsuccessful as well as successful experiments.

The implementation is based on Comfy-H3-MultiStream. Its added TP2 path is distinct from the inherited general MultiStream path. The measured domain is two SM75 GPUs with direct peer access; arbitrary GPU counts, newer architectures and multi-process concurrent rendering are outside the demonstrated scope.

This is an engineering report, not a peer-reviewed study. Most complete-generation results are individual observations on one host. The public package is a new source export; historical speed records are not automatically a performance validation of every packaging change.

<!-- pagebreak -->

## 2. Experimental environment and models

| Component | Verified configuration |
|---|---|
| Host | Supermicro X9DAi; two Xeon E5-2620 v2 at 2.10 GHz; 12 physical cores / 24 threads total |
| Memory | 4 x 64 GB DDR3; 256 GB installed; configured 1066 MT/s; two NUMA nodes |
| GPUs | Two RTX 2080 Ti, 22 GB each; NVLink, NV2 topology |
| System | Ubuntu 22.04.5 LTS; Linux 5.15.0-191-generic; NVIDIA driver 595.91.07 |
| Runtime | Python 3.11.16; PyTorch 2.11.0+cu130; PyTorch CUDA runtime 13.0 |
| Kernel builds | CUDA Toolkit 12.8; CUTLASS 3.8.0 for tuned INT8 FC2 |
| ComfyUI | 0.42.0; runtime base 46aad33e4d4c0696582611bf450dd93742fa9090 |
| Dependencies | comfy-kitchen 0.2.39; comfy-aimdo 0.5.6; gguf 0.19.0; frontend 1.57.1 |

The recorded application HEAD 8b21928 adds deployment documentation to the runtime base. The recorded final H3 runtime is 4369debd. No ComfyUI-core functional patch or GPU-unlock mechanism is required by this public package. Earlier experiments used earlier H3 implementation commits, identified in their reports.

| Asset | Tested representation |
|---|---|
| Main diffusion model | Full, unpruned FL2VA or Ref2VA INT8 ConvRot; original AdaLN structure |
| Conditioning encoder | H3-specific Qwen3-VL-32B GGUF, Q4_K_M; 50-layer variant |
| Video / audio VAE | FP16 video checkpoint / FP32 audio checkpoint |
| Turbo adapter | Larry v4 step600 EMA, BF16 checkpoint storage, runtime FP32 increment at strength 1 |

FL2VA and Ref2VA are different task variants. The encoder's Q4_K_M storage does not mean that the diffusion transformer is four-bit. Model filenames, exact bytes, SHA256 and revision-pinned author download URLs are supplied in the repository manifest; weights are not redistributed.

## Measurement contract

Complete request latency is ComfyUI history execution_start to execution_success. It includes conditioning, preparation, sampling, VAEs and saving, but excludes queueing, process startup and one-time RAM preload. Cold first-request packing, hot cache reuse and separate profiler captures are labeled independently. Failed runs have no successful end-to-end latency. Nested ranges are not added to their parent scopes.

<!-- pagebreak -->

## 3. Weight residency and canonical host caching

The main runtime streams contiguous groups through two GPU slots per rank. Copy streams and CUDA events connect prefetch with the last consumer of a slot. Event-driven peer exchange transfers activation chunks directly between the cards. These changes target movement and scheduling; they are separate from quantization or sparse attention.

Canonical packing retains one immutable CPU master per rank. A new token count can change the GPU grouping while creating views of the same host layout. The alternative grouped layout could repack the weights after a grouping change. Cache keys include model identity, relevant tensor state, adapter information and an explicit generation number. Out-of-band tensor mutations still require manual invalidation.

### E60: a successful length-changing sequence

| Request | End-to-end | Preparation | CPU pack |
|---|---:|---:|---:|
| Text-only 5 s, cold | 216.657 s | 62.234 s | 58.015796 s |
| Text-only 10 s, cache hit | 235.156 s | 0.504 s | 0.000006 s |
| Text-only 5 s, warm return | 123.774 s | 0.506 s | 0.000006 s |

All three used 864 x 480, 24 fps, seed 231058, Larry 8-step Euler/simple, full FL2VA INT8 and Kitchen 50%. The second request has more tokens and is not expected to be faster than the first purely because it is warm. Group-view creation took approximately 5.5-6.7 ms. A microsecond CPU-pack hit means that packing was skipped, not that complete preparation was free.

![E60 stage totals](../figures/e60-stage-times.png)

Figure 1. Mutually exclusive host-stage totals, with residual time placed at the end. This is a stacked stage breakdown, not an exact GPU event trace. Preparation is inside the sampler scope and is not added again.

Relative to historical E58 observations, the 10-second request used 24.6% less time and the 5-second return 31.1% less. Different cache histories and single observations prevent assigning the entire difference causally to canonical packing. Cache HIT and eliminated repacking are the direct observations.

<!-- pagebreak -->

## 4. Complete-video measurements

| Experiment | Output / request state | Complete request |
|---|---|---:|
| E52 baseline | FL2VA 5 s, warm | 115.734 s |
| E52 combined | Same scene/seed, warm | 110.617 s |
| E57 | FL2VA 15 s, first request after RAM preload | 501.450 s |
| E60 | FL2VA text-only 10 s, canonical cache hit | 235.156 s |
| E60 | FL2VA text-only 5 s, warm return | 123.774 s |
| E65 | Ref2VA 5 s, two references, cold | 256.894 s |
| E75 | Ref2VA 5 s, two references, cold, stock GPU state | 223.048 s |

All entries use 864 x 480 and 24 fps, with output frame counts of 120, 240 or 360 for five, ten or fifteen seconds. They differ in task, model variant and cache state; they are not one leaderboard of equivalent inputs. E57 uses first/last-frame references, while E60 is text-only.

### The matched E52 combination

The final combination retained original main-transformer internal outputs and enabled chunk4096, tuned FC2, offset routing, vector VAE activation fusion and local FP16 VAE FFN accumulation. Runtime LoRA remained FP32. Request latency decreased by 4.42%, or 1.046x, in one run per setting. Video VAE time decreased from 18.922 to 16.692 seconds. Because several settings changed together, this does not isolate each contribution or establish losslessness.

The initial all-enabled combination changed internal linear outputs to FP16. It produced non-finite values at main block18 FC2 and was withdrawn. Its fast step timing must not be counted as a successful result.

### Capacity and startup are separate

E57 delivered 360 video frames with audio. The first request included CPU packing; the 937.343-second RAM preload happened before the request and is excluded from 501.450 seconds. Five selected weight files consumed about 54.8 GiB before CPU packed copies and other runtime allocations. Recorded peak GPU usage was 19.477 / 17.135 GiB and minimum available host RAM about 89.5 GiB. This was not a second hot-run benchmark.

The bundled videos are E52 five seconds, E60 ten seconds and E57 fifteen seconds. They demonstrate completed outputs, not a blinded perceptual quality evaluation. Private historical reference inputs are omitted, so those input-dependent examples cannot be reproduced byte-for-byte from the public assets alone.

<!-- pagebreak -->

## 5. Local kernels and precision effects

| Scope | Baseline to candidate | Interpretation |
|---|---|---|
| E39 real TP2 VAE block | Mixed 41.364 to FP16 17.546 ms | 2.358x at block scope; different precision |
| E44 original-output TP2 main block | 223.272 to 220.378 ms | 1.013x; smaller than selected isolated GEMM gains |
| E51 B4 VAE FFN, GPU0/1 | 7.658/7.407 to 5.376/5.355 ms | About 1.38-1.42x, local FP16 accumulation |
| E65 FP32 eight-image reference fixture | 3.776 to 1.426 s | 2.647x; fixture outputs elementwise equal |

E39 measures a real B4/T1797/C2048 block including two P2P reductions, thread dispatch, casting and synchronization; it is not a full decoder. The normal comparison used interleaved repeated samples without a profiler. E44's integer checks establish local INT32 results for the tested shapes, not full-video equality.

![E44 FC2 sweep and block context](../figures/e44-fc2-tuning.png)

Figure 2. Only the original-output-precision whole-block comparison is shown: 223.272 versus 220.378 ms. The modest 1.013x effect is not an isolated-GEMM speedup. Split-K and asynchronous candidates were not promoted based on isolated speed alone.

E51 changed both accumulation and FC2 output rounding. Relative RMSE around 0.42% was measured for the real local FFN. It is not an accumulator-only ablation, a complete-VAE error bound or an assertion that the resulting video is perceptually unchanged. INT8 main GEMMs accumulate in INT32 before rescaling; a FP32 output tensor must not be confused with FP16 multiplication plus FP32 Tensor Core accumulation.

<!-- pagebreak -->

## 6. Overlap evidence and failure-driven exclusions

![E52 independent CUDA-event capture](../figures/e52-transfer-overlap.png)

Figure 3. Final-group compute-stream spans and decoder uploads from a separate event-capture run. Stream spans include waits; they are not kernel-active utilization. The capture's elapsed time is not the normal 110.617-second performance result. Overlapping copy/compute time cannot be added twice as a wall-clock saving.

Important exclusions retained in the record:

- FP16 runtime Larry increment caused audio NaNs in a full run. The recommended presets retain FP32 increments.
- Main internal FP16 outputs failed in E52. The final performance configuration keeps original outputs.
- Some half-precision long-video reference-DP cases were faster but numerically mismatched. Default gating was retained rather than labeling them validated acceleration.
- E58's 720p five-second attempt failed with an illegal instruction. The following 720p ten-second request was not submitted. Failed elapsed time is not a generation result.
- Historical stock-state E61/E64 and repeated E73 requests recorded illegal-instruction/Xid errors. The root cause is unresolved; a recent cold request's success is not a reliability cure.

E75's stock-state Ref2VA request completed in 223.048 seconds. An exploratory modified-state cold observation was 222.385 seconds, with stable-step medians 10.205 / 10.200 seconds respectively. Individual observations show no meaningful main-sampling advantage, but are not statistical equivalence tests. No GPU-unlock mechanism is distributed or required.

The evidence supports useful local and workflow-level optimizations while also showing that latency, precision and reliability must be evaluated separately.

<!-- pagebreak -->

## 7. Explicit defaults and deployment controls

| Setting | Source default | Performance preset |
|---|---|---|
| Attention / keep fraction | Sage / Kitchen keep0.5 if selected | Kitchen INT8 / 0.5 |
| Chunk / residency | 2048 / three_groups | 4096 / auto_tokens in API |
| CPU packs | packed_cache false, canonical layout | packed_cache true, canonical |
| Tuned FC2 | vendor | sm75_tuned |
| Main SwiGLU | joint | joint |
| Shared-KV / index plan | off / baseline | on / offset |
| Main internal output | original | original |
| Qwen execution | original | stream_tp2 |
| VAE policy / activation fusion | tp2_fp16 / off | tp2_fp16 / vector |
| VAE FFN accumulation | fp32 | fp16, explicitly approximate |
| Decoder staging / final-step prefetch | false / false | true / true in paired graph |
| RAM loader | absent | optional separate launch configuration |

Node inputs and environment variables are separate. Copying a workflow does not reproduce environment-controlled fusion, accumulation or index-plan settings. The conservative launch preset disables additional native paths and selects FP32 VAE FFN accumulation; it is not an all-FP32 quality-reference graph. Joint source-default fusion still requires its library unless explicitly disabled.

The public launcher disables global TF32/reduced-precision reduction/global FP16 accumulation. The local half-accumulation backend is independent. Pipeline/decode precision and cache-generation settings must match. Final-step decoder donation is validated only for the eligible Euler, zero-final-sigma, single-condition configuration.

ComfyUI integration is through custom nodes, with no model weights bundled. The source export adds portable kernel paths, build scripts and a configurable RAM launcher. E77 rebuilt all six C-ABI libraries and passed small dual-SM75 checks: exact integer FC2 results and selected SwiGLU/vector outputs; finite local half-accumulation outputs. These are packaging/kernel smoke tests, not a fresh complete-video validation.

Use the native serial queue. Three concurrently submitted lightweight control jobs completed serially. E60 supplies a successful three-render sequence, but unattended heavy-job stability remains open. After a fatal GPU error, do not assume ordinary queue continuation restores a valid CUDA context.

<!-- pagebreak -->

## 8. Reproducibility, limitations and conclusions

Reproduction requires the correct FL2VA/Ref2VA checkpoint, H3 Qwen variant, LoRA, VAE policy, backend, sampler, seed, size/frame count, process settings and cache history. The public repository provides exact model hashes/download revisions, editor/API examples, presets and attributed native sources. Selected historical reports are redacted and are not byte-identical raw logs; original source-report hashes preserve provenance without publishing private machine paths.

Limitations include a single host, mostly single full-video observations, limited seeds/scenes, incomplete perceptual ablation, input references omitted for privacy, empirical planner bounds, unvalidated hardware/software combinations and unresolved repeated-job CUDA errors. The public launcher's relocated RAM setup still needs a complete model-run regression. Performance numbers in this report refer to their recorded historical implementations, not every public packaging revision.

The strongest direct observations are successful low-memory TP2 generation, eliminated CPU repacking across a length-changing sequence, small original-precision whole-block gains from tuned kernels, and explicit precision/performance trade-offs in the VAE. A 110.617-second five-second-video result is meaningful only with its cache state, task and precision combination stated. Local speedups cannot be multiplied into a promised end-to-end gain, and local equality cannot establish lossless complete video.

## References and experiment provenance

- [1] Comfy-H3-MultiStream, upstream source and attribution: https://github.com/martonsagi/Comfy-H3-MultiStream
- [2] ComfyUI, H3 runtime base: https://github.com/Comfy-Org/ComfyUI
- [3] MiniMax H3 checkpoint collection: https://huggingface.co/Comfy-Org/MiniMax-H3
- [4] H3 Q4_K_M encoder: https://huggingface.co/leejet/MiniMax-H3-GGUF
- [5] Larry MiniMax H3 Turbo LoRA: https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora
- [6] Sparse Kitchen dependency: https://github.com/Zironic/H3-Optimizations
- [7] NVIDIA CUTLASS: https://github.com/NVIDIA/cutlass
- [8] This release: https://github.com/Luna-AI-Infra/ComfyUI-H3-TP

E52 implementation eb7ab3c; E57 implementation 1d1a3ba; E60 recorded source bb1fd3c above canonical implementation b1c39d2; E75 runtime 4369debd. The public source export starts fresh Git history and does not redistribute private historical commit/email metadata. The experiment IDs link the tables and figures to the sanitized reports included in docs/experiments.
