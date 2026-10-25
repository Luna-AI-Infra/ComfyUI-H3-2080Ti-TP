# Performance, precision and default settings

**Recommended default:** `scripts/launch_comfyui.py --comfyui PATH` selects `configs/performance.json` without `--config`. Build its native dependencies and pair it with the supplied performance workflows. [Speed profiles, output fps and success evidence](SPEED.md).

中文要点：默认启动入口以速度优先，省略config自动加载performance；工作流参数、手动新建节点的源码回退值和进程配置仍需分别理解。`conservative.json`主要减少额外内核依赖，留作显式排查选项，不是全FP32画质对照；`performance.json`复用实验速度配置，其中Kitchen稀疏、局部FP16累加、FP16 VAE和蒸馏8步均有质量取舍。修改环境设置后重启服务，修改缓存外部权重后增加generation。

## Main switches

| Setting / location | Source fallback | Recommended performance deployment | Effect and rollback |
|---|---|---|---|
| TP attention node | `sage_turing_int8` | `sparse_kitchen_int8` | Changes backend/quantization. Choose Sage/Flash/PyTorch explicitly for comparison; no automatic quality equivalence. |
| `sparse_video_keep` / TP node | 0.5, Kitchen only | 0.5 | Retain half target-video KV tiles; conditioning dense. Set 1.0 for a dense-routing control within Kitchen, not to undo all quantization/backend changes. |
| `communication` / TP node | `event_p2p` | `event_p2p` | Direct event-driven exchange. `legacy` is rollback; direct peer access is required for TP2. |
| `chunk_tokens` / TP node | 2048 | 4096 | Activation chunking affects kernels, memory and scheduling. Does not halve precision by itself. |
| `weight_residency` / TP node | `three_groups` | `auto_tokens` in paired API | Bounded empirical planner, groups vary with tokens. `three_groups` / `manual_groups` are explicit alternatives. |
| `num_groups` / TP node | 3 | Planner-selected | Used for manual mode, ignored as a fixed group count in auto mode. |
| `resident_slots` / TP node | 2 | 2 | GPU weight slots; in auto mode this is a maximum. More slots need more VRAM. |
| `residency_safety_gib` / TP node | 2 | 2 | Planner margin, not a universal OOM guarantee; tested model/profile domain is limited. |
| `packed_cache` / TP node | false | true | Reuse CPU packs between jobs. Disable to avoid retained pack RAM; invalidate on weight mutations. |
| `cpu_pack_layout` / TP node | `canonical` | `canonical` | Canonical CPU masters plus group views avoid repacking after group changes. `grouped` restores legacy layout. |
| `cache_generation` / relevant nodes | 0 | 0 | Increment after out-of-band changes; cache keys cannot reliably detect arbitrary `.data`/inference-tensor mutations. |
| `H3_TP_FC2_GEMM` / environment | `vendor` | `sm75_tuned` | Selected SM75 INT8→INT32 FC2 shapes. Missing library/unsupported shapes use vendor; check whether it activated. |
| `H3_TP_SWIGLU_FUSION` / environment | `joint` | `joint` | Shared main/LoRA activation work. `off` disables; requires built kernel when active. |
| `H3_KITCHEN_SHARED_KV` / environment | `0` | `1` | Shared-KV kernel path. Needs Kitchen dependency and `kitchen_shared_kv` library. |
| `H3_KITCHEN_KERNEL_PLAN` / environment | `baseline` | `offset` | Offset index packing; requires shared-KV. `baseline` rolls back. Other barrier variants remain experiments. |
| `H3_TP_INTERNAL_OUTPUT` / environment | `original` | `original` | **Keep original**. `linears_fp16` failed with non-finite FC2; `ffn_fp16` is not validated as safe either. |
| Qwen execution / node | `original` | `stream_tp2` | Immutable CPU packs and two-slot streaming; fixed H3 architecture, unpatched GGUF weights only. `original` rolls back. |
| Runtime Larry `compute_precision` / node | FP32 | FP32 | FP16 increment failed with audio NaNs; do not confuse checkpoint BF16 storage with runtime math. |
| VAE `precision_mode` / decode+pipeline | `tp2_fp16` | `tp2_fp16` | Lower-precision weights/activations, FP32 partials/residual operations where specified. `single_fp32` is a separate diagnostic/reference path. |
| VAE `ffn_fusion` / decode node | `off` | `vector` | Explicit VAE activation fusion; `off` restores operations. Vector kernel is SM75/layout guarded. |
| `H3_VAE_FFN_ACCUM` / environment | `fp32` | `fp16` | **Approximate** FC1/FC2 half accumulation and FC2 half output. Set `fp32` to disable; validated only for TP2 FP16/SM75. |
| Pipeline `enabled` / node | false | true | Prepare decoder CPU shards; pair precision/generation with decode node. |
| Pipeline `prefetch_last_euler_step` / node | false | true | Donate/prefetch decoder slots on the last eligible Euler step. Only zero final sigma/single-condition use validated. |
| Reference `encode_mode` / DP node | `auto` | `auto` or explicit `dp2` | Whole-reference data parallelism; one item falls back to single. FP32 multi-item validation exists; half-video cases remain gated. |
| `allow_unvalidated_video` / DP node | false | false | Keep false: long/non-FP32 video reference DP showed mismatches in historical tests. |
| RAM loader / launch JSON | absent | optional separate preset | Locked anonymous host copies, not a precision change. Remove `ram_weights` and restart to disable. |
| Steps / native sampler | graph-defined | 8, Euler/simple | Turbo LoRA, schedule and step count affect quality; not a lossless implementation-only optimization. |

The portable launcher disables global TF32, FP16 reduced-precision reduction and global FP16 accumulation. The VAE's local FP16 cuBLASLt path is separate and does **not** enable global half accumulation.

## Presets

- `configs/conservative.json`: vendor FC2, fusion off, Kitchen plan baseline/shared-KV off, local VAE FFN FP32 accumulation, main internal output original. Change the graph's attention/precision options explicitly if you need a different quality control.
- `configs/performance.json`: **launcher default when `--config` is omitted**, containing the explicit environment portion of the performance experiments. Use matching `workflows/api` for graph settings. Does not enable RAM preload. The file is resolved from the installed plugin, not from the working directory.
- `configs/performance-ram-ref2va.json`: performance settings plus the five Ref2VA weights. Requires Linux memlock capacity and substantial spare RAM. Original E75 used the same math, but the public launcher itself has not yet completed an E75-sized end-to-end rerun.

## Measured precision effects

E51 isolated VAE FFN: FP16 accumulation improved about 1.38–1.42× over its FP32-accumulation baseline, with relative RMSE around 0.42%. This also changes FC2 output rounding. It is neither a pure accumulator-only experiment nor a complete VAE quality bound.

E39 real VAE block: mixed 41.364 ms versus FP16 TP2 17.546 ms, with real P2P and thread/synchronization costs included. This is a block comparison, not a full-video speedup or proof of identical images.

E52 combined full generation: successful at 110.617 s versus 115.734 s, with several options changed together. The initial internal-FP16-output combination failed and was withdrawn. Combinations must not be credited as independent multiplicative gains.

## Diagnostics

`H3_TP_DIAGNOSTIC_SYNC=0` is the normal performance setting. Enabling it adds synchronization and must be labeled diagnostic, not used as a normal speed result. Profiling/measurement environment variables in the source are opt-in and can add overhead or write output files. Keep them off for ordinary use. Node timing, CUDA-event spans and kernel traces measure different scopes; see the paper for the measurement definitions.
