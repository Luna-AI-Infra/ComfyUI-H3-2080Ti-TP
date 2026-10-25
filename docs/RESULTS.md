# Selected experimental results / 关键实验结果

This is an engineering measurement record, not a peer-reviewed benchmark or a claim of lossless inference. Full discussion is in the [English paper](paper/H3-TP-Experiments.en.pdf) and [中文论文](paper/H3-TP-Experiments.zh-CN.pdf). Selected sanitized historical reports are in [experiments](experiments/).

## Measurement contract

Complete request latency means ComfyUI history `execution_start → execution_success`: conditioning/model preparation/sampling/VAEs/frame processing/encoding/saving, excluding queueing, process startup and RAM preload. A failed request has **no successful latency**. Cold first-request packing, warm cache reuse and profiler captures are separate categories. Inner scopes belong to outer scopes and must not be added twice. Event spans may include waits and are not kernel-active time.

Most full-generation entries have one observation. Different durations, token counts, conditioning tasks, seeds and caching histories do not form strict speedup comparisons. Reported times belong to their historical commits, not automatically to every later packaging change.

## Full generation

| Experiment | Configuration / scope | Successful request time | Interpretation |
|---|---|---:|---|
| E52 baseline | FL2VA, 864×480, 5 s, warm, seed231024 | 115.734 s | Same-scene paired baseline, one run. |
| E52 combined | Same scene/seed, chunk4096 + tuned FC2 + offset + vector VAE + local FP16 accumulation | 110.617 s | 1.046×, 4.42% less time; multi-factor and precision-changing. |
| E57 | FL2VA first/last-frame, 15 s/360 output frames, seed231025, first request after RAM preload | 501.450 s | CPU pack included; 937.343 s startup preload excluded. No second hot run measured. |
| E58 | FL2VA text-only 5 s / 10 s, seed231058 | 179.656 / 311.940 s | Historical cache/packing behavior, not a paired comparison against all later versions. |
| E60 | FL2VA text-only 5 s cold → 10 s → 5 s return, seed231058 | 216.657 → 235.156 → 123.774 s | Three successful sequential requests, cache MISS→HIT→HIT. |
| E65 | Ref2VA, two images, cold, seed231058 | 256.894 s | Reference encoding FP32; reference DP 1.5475 s. |
| E75 | Ref2VA, same two-reference workload, stock GPU state, cold, seed231058 | 223.048 s | Current successful single-run check; stable 50-block step median10.205 s. |

All rows use full INT8 ConvRot main weights, 24fps, Larry v4 step600 EMA/FP32 increment and Euler/simple8, but model variant and cache states differ. The E57 frames use private first/last reference assets, which are not distributed. No direct E57↔E60 speed ratio is claimed.

![CPU pack reuse](figures/e60-stage-times.png)

E60 direct evidence for canonical caching: sampler preparation **62.234 → 0.504 → 0.506 s**; CPU packing **58.015796 → 0.000006 → 0.000006 s**; group-view setup **5.527 → 6.664 → 6.168 ms**. Microsecond cache-hit pack scopes indicate skipped work, not zero total preparation cost. At different lengths the CPU master was reused while GPU groups changed. Relative to E58, E60's10-second request used24.6% less time and the5-second warm return31.1% less, but cache histories differ and these are not randomized A/B estimates.

## Local computation and precision

| Experiment | Measured scope | Result | Accuracy boundary |
|---|---|---|---|
| E39 | Real TP2 VAE block, B4/T1797/C2048 | Mixed41.364 ms → FP16 TP17.546 ms, 2.358× | Local precision variants; not a full-VAE speedup or pixel identity. |
| E44 | Main transformer TP2 block, original output precision | 223.272 → 220.378 ms, 1.013× | Tuned INT8 GEMM local exactness checked; much smaller whole-block gain than selected isolated GEMMs. |
| E51 | Real B4 VAE FFN, no P2P | FP32 accumulation7.658/7.407 ms → FP16 accumulation5.376/5.355 ms | ~1.38–1.42×; relative RMSE~0.42%; FC2 half-output rounding changes too. |
| E65 | FP32 eight-image reference-encoder fixture | 3.776 → 1.426 s, 2.647× | Fixture outputs elementwise equal; not the full request's speedup. |
| E65 withdrawn case | Half-precision long-video reference DP | Example26.364 → 14.719 s | Numerical mismatch, max_abs2.023/RMSE0.04639; not accepted as a valid optimization result. |

![FC2 tuning](figures/e44-fc2-tuning.png)

The E44 figure shows only the original-output-precision whole-block comparison; detailed isolated GEMM sweeps remain in the sanitized report. Split-K and asynchronous LoRA candidates were not promoted just because one microbenchmark improved.

## Overlap evidence

![Independent transfer capture](figures/e52-transfer-overlap.png)

This E52 diagram comes from a separate CUDA-event capture and is not the normal110.617-second performance run. It shows stream intervals including waits; it cannot be read as a kernel-utilization percentage. Weight transfer and computation overlap cannot be summed as independent wall-clock savings. E57 did not collect such events, so its near-zero host `load_remaining` call is not evidence of a measured100% overlap ratio.

## Failures retained in the public record

- E52 initial `linears_fp16` combination: first non-finite value at main block18 FC2; withdrawn. Its fast step time is invalid as successful-generation performance.
- FP16 runtime Larry increment: audio NaNs in a complete run; the measured presets retain FP32 increments.
- E58 720p5-second attempt: CUDA illegal instruction; no successful latency. 720p10-second request not submitted afterward.
- E61/E64 stock-state historical runs and E73 repeated runs: illegal-instruction/Xid errors. The root cause is unresolved. E75's successful cold request is not a sustained-queue reliability certification.
- Some DP-video half-precision cases were fast but mismatched. Default gating remains conservative.

No claim is made that a GPU unlock is needed. It is absent from this release. An exploratory modified-state cold run at222.385 s and the stock E75 run at223.048 s differed by only0.30%; steady sampling10.200/10.205 s was effectively unchanged in those individual observations. These are not statistical equivalence tests.

## Source mapping

E52 implementation `eb7ab3cb2e36ecd023389bc12c7efdaac8a654ee`; E57 `1d1a3ba6074719326f06e3701430a25763f7b90f`; E60 recorded source `bb1fd3c` above canonical implementation `b1c39d2`; E75 H3 source `4369debd37e4f5a426796417fdcdeb5d2cf0e5c7`. These identify the private historical measurement tree; not every historical Git object is shipped in this fresh public history. Redacted reports preserve their implementation/measurement scopes. Public portability changes have their own commits and should not be described as newly measured full-generation speedups.
