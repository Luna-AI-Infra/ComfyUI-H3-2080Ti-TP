# Speed-first profiles / 速度优先配置

## Recommended default / 推荐默认

After [building the required kernels and installing models](INSTALL.md), launch from the plugin directory:

```bash
python scripts/launch_comfyui.py --comfyui ../.. -- --port 8189
```

Omitting `--config` selects the bundled **`configs/performance.json`**, independent of the current working directory. Pair it with the bundled API/UI performance workflows. This is the recommended speed-oriented deployment for the measured dual-SM75/P2P setup, not a claim of a universally optimal configuration or a newly measured complete render.

不带 `--config` 时，启动器默认使用性能配置。它覆盖调优 FC2、joint SwiGLU、shared-KV/offset、VAE FFN 局部 FP16 累加；图中搭配 Sparse Kitchen 50%、chunk4096、canonical CPU 缓存、auto_tokens、Qwen stream_tp2、tp2_fp16/vector 解码、流水预取及 8 步 Euler/simple。主干内部输出保留 original，Larry 增量保留 FP32。直接运行 ComfyUI `main.py` 或手动新建节点不会自动设置这整组配置。

RAM preload is optional: `--config configs/performance-ram-ref2va.json` selects its Ref2VA weight set. It needs substantial spare host RAM and memlock capacity, and can take minutes before the server is ready. It is not a free speed upgrade, and the public RAM launcher has no new full-generation timing. The default avoids forcing those machine-specific requirements. `--config configs/conservative.json` is an explicit troubleshooting alternative; adjust graph-side fusion/attention settings too. It is not the speed recommendation.

RAM 预载不强制默认开启：需要充足内存、memlock 权限和较长预载时间，不能把排除预载后的请求时间当成从启动到出片的总时间。保守配置留作排查，也要配套调整图里的设置。

## Successful configurations and speed / 已成功出片的配置与速度

All rows: measured X9DAi host, dual 2080 Ti 22 GB/NVLink, 864×480 output, 24 fps playback, full INT8 ConvRot weights, Larry v4 step600 EMA with FP32 runtime increments, Euler/simple 8 steps. These are **historical measurements**, not fresh performance tests of the public launcher. Separate tasks/cache histories are not a controlled ranking.

| Use / 用途 | Historical configuration / 历史配置 | Complete time / 完整耗时 | Output speed / 有效速度 | Evidence / 验证范围 |
|---|---|---:|---:|---|
| Fastest recorded warm 5 s / 已记录最快5秒热跑 | E52 FL2VA; three_groups/2 slots, packed cache, chunk4096, tuned FC2, joint fusion, offset, vector VAE, local FP16 FFN accumulation | **110.617 s** | **1.085 fps; 22.12 s/video-s** | One successful combined run; matched baseline 115.734 s, 1.046×. Not a queue test. |
| Repeated text-only 5 s / 重复文生5秒 | E60 RAM-preloaded, canonical CPU cache + auto_tokens; 5 s return after 10 s | **123.774 s** | **0.970 fps; 24.75 s/video-s** | Third successful request in a 5→10→5 s sequence. |
| Text-only 10 s / 文生10秒 | E60 same RAM-preloaded sequence; canonical CPU cache hit at changed token count | **235.156 s** | **1.021 fps; 23.52 s/video-s** | Second successful request; first cold 5 s was 216.657 s. |
| First/last-frame 15 s / 首尾帧15秒 | E57 FL2VA, staged pipeline and RAM preload; first request | **501.450 s** | **0.718 fps; 33.43 s/video-s** | One success; **937.343 s RAM preload excluded**; no second hot run measured. |
| Two-reference 5 s / 双参考图5秒 | E75 Ref2VA, RAM-preloaded, stock GPU state, cold request | **223.048 s** | **0.538 fps; 44.61 s/video-s** | One success; steady 50-block sampling-step median 10.205 s. Not repeated-heavy-job certification. |

Effective output fps = saved frames / complete request seconds. Seconds per video-second = complete time / saved duration; **lower is faster**. Complete time is `execution_start → execution_success`, including conditioning, sampler, VAEs and saving; it excludes queueing, process startup and RAM preload. These throughput numbers are derived from the recorded durations, not new measurements or playback rates.

E60 also used RAM preload (1016.673 s initially, 292.848 s on a later file-cache-assisted preload); E75 preload took 844.664 s. Those setup intervals are excluded from the table. A cold request is not a cold machine or a measurement from server launch. The no-RAM default therefore must not be promised those exact historical times.

有效速度按完整请求计算，包含条件编码、采样、VAE 和保存，不是单纯采样速度；播放帧率始终是 24fps。E60/E57/E75 均用了 RAM 预载，表中未包含这部分启动成本；不带 RAM 的默认档不能承诺完全相同耗时。各行任务、缓存和历史版本不同，不能直接得出某一开关的独立加速比。原始配置与证据见[实验结果](RESULTS.md)、[E52](experiments/E52.md)、[E60](experiments/E60.md)、[E57](experiments/E57.md)、[E75记录](experiments/E75.md)。公开机器人参考图是新接入示例，不能直接套用 E75 的耗时。

## What is reasonably established / 稳定性证据怎么理解

- **E60 has the strongest short-sequence evidence here:** three full requests succeeded while changing length, with canonical packing reused. Preparation fell from 62.234 s cold to 0.504/0.506 s on cache hits. Reuse the same server for that cache benefit; restarting drops process caches.
- **E52, E57 and E75 establish successful individual outputs.** They do not establish repeatability distributions or hours of service. E52's local precision changes are part of its result.
- **Long unattended heavy queues remain unresolved.** Historical stock-state and later repeated runs produced illegal-instruction/Xid errors. Stop after a fatal CUDA error; do not continue or auto-retry the next GPU task. See [known issues](KNOWN_ISSUES.md).

对偏重速度的使用方式，优先保持同一服务进程、复用权重缓存，并从 480p、5 秒、match 参考图开始。E60 只证明过一次三任务序列成功，不能把“有连续成功记录”改写成“长期稳定”。参考图数量、缩放模式、输出长度及缓存状态都应随耗时一起记录。

## Accepted tradeoffs / 已采用的取舍

The default includes quantization, 50% sparse routing, FP16 VAE computation, local half accumulation and distilled 8-step sampling. They can affect outputs. E51's isolated FFN relative RMSE was about 0.42%; this is not a full-video quality bound. Failed internal-FP16-output and FP16 runtime-LoRA candidates remain excluded. Keep `allow_unvalidated_video=false`. [All switches and rollback methods](SETTINGS.md).

默认档以速度为先，但仅纳入已有验证依据的实现。局部 FFN 误差不等于最终视频误差；未通过验证的“更快”候选不默认开启。若需要精度消融，请单独改对应开关并固定输入、seed、尺寸、帧数和其他配置。
