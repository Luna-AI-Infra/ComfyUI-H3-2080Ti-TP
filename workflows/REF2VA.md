# Reference-image workflow / 参考图工作流

## English

Use [the editor workflow](ui/ref2va-robot-480p-5s.json) in the ComfyUI frontend. The matching [API workflow](api/ref2va-robot-480p-5s.json) is for HTTP clients. Both use the full **Ref2VA INT8 ConvRot** model, H3 Qwen **Q4_K_M**, the video/audio VAEs and Larry **v4 step600 EMA** listed in [the model guide](../docs/MODELS.md). Do not substitute the FL2VA diffusion model.

1. Complete [installation](../docs/INSTALL.md), including Sparse Kitchen and the native libraries required by `configs/performance.json`. These graphs use vector VAE fusion and the performance graph settings; the conservative process preset alone does not disable graph-side fusion.
2. From the plugin directory, run `python scripts/verify_models.py --comfyui ../.. --task ref2va`.
3. Copy both [public reference PNGs](../examples/reference/) into `ComfyUI/input/`, keeping the filenames `reference-a.png` and `reference-b.png`. Alternatively, upload your own images in the two **Load Image** nodes and select the returned filenames. The repository's example directory is not searched automatically by ComfyUI.
4. Start the server with `python scripts/launch_comfyui.py --comfyui ../.. -- --port 8189`. Performance is the default; the explicit `--config configs/performance.json` remains equivalent. Use the same Python environment as ComfyUI and an available port.
5. Drag/open `workflows/ui/ref2va-robot-480p-5s.json` in the frontend. Confirm the Ref2VA model, both reference previews, GPU1 as the second GPU, and that no nodes are missing. The two image links go into `ref_image_0` and `ref_image_1` on **H3 Reference to Video (DP encoders)**.
6. The prompt's `<Picture 1>` means the first connected reference; `<Picture 2>` means the second. Replace the prompt when replacing the subject. Change **RandomNoise → noise_seed** for a new sample, keeping it fixed for comparisons.
7. Queue one job when ready. The intended saved output is **864 × 480, 24 fps, 5 seconds** (120 frames after trimming), with 8 Euler/simple steps. The latent-length input is 124, not the saved frame count. Output prefix: `H3-TP/ref2va-robot-480p-5s`.

For an API client, after placing the inputs and starting the server:

```bash
python scripts/run_workflow.py --url http://127.0.0.1:8189 --workflow workflows/api/ref2va-robot-480p-5s.json
# Add --submit to generate once; the command above only checks the live schema.
```

### Reference size and quality controls

- `ref_image_size=match` is the default in this example. Each image is scaled down with its aspect ratio preserved to approximately the output's pixel area, then aligned to 32 pixels. No crop is applied; small images are not deliberately upscaled, apart from alignment. These 768-square examples become **640 × 640** before reference encoding for an 864 × 480 output.
- `max` preserves more reference resolution, capped at a 2048-pixel short edge for large images. It can increase reference tokens, memory use and sampling time. It is not a guaranteed quality improvement.
- `encode_mode=auto` permits the reference encoder's eligible dual-GPU scheduling. `allow_unvalidated_video=false` stays disabled. The example uses images, not reference videos/audio.
- Pipeline and decode both use `tp2_fp16`; decode fusion is `vector`. Sparse Kitchen keeps 50% of eligible target-video attention tiles. Precision, sparsity, LoRA and sampling settings are described in [SETTINGS.md](../docs/SETTINGS.md).

### Validation scope

The graph imported into frontend 1.57.1 on ComfyUI 0.42.0; the two dynamic reference connections survived API export. Exported node inputs matched the API example, and both passed ComfyUI's native graph validator with the bundled PNGs. See [validation evidence](../docs/validation/ref2va-workflow.json). This does not establish a successful full render, output quality, a speed measurement, or long-queue stability for this new example. Historical example videos and timings used different inputs. Follow the resource monitoring and fatal-error handling in [AGENT_SETUP.md](../docs/AGENT_SETUP.md).

## 中文

前端导入 [UI 工作流](ui/ref2va-robot-480p-5s.json)；HTTP 客户端使用对应的 [API 工作流](api/ref2va-robot-480p-5s.json)。两者配置一致。

1. 按[安装说明](../docs/INSTALL.md)安装插件、Sparse Kitchen 和性能配置所需内核，并准备[模型说明](../docs/MODELS.md)中的 **Ref2VA INT8 ConvRot** 主干、H3 Qwen Q4_K_M、两个 VAE 及 Larry v4 step600 EMA。不要选成 FL2VA 主干。
2. 在插件目录执行 `python scripts/verify_models.py --comfyui ../.. --task ref2va`，确认权重齐全且哈希一致。
3. 把[两张公开参考图](../examples/reference/)复制到 `ComfyUI/input/`，保留原文件名。也可以在两个 **Load Image** 节点上传自己的图片，并重新选择文件。ComfyUI 不会自动去插件的 examples 目录查找图片。
4. 使用同一个 ComfyUI Python，在插件目录执行 `python scripts/launch_comfyui.py --comfyui ../.. --config configs/performance.json -- --port 8189`。确认端口空闲。图中启用了 vector VAE 融合，只改为保守启动配置不会自动关闭图里的融合选项。
5. 在前端拖入 UI JSON，确认没有缺失节点、两个图片预览正确、主模型为 Ref2VA、第二张显卡编号为 1。参考图节点内部会自动缩放，不需要另接 Resize 节点。
6. 提示词中的 `<Picture 1>` 和 `<Picture 2>` 对应连接顺序。换人物或物体时一并改提示词。修改 RandomNoise 的 seed 可生成不同样本；做对比时保持固定。
7. 准备好后提交一条任务。预期保存规格为 **864×480、24fps、5秒、120帧**，8 步 Euler/simple。节点中的 length=124 是生成潜变量长度，保存前会裁剪到 5 秒。

默认 `match` 保持宽高比、按生成画面的像素面积缩小，并对齐到 32 的倍数。这两张 768×768 示例图会变成 **640×640** 再编码，不会被强制拉伸成 864×480。`max` 会保留更多分辨率，可能增加 token、显存及每步耗时，不保证画质一定更好。图中编码为 `auto`，参考视频实验选项保持关闭；其他精度、稀疏和采样开关见[设置说明](../docs/SETTINGS.md)。

已验证前端导入、双参考图动态接线、导出 API 参数一致及原生图校验。**这组新图片尚未验证完整成片、画质和耗时**，不能直接套用 README 的历史速度数据；也不代表长期排队稳定性已验证。
