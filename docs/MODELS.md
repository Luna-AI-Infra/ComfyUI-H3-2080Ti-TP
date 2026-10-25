# Model variants, precision and verified downloads

中文要点：实验使用完整、未剪枝的H3 FL2VA/Ref2VA INT8 ConvRot主干；文本编码器是H3专用Qwen3-VL-32B的Q4_K_M GGUF，不是普通聊天Qwen，也不是把整个H3模型都改为4位。视频VAE文件为FP16，音频VAE为FP32，参考编码与视频解码的实际计算精度另由节点/启动设置决定。

## Exact tested assets

These are MiniMax **H3** checkpoints, not Hailuo API endpoints or MiniMax language-model checkpoints. FL2VA and Ref2VA are different task variants; select the model that matches the conditioning graph. The benchmarked diffusion checkpoints retain the complete model/AdaLN structure and are **not pruned**, FP8, NVFP4 or mixed INT4/INT8 replacements.

| Role | Tested file / format | Approx. download | Put under ComfyUI |
|---|---|---:|---|
| FL2VA diffusion model | `minimax_h3_fl2va_int8_convrot.safetensors` | 34.04 GB | `models/diffusion_models` |
| Ref2VA diffusion model | `minimax_h3_ref2va_int8_convrot.safetensors` | 34.04 GB | `models/diffusion_models` |
| H3 text/vision encoder | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 18.22 GB | `models/text_encoders` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB | `models/vae` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB | `models/vae` |
| Runtime Turbo LoRA | `minimax_h3_turbo_v4_step600_ema.safetensors` | 0.78 GB | `models/loras` |

Download entry points:

- [Comfy-Org H3 diffusion models](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models): full FL2VA / Ref2VA INT8 ConvRot.
- [leejet H3 Q4_K_M encoder](https://huggingface.co/leejet/MiniMax-H3-GGUF/blob/main/qwen3vl_32b_minimax_h3-Q4_K_M.gguf); [byte-identical Unsloth mirror](https://huggingface.co/unsloth/MiniMax-H3-GGUF/blob/main/qwen3vl_32b_minimax_h3-Q4_K_M.gguf).
- [Comfy-Org H3 VAEs](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae).
- [Larry v4 step600 EMA LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/blob/main/minimax_h3_turbo_v4_step600_ema.safetensors).

`configs/models.json` contains direct **revision-pinned** download URLs, exact bytes, destinations and SHA256. Published source metadata was checked against the tested weight hashes; model weights are not downloaded by the installer or redistributed in this repository.

| Asset | Tested SHA256 |
|---|---|
| FL2VA INT8 | `7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5` |
| Ref2VA INT8 | `9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9` |
| Qwen Q4_K_M | `11e6efe70a57ce7f4838c47bdbd1a1c4b8ce10e2b7747f1b065990b70f4b05fc` |
| Video VAE | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| Audio VAE | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| LoRA | `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3` |

## Storage precision is not accumulation precision

- Main INT8 GEMMs multiply integer activations/weights and accumulate to **INT32**, then rescale to floating-point outputs. FP32 residuals/output tensors do not make them FP16×FP16→FP32 Tensor Core GEMMs.
- The H3 Qwen encoder is a 50-layer H3-specific GGUF variant. Q4_K_M describes its quantized storage, not all operations or the video model's weights.
- The VAE checkpoint is FP16. `tp2_fp16` selects a decoder policy; the separate `H3_VAE_FFN_ACCUM` switch controls local FC1/FC2 FP16 accumulation. Default accumulation is FP32, while the performance preset selects FP16. FC2 in that FP16 branch also rounds its output to half before casting to FP32.
- Reference encoding used `--fp32-vae` in E65/E75. It is distinct from TP2 FP16 **decoding** and should not be described as FP16 reference encoding.
- The Larry checkpoint stores BF16 tensors, while the benchmarked runtime LoRA increment is FP32 with strength 1. The original FP16 runtime increment produced audio NaNs in a full run; it is not recommended.

Downloads and model usage are subject to their authors' terms. See the upstream model cards and [NOTICE](../NOTICE); this plugin's code license does not license model weights.
