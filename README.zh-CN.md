# 双 RTX 2080 Ti + NVLink 的 MiniMax H3 加速方案

[English](README.md) · [安装接入](docs/INSTALL.md) · [模型与下载](docs/MODELS.md) · [开关配置](docs/SETTINGS.md) · [实验结果](docs/RESULTS.md)

给安装Agent：[逐步接入教程与成功标准](docs/AGENT_SETUP.md)、[AGENTS.md](AGENTS.md)。

**专门面向 RTX 2080 Ti / SM75 的 ComfyUI 加速项目，实测环境为两张 22GB 2080 Ti 加 NVLink。**

| 历史同输入 TP 对照 | 优化前：单卡 | 优化后：双卡张量并行 | 加速比 |
|---|---:|---:|---:|
| Qwen 热缓存流式编码，含准备/释放 · E36 | 5.305秒 | 2.570秒 | **2.064倍** |
| Attention 路由+量化+kernel，同一 Kitchen 后端 · E21 | 60.307毫秒 | 30.981毫秒 | **1.947倍** |
| 完整视频 VAE 解码，两边均为 FP32 · E29 | 105.816秒 | 57.684秒 | **1.834倍** |
| 早期完整 Transformer block，INT8/Kitchen50、chunk2048 · E22 | 238.039毫秒 | 176.618毫秒 | **1.348倍** |

以上是**对应范围内的 TP 对照**，不是单卡与双卡完整生成视频的加速比。E36 两边都命中 CPU 打包缓存；E21 为 CUDA event 计时，TP 取较慢卡。E22 的单卡来自本项目同算子实现、权重已驻留，且早于当前 chunk4096/调优FC2/joint/offset 默认档，不能代表当前默认实测。E29 不含模型加载，取第二次解码。Qwen/VAE 输出存在数值差异。[配置复核、具体输入、计时边界和误差](docs/TP-COMPARISON.md)。

叠加其他优化后，已记录最快的 **864×480、5秒视频完整请求为 110.617秒**（E52，热跑，蒸馏8步）。这是整体管线加速后的出片数据，包含额外优化和精度取舍，不能当成上表的纯 TP 收益。

### 张量并行为什么能利用双卡

张量并行把大矩阵权重切成两份，**让两张 2080 Ti 同时计算同一层的不同部分**，再交换所需的激活或部分结果，从而让一次生成利用双卡计算资源。配合分组流式加载和 CPU 缓存，权重分片按阶段进入显存。

**NVLink 提供高速卡间带宽，是这套双卡方案的重要条件。** 本项目通过直接 P2P 拷贝和 CUDA event 交换数据，避免这部分通信绕经 CPU 内存；实测拓扑为 `NV2`。本项目没有做拔掉 NVLink 的对照，其他互联环境未验证。明确支持范围是**双 2080 Ti 22GB / SM75 实测配置**，不能直接推广到普通 11GB 卡或其他显卡组合。

本项目基于 [Comfy-H3-MultiStream](https://github.com/martonsagi/Comfy-H3-MultiStream) 扩展，实验平台是 **Supermicro X9DAi、双路 Xeon E5-2620 v2、256 GB DDR3、双 RTX 2080 Ti 22 GB 和 NVLink**。不需要显卡算力解锁，也不需要修改 ComfyUI 核心。

**默认以速度优先：** 启动器省略 `--config` 时自动选择 `configs/performance.json`，配合附带的性能工作流使用。先安装所需算子，再开启调优 FC2、joint SwiGLU、shared-KV/offset 和 VAE FFN 局部 FP16 累加。RAM 预载单独按需开启；曾失败的主干内部 FP16 输出和 FP16 LoRA 增量不纳入推荐组合。

## 项目特点与新增优化

- **H3 主干 TP2**：INT8 ConvRot 线性层、事件驱动的双卡 P2P、分块激活通信、分组双缓冲权重流式执行。
- **减少重复打包**：canonical 不可变 CPU 权重母版可跨 token 数变化复用；`auto_tokens` 选择有边界的分组和驻留槽，适合切换视频长度后继续使用缓存。
- **加速条件编码**：Qwen H3 GGUF 的 CPU 流式执行/TP2；独立参考编码 DP 节点按完整素材分配两张卡。
- **优化视频 VAE**：FP32/混合/FP16 TP2 精度档，独立 SwiGLU 融合和局部 FFN FP16 累加，符合条件时在 Euler 最后一步预取解码器。
- **SM75 专用内核**：调优 INT8 FC2、主分支与 LoRA 共用 SwiGLU、Sparse Kitchen shared-KV 与 offset 路径，附源码、构建入口和[带校验值的 Linux 预编译包](docs/PREBUILT.md)。
- **可选 RAM 权重常驻**：把指定权重预载到锁定的主机内存；这是独立启动选项，普通节点使用不强制开启。

新增 TP2 与继承的通用 MultiStream 是两条实现路径。本项目速度数据覆盖的是**有直接 P2P 的双 SM75 卡**，不能直接推广到任意卡数与架构。

## 关键速度数据

以下均为 864×480、24fps、Larry v4 step600 EMA、Euler/simple 8步、INT8 ConvRot H3。不同任务和缓存条件分列，**不是同一基线上的累计加速比**。

| 实验 | 输出 | 任务与缓存状态 | 完整请求耗时 | 有效生成帧率 |
|---|---|---|---:|---:|
| E52 | 5秒 / 120帧 | FL2VA，组合配置热跑 | **110.617秒** | **1.085 fps** |
| E60 | 5秒 / 120帧 | FL2VA纯文生，RAM已预载，切换长度后的热回切 | **123.774秒** | **0.970 fps** |
| E60 | 10秒 / 240帧 | FL2VA纯文生，RAM已预载，canonical CPU缓存命中 | **235.156秒** | **1.021 fps** |
| E57 | 15秒 / 360帧 | FL2VA首尾帧条件，RAM预载后的首次请求 | **501.450秒** | **0.718 fps** |
| E75 | 5秒 / 120帧 | Ref2VA双参考，RAM已预载，未解锁系统冷请求 | **223.048秒** | **0.538 fps** |

耗时定义为 ComfyUI `execution_start → execution_success`，含条件编码、采样、VAE、封装保存，不含排队、服务启动和权重预载。多数为单次观测。E52 配对组合对照为 **115.734→110.617秒，1.046倍**，其中包含影响精度的选项，不能称无损加速。详见[实验结果与限制](docs/RESULTS.md)。

有效生成帧率＝保存帧数÷完整请求耗时，不是播放帧率或仅采样阶段的吞吐。关注速度时，**E60 的短队列证据最完整：5秒冷跑→10秒→5秒连续三次成功**，跨长度复用缓存；E52 是这里记录的最快成功 5 秒热跑，E57/E75 是单次成功记录。它们都不能证明长期无人值守队列稳定。[推荐配置、具体开关与稳定性证据](docs/SPEED.md)。

![CPU缓存与阶段耗时](docs/figures/e60-stage-times.png)

### 示例视频

点击预览打开带音轨的视频。已移除容器元数据；私人参考输入与模型权重不随仓库分发。

| 5秒输出 · E52 · 110.617秒 | 10秒输出 · E60 · 235.156秒 | 15秒输出 · E57 · 501.450秒 |
|---|---|---|
| [![5秒](examples/images/e52-480p-5s.jpg)](examples/videos/e52-480p-5s.mp4) | [![10秒](examples/images/e60-480p-10s.jpg)](examples/videos/e60-480p-10s.mp4) | [![15秒](examples/images/e57-480p-15s.jpg)](examples/videos/e57-480p-15s.mp4) |

## 安装与接入

把本仓库安装到 `ComfyUI/custom_nodes/ComfyUI-H3-2080Ti-TP`。**先移除或禁用原版 MultiStream 副本**，两者使用相同节点标识，不能重复安装。

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Luna-AI-Infra/ComfyUI-H3-2080Ti-TP.git
git clone https://github.com/city96/ComfyUI-GGUF.git
git -C ComfyUI-GGUF checkout 373048b8403a7820620065210a691263d4da0a61
python -m pip install -r ComfyUI-GGUF/requirements.txt
cd ComfyUI-H3-2080Ti-TP
# 使用运行 ComfyUI 的 Python 环境。
python scripts/bootstrap_backends.py
python scripts/install_prebuilt.py
python scripts/doctor.py --comfyui ../..
```

以上步骤要求已有正常运行的 **Linux ComfyUI 环境**及其依赖，以及两张支持 P2P 的 CUDA 显卡，不是从空系统开始的一键安装。固定版本的 Sparse Kitchen 已附带 Linux 预编译库；重编译后端或启用本项目额外内核时才需要工具链。已有 GGUF 插件时先检查版本，不要重复克隆。所有命令使用同一个 ComfyUI Python。`doctor.py` 只报告能力，退出码为零不代表已具备运行条件。

按[模型说明](docs/MODELS.md)下载并放置对应权重，然后保持在本插件目录执行：

```bash
# 文生视频/首尾帧任务；参考任务改用 --task ref2va。
python scripts/verify_models.py --comfyui ../.. --task fl2va
python scripts/launch_comfyui.py --comfyui ../.. -- --port 8189
# 另开终端，使用同一 Python，并进入本插件目录：
python scripts/run_workflow.py --url http://127.0.0.1:8189 --workflow workflows/api/t2va-480p-5s.json
```

权重校验会读取约 55 GiB，缺失或哈希不匹配均会失败。工作流检查不会提交视频。默认性能配置需要上述六份算子库；[预编译包、校验值、CUDA12运行库要求和源码构建方法](docs/PREBUILT.md)已提供。`--config configs/conservative.json` 留作显式排查选项，不是推荐速度档；还需要对应调整图里的融合和后端设置。详细步骤见[安装说明](docs/INSTALL.md)。节点注册成功不能代替后端动态库加载及实际执行验证。

典型接线：

```text
H3模型加载 → Larry运行时LoRA → H3 Tensor Parallel → H3 Load Pipeline → 原生采样器
Qwen GGUF加载 → H3 Qwen Execution → 原生H3条件编码 / H3 Reference to Video DP
原生采样器 → H3 TP VAE Decode → 原生音频/视频保存
```

[UI示例](workflows/ui/)用于前端导入，[API示例](workflows/api/)用于 HTTP 提交。两种格式及验证范围见[工作流说明](workflows/README.md)。仅复制工作流不会自动携带进程环境开关。

参考图生视频可直接使用[双参考图 UI 工作流与教程](workflows/REF2VA.md)，附带[两张 CC0 机器人参考图](examples/reference/)及参数一致的 API 图，默认使用 `match` 缩放。已通过前端导入/导出和原生图校验；这组新输入尚无完整生成耗时数据。

实测主程序为 **ComfyUI 0.42.0**，运行时代码等同上游[`46aad33e4d4c0696582611bf450dd93742fa9090`](https://github.com/Comfy-Org/ComfyUI/commit/46aad33e4d4c0696582611bf450dd93742fa9090)，记录中的部署HEAD只增加文档。Agent接入顺序：识别现有Python环境→检查重复节点包→构建选定后端/内核→校验所需权重→启动独立服务→用`scripts/run_workflow.py`检查图→按需加`--submit`提交一条测试。

### 实测权重下载

| 组件 | 对应版本 | 下载 |
|---|---|---|
| FL2VA主干 | 完整未剪枝 INT8 ConvRot | [权重](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors) |
| Ref2VA主干 | 完整未剪枝 INT8 ConvRot，参考任务使用 | [权重](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors) |
| H3 Qwen3-VL-32B | H3专用 GGUF **Q4_K_M**，不是普通聊天模型 | [权重](https://huggingface.co/leejet/MiniMax-H3-GGUF/blob/main/qwen3vl_32b_minimax_h3-Q4_K_M.gguf) |
| 视频VAE | FP16权重文件 | [权重](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_fp16.safetensors) |
| 音频VAE | FP32权重文件 | [权重](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) |
| Larry Turbo LoRA | **v4 step600 EMA**，运行时FP32增量 | [权重](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/blob/main/minimax_h3_turbo_v4_step600_ema.safetensors) |

FL2VA与Ref2VA对应不同任务；权重存储精度、计算精度和累加精度需要分别理解。[模型清单](configs/models.json)提供固定版本URL、放置位置及SHA256，仓库不打包权重。

## 默认值与质量取舍

**推荐默认入口是性能启动器加附带工作流。** 手动新建节点或直接运行 `main.py` 时，源码回退值仍是 Sage、chunk2048、三组两槽、VAE FFN FP32累加及 joint SwiGLU，并不会自动复现推荐速度配置。保守启动配置仅作为显式排查选项，不等于全 FP32 画质参考模式。

性能配置采用 Kitchen50、chunk4096、配套API图中的 canonical packed缓存、调优FC2、joint融合、offset路由及 VAE FFN 局部FP16累加。主干内部输出仍为 `original`，运行时 LoRA 增量仍为 FP32。完整默认值、作用位置和关闭方式见[开关表](docs/SETTINGS.md)。

稀疏注意力、量化、VAE FP16计算、FP16累加和蒸馏8步采样均可能改变结果。局部逐值一致不能证明整片无损。`linears_fp16` 曾在完整采样中出现非有限值，不作为推荐配置。

## 实验环境

| 项目 | 实测配置 |
|---|---|
| 主板 / CPU | Supermicro X9DAi；2×Xeon E5-2620 v2 @ 2.10GHz；共12核24线程 |
| 内存 | DDR3 64GB×4，共256GB；配置速率1066MT/s；双NUMA节点 |
| 显卡 / 互联 | RTX2080Ti 22GB×2；NVLink，拓扑显示NV2 |
| 系统 / 内核 | Ubuntu22.04.5 LTS；Linux5.15.0-191-generic |
| 驱动 | NVIDIA595.91.07；不需要算力解锁 |
| Python / PyTorch | 3.11.16 / 2.11.0+cu130；PyTorch CUDA运行时13.0 |
| CUDA内核编译 | CUDA Toolkit12.8；调优FC2使用CUTLASS3.8.0 |
| ComfyUI | 0.42.0；实测提交`8b219284fd56145c21fa7522c9ebb85125b80d08` |
| 其他依赖 | comfy-kitchen0.2.39；comfy-aimdo0.5.6；gguf0.19.0；前端1.57.1 |

以上是验证环境，不代表任意新版本都兼容。[固定依赖提交](docs/INSTALL.md)和[权重哈希](configs/models.json)随仓库提供。

## 排队与当前限制

支持 ComfyUI 标准队列：多个客户端可提交，单工作线程依次执行；一条 TP2 任务会使用两张卡，不是两条视频同时各用一张卡，也不提供多进程显卡隔离。

3个同时提交的轻量资源控制任务已串行成功。E60曾完成不同长度视频的连续执行，但其他重复H3任务出现过非法指令/Xid；最新未解锁冷请求成功，不代表重任务热跑稳定性已解决。致命GPU错误后应停止处理，不能假定后续排队任务可安全继续。详见[已知问题](docs/KNOWN_ISSUES.md)。

## 实验论文与来源

- [中文实验论文（PDF）](docs/paper/H3-TP-Experiments.zh-CN.pdf)
- [English experiment paper (PDF)](docs/paper/H3-TP-Experiments.en.pdf)
- [详细实验结果](docs/RESULTS.md)、[精选脱敏原始报告](docs/experiments/)
- [上游来源说明](docs/ATTRIBUTION.md)、[LICENSE](LICENSE)、[NOTICE](NOTICE)

公开仓库保留上游来源，但使用新的Git历史，避免带入私人路径和历史个人邮箱。精选报告是脱敏导出，不等同原始日志逐字节副本。模型由各作者按其自身条款提供，本仓库不分发权重。
