# Agent-readable integration procedure

Follow these steps in order. Ask for a ComfyUI root or Python environment only when they cannot be identified from the user's supplied workspace. Do not change an unrelated running service or replace its PyTorch installation.

## Inputs to identify

- The ComfyUI checkout and the Python executable that launches it.
- Which two CUDA devices the user intends to use; this TP2 implementation uses device0/1 and direct P2P.
- Desired task: FL2VA/text-only/first-last-frame versus Ref2VA references.
- Whether model assets already exist, and whether optional performance kernels/RAM preload are wanted.

## Step 1: identify versions and duplicate nodes

The measured application is ComfyUI0.42.0, runtime-equivalent to upstream commit `46aad33e4d4c0696582611bf450dd93742fa9090` (the recorded deployment HEAD adds documentation only). The complete tested environment is in the README. Other releases are not automatically validated.

Use the existing ComfyUI Python. Check `comfyui_version.py`, `git rev-parse HEAD`, and the plugin directories. Only one of this pack/upstream MultiStream may be enabled. Install this pack under `custom_nodes/ComfyUI-H3-TP` and the pinned GGUF plugin described in INSTALL.md.

**Success condition:** one copy of each custom-node package, known ComfyUI/Python/PyTorch versions, intended GPUs visible. Avoid globally changing CUDA_VISIBLE_DEVICES for unrelated services.

## Step 2: check capabilities before loading a model

From this plugin's directory, with the ComfyUI Python:

```bash
python scripts/doctor.py --comfyui ../..
```

For TP2, require two compatible devices and `p2p_0_1=true`. For Sparse Kitchen run `python scripts/bootstrap_backends.py`. Build native libraries as described in INSTALL.md; use a CUDA toolkit supporting SM75 and the tested CUTLASS headers. Missing optional dependencies are not evidence that an optimization is active.

**Success condition:** selected backend installed, selected native libraries present, compatible CUDA devices/P2P. Use a deliberate alternative backend rather than silently changing quality settings.

## Step 3: resolve exact model variants

Read `configs/models.json`. Download or locate the required assets in the stated ComfyUI model folders. Do not substitute a generic Qwen chat checkpoint, a pruned diffusion checkpoint, FL2VA for Ref2VA, or an unrelated Turbo LoRA. Do not commit the assets.

```bash
python scripts/verify_models.py --comfyui ../.. --task fl2va
```

Select `--task ref2va` for reference tasks. The tool requires all five assets for the chosen bundled example and fails on missing files or hash mismatches; only the other diffusion-model variant is skipped. Ref2VA also needs user-provided references; example filenames are placeholders. Preserve private input media outside Git.

**Success condition:** exact required filenames/hashes and correct task variant. Record user-requested substitutions as unvalidated changes.

## Step 4: launch a separate or authorized server

Check whether the intended port is already in use. To create an isolated test server without changing other instances:

```bash
python scripts/launch_comfyui.py --comfyui ../.. --config configs/conservative.json -- --port 8189
```

Use `configs/performance.json` only after corresponding backends/libraries are ready. For the exact graph-side performance settings use the API examples. Source defaults and process presets differ. RAM preload is optional and requires additional host memory/memlock capacity; no privilege elevation is performed by this launcher.

**Success condition:** startup log has no import failures, `/object_info` contains the graph's H3 and GGUF nodes, `/queue` is reachable. Do not mistake node registration for kernel execution.

## Step 5: validate a graph before submission

Editor JSON is under `workflows/ui`; API JSON is under `workflows/api`. For text-only API validation:

```bash
python scripts/run_workflow.py --url http://127.0.0.1:8189 --workflow workflows/api/t2va-480p-5s.json
```

This checks live registration, required input presence and obvious enum selections without submitting. It does not replace ComfyUI's complete graph validator. For editor graphs, load them in the frontend and inspect widgets/links/model selections; the API JSON is not an editor canvas.

**Success condition:** no missing node types/required inputs; matching staged-pipeline/decode precision/generation; correct filenames and available reference inputs.

## Step 6: one requested end-to-end test, then queue regression

For a requested generation test:

```bash
python scripts/run_workflow.py --url http://127.0.0.1:8189 --workflow workflows/api/t2va-480p-5s.json --submit
```

This submits once and waits for history, failing on execution error without retry. Monitor resources separately: experiments stopped at75°C, or power above110% of the configured cap for10 continuous seconds, or any CUDA/Xid/monitor error. Do not claim the submission helper implements that monitor.

Validate video and audio decode, output dimensions/frame count, and history success. Record execution_start→execution_success separately from queue/startup/preload. Test multiple real jobs only after the first passes; repeat with cache reuse and explicitly test any intended model/length changes. Current repeated-heavy-job stability is unresolved.

**Success condition:** complete audio/video output and no new GPU errors, with the exact configuration recorded. A single success does not certify a long queue. After a fatal CUDA error stop the queue; a new Python process is normally required for a fresh context.
