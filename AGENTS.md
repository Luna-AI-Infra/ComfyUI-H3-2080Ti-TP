# Working with this repository

This is a ComfyUI custom-node pack, not a ComfyUI core fork or a standalone video server.

## Setup tasks

1. Read `docs/AGENT_SETUP.md` and use the user's existing ComfyUI Python environment.
2. Inspect `configs/models.json`, `docs/INSTALL.md` and `docs/SETTINGS.md` before selecting a preset.
3. Do not install alongside the original MultiStream pack: node identifiers overlap.
4. Do not infer that a workflow JSON includes process-level environment options.
5. Keep model weights, user reference media, tokens, machine paths and generated logs out of Git.

## Validation and claims

- `python scripts/check_release.py` checks public package structure, links and basic privacy patterns without CUDA.
- `python scripts/doctor.py` reports optional kernel/backend capabilities.
- `python tests/smoke_kernels.py` uses SM75 CUDA and needs the native libraries built first. It is not an end-to-end generation test.
- `python scripts/run_workflow.py --url URL --workflow FILE` checks live node registration/schema without submitting; add `--submit` only for a requested generation test.
- Use the standard serial ComfyUI queue; concurrent rendering on shared GPUs is not established.
- Stop testing on CUDA illegal instructions/Xid, overheating or sustained power-limit violations. Do not auto-retry a failed GPU workload.
- Keep performance-only scheduling changes separate from approximation/precision changes. Record parent/implementation commits and rollback, and never claim local parity proves a lossless full video.
- Update both READMEs and the experiment paper source when changing published measurements or configuration semantics.

No GPU-unlock mechanism is part of this repository. Do not introduce administrator deployment helpers, private paths or historical private Git author metadata.
