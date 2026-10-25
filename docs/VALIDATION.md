# Public export validation

## RTX 2080 Ti identity and prebuilt release (E80)

The repository is now `ComfyUI-H3-2080Ti-TP`. README comparisons distinguish E21 attention, E22 early whole-block, E29 VAE and E36 warm Qwen timings. E22's original normal JSONs share commit, input, backend, keep fraction, chunk and timing repetitions; the 1.348× result predates the current speed preset. See [configuration audit](TP-COMPARISON.md).

Six no-lineinfo Linux x86_64/SM75 libraries were built with CUDA12.8 (FC2 with pinned CUTLASS3.8), debug-stripped and scanned for private build markers. The final build passed ten small checks on both RTX2080Ti cards under the resource guard. A clean staging installation using the release archive passed its SHA256/source checks and loaded all six ELF libraries. A CPU regression checks rejection of a tampered archive. This is not a new performance result or a complete generation test. [Build provenance](validation/prebuilt-build.json), [small GPU checks](validation/prebuilt-smoke.json), [prebuilt installation](PREBUILT.md).

## Installation audit (2026-10-25)

The README procedure was reviewed as a sequence of shell commands. Fixes add the pinned GGUF dependency installation, keep the working directory consistent, require a selected model task, and provide explicit startup/graph-check commands in both languages. Missing required weights now fail verification instead of returning success. The RAM launcher now sets ComfyUI arguments before imports that parse them.

The speed-first follow-up makes `--config` optional and selects the installed `configs/performance.json` by default. A CPU regression checks the complete preset environment/ComfyUI argument handoff without loading real Torch or submitting a GPU job; all three setup regression methods passed. The speed table derives effective fps from the existing measurements and does not add new generation results. Both rebuilt papers retain eight pages; page renders, text bounds and Unicode extraction were checked.

CPU regressions (`python -m unittest discover -s tests -p test_setup.py -v`) passed for both missing-weight task variants and the RAM-loader argument handoff. On the measured Linux host, the pinned backend's shipped library loaded through ctypes and reported ABI 4; `ldd` found its shared dependencies. It is a prebuilt C-ABI library, not an automatic first-use extension build. No model inference or fresh-machine environment installation was performed during this audit. The prior GPU/package checks below retain their original scope.

The export starts from historical H3 runtime `4369debd37e4f5a426796417fdcdeb5d2cf0e5c7`. Public code packaging commit `5c2723d` relocates native-library lookup and includes original kernel sources plus portable helpers. Follow-up `de32c57` adds pinned archive bootstrap and optional-backend guidance. Neither is a newly measured complete-generation speedup.

| Check | Result | Evidence |
|---|---|---|
| Six native C-ABI libraries | Rebuilt with CUDA12.8 and the tested CUTLASS headers | [Build/source hashes](validation/kernel-builds.json) |
| Small dual-SM75 kernel smoke | 10 cases passed; tested integer FC2 tensors exactly equal; selected activation outputs equal; local half-accumulation outputs finite | [Kernel checks](validation/kernel-smoke.json) |
| Pinned Sparse Kitchen source | Archive bootstrap/import passed; selected backend file matches the historical pinned revision | [Backend check](validation/backend-check.json) |
| Public package node registration | Passed in isolated ComfyUI0.42.0 runtime | [Integration](validation/comfyui-integration.json) |
| Three public API examples | Native validation passed; reference example validated with supplied local inputs replacing placeholders | [Integration](validation/comfyui-integration.json) |
| Two UI examples | Current H3 widget order/count/types/enums passed after updating VAE fusion and TP planner fields | [UI schemas](validation/ui-schemas.json) |
| Public robot Ref2VA UI/API example | Frontend import/export preserved both dynamic reference inputs; API inputs match; both native graph validations passed with bundled CC0 PNGs; no generation submitted | [Ref2VA check](validation/ref2va-workflow.json) |
| Serial queue smoke | Three concurrently submitted lightweight control tasks passed, execution intervals nonoverlapping | [Integration](validation/comfyui-integration.json) |
| Three remuxed example videos | Full video/audio decode;120/240/360 video frames; private container metadata removed | [Media manifest](../examples/media-manifest.json) |
| Bilingual papers | Eight pages each; rendered/visually reviewed; text bounds and Unicode extraction checked | [PDF checks](validation/pdf-qa.json) |

The first Git transport attempt for the external Sparse Kitchen repository failed TLS on the test host. The pinned archive path succeeded without disabling certificate validation. This is not a GPU failure or a substitution of another backend revision.

These checks do **not** validate a complete render with the relocated public RAM launcher or the new robot reference inputs, a fresh-machine installation, every frontend interaction, all optional attention variants, perceptual equivalence, or unattended repeated-heavy-job reliability. The historical end-to-end measurements retain their own code/configuration scopes.

No GPU unlock was enabled. The temporary review server was isolated from production and stopped afterward. This repository ships neither the test machine's paths/SSH helpers nor private raw logs/input references. Public commits use the Luna-AI-Infra GitHub noreply author/committer identity; no private Git history was imported.
