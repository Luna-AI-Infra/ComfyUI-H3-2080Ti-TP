# Adapted from installed ComfyUI comfy_extras/nodes_minimax_h3.py.
# E65 preserves reference preprocessing/presentation; only encoding scheduling changes.
import math
import comfy.audio
import nodes
import node_helpers
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import _empty_av_latent, _resize, adapt_canvas, CANVAS_MULTIPLE, REF_IMAGE_SHORT_EDGE, FPS
from .ref_encode_dp import encode_jobs

class H3ReferenceToVideoDP(io.ComfyNode):
    """ref2va: prompt + reference images / videos / audio -> conditioning + AV latent.

    References enter the presentation in fixed order: images, then videos (each
    soundtrack's <Audio j> label right before its <Video k>), then standalone
    audio. Ordinals are 1-based per type, so the prompt refers to them as
    <Picture i> / <Video k> / <Audio j>.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ReferenceToVideoDP",
            description="<Picture i> / <Video k> / <Audio j> reference conditioning for MiniMax H3. Use the same tags when prompting.",
            display_name="H3 Reference to Video (DP encoders)",
            category="model/conditioning/minimax",
            inputs=[
                io.Combo.Input("encode_mode", options=["auto", "single", "dp2"], default="auto"),
                io.Boolean.Input("allow_unvalidated_video", default=False, tooltip="Experimental: low-precision video DP showed latent differences. FP32 video DP is validated. Keep disabled for normal generation."),
                io.Clip.Input("clip"),
                io.Vae.Input("vae", optional=True, tooltip="Video VAE. Without it reference images/videos only condition the text encoder."),
                io.Vae.Input("audio_vae", optional=True, tooltip="Audio VAE. Without it reference audio only conditions the text encoder."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, (124 = ~5s, trained range is ~124-362)"),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                    tooltip="Reference image sizing. 'match' scales each ref (down only, keeping aspect) to the generation's pixel area; 'max' uses the reference pipeline's 2048px short edge for best identity fidelity. Reference tokens ride through every sampling step, so 'max' can be several times slower."),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference image (downscaled to 2048 short edge if larger, never upscaled)"),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps (2-15s)"),
                        prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio", tooltip="Soundtrack of the same-numbered reference video"),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", tooltip="Standalone reference audio"),
                        prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, prompt, width, height, length, ref_image_size="match", vae=None, audio_vae=None,
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None, encode_mode="auto", allow_unvalidated_video=False) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        jobs = []
        def enqueue(encoder, value):
            index = len(jobs)
            jobs.append((encoder, value))
            return index
        def enqueue_audio(encoder, audio):
            waveform = audio["waveform"]
            sr = audio["sample_rate"]
            target = getattr(encoder, "audio_sample_rate", 32000)
            if sr != target:
                waveform = comfy.audio.resample(waveform, sr, target)
            return enqueue(encoder, waveform[:1].movedim(1, -1)), 1

        ref_items = []   # for the tokenizer presentation, in request order
        ref_blocks = []  # for the DiT payload, same order

        for img in (ref_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                # aspect-preserving scale (down only) to the generation's pixel area
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            ref_items.append({"type": "image", "data": resized})
            if vae is not None:
                z = enqueue(vae, resized)
                ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

        ref_video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            # index-paired soundtrack: ref_video_audio_N belongs to ref_video_N
            soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            if soundtrack is not None:
                # the soundtrack gets its own <Audio j> label, emitted before <Video k>
                ref_items.append({"type": "audio"})
            # Qwen sees the video at 2 fps with timestamps
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({"type": "video", "data": qwen_frames,
                              "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            if vae is None:
                continue
            z = enqueue(vae, frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None and audio_vae is not None:
                audio_latent, ref_audio_t = enqueue_audio(audio_vae, soundtrack)
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": None, "latent_h": ch // 16, "latent_w": cw // 16,
                               "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            ref_items.append({"type": "audio"})
            if audio_vae is not None:
                audio_latent, ref_audio_t = enqueue_audio(audio_vae, audio)
                ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        outputs = encode_jobs(jobs, encode_mode, allow_unvalidated_video)
        for block in ref_blocks:
            if "latent" in block:
                block["latent"] = outputs[block["latent"]]
                if "latent_t" in block:
                    block["latent_t"] = block["latent"].shape[2]
            if block.get("audio_latent") is not None:
                block["audio_latent"] = outputs[block["audio_latent"]]
                block["ref_audio_t"] = block["audio_latent"].shape[-1]
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return io.NodeOutput(cond, latent)

