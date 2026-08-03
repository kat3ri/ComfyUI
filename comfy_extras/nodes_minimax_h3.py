"""MiniMax H3 nodes: AV latent creation and task conditioning (t2va / fl2va / ref2va).

The H3 packed-DiT consumes, via conditioning:
- Qwen3-VL-32B hidden states with per-token modality tags (from the minimax CLIP)
- keyframe / reference condition latents, re-injected every step (never denoised)

Latents are NestedTensor pairs (video [B,24,T,H/16,W/16], audio [B,32,2,T40]);
sampling runs on the flat pack with any stock sampler (the model handles the
audio stream's shifted schedule internally).
"""

import math

import torch
import torchaudio

import nodes
import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy_api.latest import ComfyExtension, io

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def adapt_canvas(width, height):
    """768-short-edge canvas with 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


class EmptyMiniMaxH3LatentAV(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyMiniMaxH3LatentAV",
            display_name="Empty MiniMax H3 AV Latent",
            category="model/latent/minimax",
            description="Joint video+audio latent for MiniMax H3. Duration snaps to the model's 17k+5 frame grid at 24 fps.",
            inputs=[
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested)"),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, width, height, length) -> io.NodeOutput:
        latent, _ = _empty_av_latent(width, height, length)
        return io.NodeOutput(latent)


class MiniMaxH3ImageToVideo(io.ComfyNode):
    """t2va and fl2va: prompt (+ optional first/last keyframes) -> conditioning + AV latent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ImageToVideo",
            display_name="MiniMax H3 Image to Video",
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested)"),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, length,
                first_frame=None, last_frame=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        images = []
        keyframes = []
        if first_frame is not None:
            # geometry anchor: plain stretch to canvas
            img = _resize(first_frame[:1], width, height, "disabled")
            images.append(img)
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if last_frame is not None:
            # follower: aspect-preserving cover-crop
            img = _resize(last_frame[:1], width, height, "center")
            images.append(img)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

        tokens = clip.tokenize(prompt, images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)

        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": frame_count,
            })
        return io.NodeOutput(cond, latent)


def _encode_ref_audio(audio_vae, audio):
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]


def _build_ref_blocks(vae, audio_vae, width, height, frame_count, ref_image_size,
                      ref_images, ref_videos, ref_video_audios, ref_audios):
    """Shared ref2va block builder: reference images / videos / audio -> (ref_items, ref_blocks).

    ref_items feed the tokenizer's <Picture i>/<Video k>/<Audio j> presentation;
    ref_blocks feed the DiT payload, in the same order.
    """
    ref_items = []
    ref_blocks = []

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
        z = vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
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
        z = vae.encode(frames)
        audio_latent, ref_audio_t = (None, 0)
        if soundtrack is not None:
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
            # the soundtrack gets its own <Audio j> label, emitted before <Video k>
            ref_items.append({"type": "audio"})
        # Qwen sees the video at 2 fps with timestamps
        sample_idx = list(range(0, frames.shape[0], FPS // 2))
        qwen_frames = frames[sample_idx]
        ref_items.append({"type": "video", "data": qwen_frames,
                          "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
        ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                           "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                           "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

    for audio in (ref_audios or {}).values():
        if audio is None:
            continue
        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

    return ref_items, ref_blocks


_REF_INPUTS = [
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
]


class MiniMaxH3ReferenceToVideo(io.ComfyNode):
    """ref2va: prompt + reference images / videos / audio -> conditioning + AV latent.

    References enter the presentation in fixed order: images, then videos (each
    soundtrack's <Audio j> label right before its <Video k>), then standalone
    audio. Ordinals are 1-based per type, so the prompt refers to them as
    <Picture i> / <Video k> / <Audio j>.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceToVideo",
            description="<Picture i> / <Video k> / <Audio j> reference conditioning for MiniMax H3. Use the same tags when prompting.",
            display_name="MiniMax H3 Reference to Video",
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, (124 = ~5s, trained range is ~124-362)"),
                *_REF_INPUTS,
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length, ref_image_size="match",
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items, ref_blocks = _build_ref_blocks(vae, audio_vae, width, height, frame_count, ref_image_size,
                                                  ref_images, ref_videos, ref_video_audios, ref_audios)

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return io.NodeOutput(cond, latent)


class MiniMaxH3VideoExtend(io.ComfyNode):
    """Continue a prior MiniMax H3 clip from its trailing latent frames (chunked
    video extension), optionally alongside character/background references.

    The trailing frames are taken directly from the prior generation's own AV
    latent (no VAE re-encode, which would otherwise hit the causal VAE's
    front-padding as if the tail were a fresh sequence start). They ride in as
    clean "cond" rows placed at negative time immediately before this call's own
    frame 0 -- the same never-denoised treatment MiniMaxH3ImageToVideo's
    first/last-frame keyframes get -- so the new clip picks up where the old one
    left off instead of just being told what an anchor frame should look like.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VideoExtend",
            display_name="MiniMax H3 Video Extend",
            category="model/conditioning/minimax",
            description="Continue a prior MiniMax H3 clip from its trailing latent frames. Width/height are inherited from context_latent.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae", optional=True),
                io.Latent.Input("context_latent", tooltip="AV latent output from a prior MiniMax H3 generation to continue from"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="New frame count at 24 fps for the continuation only (excludes context_frames)"),
                io.Int.Input("context_frames", default=2, min=1, max=8, tooltip="Trailing latent frames of context_latent carried over as context (1 latent frame covers 1-4 pixel frames)"),
                io.Float.Input("context_strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="1.0 = context frames stay clean/hard-pinned; lower blends in noise for a softer handoff"),
                *_REF_INPUTS,
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, context_latent, prompt, length, context_frames=2, context_strength=1.0,
                audio_vae=None, ref_image_size="match",
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None) -> io.NodeOutput:
        ctx_video, _ctx_audio = context_latent["samples"].tensors
        if ctx_video.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        ctx_t, ctx_h, ctx_w = ctx_video.shape[2], ctx_video.shape[3], ctx_video.shape[4]
        n_frames = min(context_frames, ctx_t)
        width, height = ctx_w * 16, ctx_h * 16  # inherit the source clip's canvas exactly

        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items, ref_blocks = ([], [])
        if any((ref_images, ref_videos, ref_audios)):
            if audio_vae is None and (ref_video_audios or ref_audios):
                raise ValueError("audio_vae is required when ref_video_audios or ref_audios are supplied")
            ref_items, ref_blocks = _build_ref_blocks(vae, audio_vae, width, height, frame_count, ref_image_size,
                                                      ref_images, ref_videos, ref_video_audios, ref_audios)

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        context_kf = {"kind": "context", "num_frames": n_frames,
                     "latent": ctx_video[:, :, ctx_t - n_frames:, :, :]}
        values = {"minimax_keyframes": [context_kf], "minimax_frame_count": frame_count,
                 "minimax_visual_cond_noise_aug": context_strength}
        if ref_blocks:
            values["minimax_refs"] = ref_blocks
        cond = node_helpers.conditioning_set_values(cond, values)
        return io.NodeOutput(cond, latent)


class MiniMaxH3SigmaShift(io.ComfyNode):
    """Set the video/audio flow shifts coherently.

    The video shift drives the sampler's sigma schedule; both values are also
    handed to the DiT, which inverts the video schedule to the shared base grid
    and derives the audio schedule from it.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SigmaShift",
            description="Set the video/audio flow shifts.",
            display_name="MiniMax H3 Sigma Shift",
            category="model/patch/minimax",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, shift_video, shift_audio) -> io.NodeOutput:
        m = model.clone()

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
            pass

        original = m.get_model_object("model_sampling")
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift_video)
        if hasattr(original, "noise_scale"):
            model_sampling.set_noise_scale(original.noise_scale)
        m.add_object_patch("model_sampling", model_sampling)

        to = m.model_options["transformer_options"] = m.model_options.get("transformer_options", {}).copy()
        to["minimax_h3_sigma_shift_video"] = shift_video
        to["minimax_h3_sigma_shift_audio"] = shift_audio
        return io.NodeOutput(m)


class MiniMaxH3Extension(ComfyExtension):
    async def get_node_list(self):
        return [
            EmptyMiniMaxH3LatentAV,
            MiniMaxH3ImageToVideo,
            MiniMaxH3ReferenceToVideo,
            MiniMaxH3VideoExtend,
            MiniMaxH3SigmaShift
            ]


async def comfy_entrypoint() -> MiniMaxH3Extension:
    return MiniMaxH3Extension()
