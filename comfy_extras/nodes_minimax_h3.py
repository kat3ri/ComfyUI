"""MiniMax H3 nodes: AV latent creation and task conditioning (t2va / fl2va / ref2va).

The H3 packed-DiT consumes, via conditioning:
- Qwen3-VL-32B hidden states with per-token modality tags (from the minimax CLIP)
- keyframe / reference condition latents, re-injected every step (never denoised)

Latents are NestedTensor pairs (video [B,24,T,H/16,W/16], audio [B,32,2,T40]);
sampling runs on the flat pack with any stock sampler (the model handles the
audio stream's shifted schedule internally).
"""

import logging
import math
import os
import time

import safetensors.torch
import torch
import torchaudio

import folder_paths
import nodes
import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy.ldm.minimax.model import context_span
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


class MiniMaxH3EncodeAV(io.ComfyNode):
    """VAE-encode video frames (+ optional audio) into a MiniMax H3 AV latent.

    Inverse of decoding: for recombining externally-sourced or already-decoded
    footage into a context_latent for MiniMaxH3VideoExtend. Frames are encoded as
    a fresh sequence (the causal VAE's front padding treats whatever you pass in
    as the start), so prefer feeding VideoExtend a sampler's own AV latent output
    directly when you have it -- use this node when you only have pixels/audio.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EncodeAV",
            display_name="MiniMax H3 Encode AV",
            category="model/latent/minimax",
            description="VAE-encode video frames (+ optional audio) into a MiniMax H3 AV latent (NestedTensor pair).",
            inputs=[
                io.Vae.Input("vae"),
                io.Image.Input("images", tooltip="Video frames at 24 fps"),
                io.Vae.Input("audio_vae", optional=True),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, vae, images, audio_vae=None, audio=None) -> io.NodeOutput:
        video_z = vae.encode(images[..., :3])
        if audio is None:
            return io.NodeOutput({"samples": video_z})
        if audio_vae is None:
            raise ValueError("audio_vae is required when audio is supplied")
        audio_z, _ = _encode_ref_audio(audio_vae, audio)
        return io.NodeOutput({"samples": comfy.nested_tensor.NestedTensor((video_z, audio_z))})


def _pin_last_context_frame(vae, context_latent, width, height):
    """Decode context_latent's true trailing pixel frame and return a
    zero-RoPE-distance first_frame-style keyframe for it.

    context_latent's own context block (see _context_keyframes) only carries
    whole *latent* frames -- each spans 1-4 pixel frames (vae_ratio_t=4) -- so
    the exact pixel-level state at the handoff is never pinned precisely, just
    approximated by whichever latent frame happens to land nearest. That
    ambiguity is enough for the model to occasionally re-play a moment that
    already happened instead of continuing forward from it. Decoding the real
    trailing pixels and pinning them as an ordinary first_frame keyframe (the
    same zero-RoPE-distance anchor MiniMaxH3ImageToVideo already combines with
    context_latent) removes the ambiguity outright.

    Decodes a few extra trailing latent frames as a buffer before keeping only
    the very last pixel frame, so the causal VAE's own front-of-sequence
    transient (decoding a slice as if it were a fresh start) lands safely
    before the frame actually used."""
    ctx_samples = context_latent["samples"]
    ctx_video = ctx_samples.tensors[0] if ctx_samples.is_nested else ctx_samples
    ctx_t = ctx_video.shape[2]
    tail = ctx_video[:, :, max(0, ctx_t - 6):, :, :]
    decoded = vae.decode(tail)
    img = _resize(decoded[:, -1], width, height, "disabled")
    return {"resolved_frame_index": 0, "image": img}


def _context_keyframes(context_latent, context_frames, context_strength=1.0, static_time=False, expect_hw=None):
    """Build context/context_audio keyframe dicts continuing a prior generation's AV
    latent, plus the (width, height) it implies. Shared by MiniMaxH3ImageToVideo and
    MiniMaxH3VideoExtend.

    static_time: pin every context frame to target_origin (zero RoPE distance, all of
    them) instead of the default stepped-back-through-real-time spacing. Use this when
    context_latent isn't actually a prior moment in the same continuous clip (e.g. a
    rendered reference of a space) -- the default spacing tells the model these are
    sequential instants, which it then reads as motion to continue.

    expect_hw: (latent_h, latent_w) to validate this context latent's canvas against --
    used for a second, independent context-latent slot, which must share the primary
    context latent's exact canvas since PackedLayout builds one spatial grid per call."""
    ctx_samples = context_latent["samples"]
    # accept either a full AV NestedTensor (from an H3 sampler output) or a
    # plain video-only latent (e.g. a straight VAEEncode of external footage)
    is_av = ctx_samples.is_nested
    ctx_video = ctx_samples.tensors[0] if is_av else ctx_samples
    ctx_audio = ctx_samples.tensors[1] if is_av else None
    if ctx_video.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    ctx_t, ctx_h, ctx_w = ctx_video.shape[2], ctx_video.shape[3], ctx_video.shape[4]
    if expect_hw is not None and (ctx_h, ctx_w) != expect_hw:
        eh, ew = expect_hw
        raise ValueError(f"second context latent's canvas ({ctx_w * 16}x{ctx_h * 16}) doesn't match "
                         f"the primary context_latent's canvas ({ew * 16}x{eh * 16}) -- both must match exactly")
    n_frames = min(context_frames, ctx_t)
    width, height = ctx_w * 16, ctx_h * 16  # inherit the source clip's canvas exactly

    keyframes = [{"kind": "context", "num_frames": n_frames, "aug": context_strength,
                 "static_time": static_time,
                 "latent": ctx_video[:, :, ctx_t - n_frames:, :, :]}]
    if ctx_audio is not None:
        # match the audio context's duration to the video context's, both ending
        # at the same target origin (see context_span)
        ctx_audio_t = ctx_audio.shape[-1]
        n_audio_frames = min(round(context_span(n_frames)), ctx_audio_t)
        if n_audio_frames > 0:
            keyframes.append({"kind": "context_audio", "num_frames": n_audio_frames, "aug": context_strength,
                              "audio_latent": ctx_audio[:, :, :, ctx_audio_t - n_audio_frames:]})
    return width, height, keyframes


class MiniMaxH3ImageToVideo(io.ComfyNode):
    """t2va and fl2va: prompt (+ optional first/last keyframes, + optional prior-clip
    context) -> conditioning + AV latent."""

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
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32, tooltip="Ignored when context_latent is connected -- width/height are inherited from it instead"),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32, tooltip="Ignored when context_latent is connected -- width/height are inherited from it instead"),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested). Continuation-only length when context_latent is connected."),
                io.Image.Input("first_frame", optional=True, tooltip="Pins this call's own frame 0. Redundant/conflicting if context_latent is also connected -- context already determines what follows the prior clip."),
                io.Image.Input("last_frame", optional=True, tooltip="Pins this call's own last frame -- composes well with context_latent to land a continuation on a specific target image."),
                io.Latent.Input("context_latent", optional=True, tooltip="Continue from a prior MiniMax H3 generation's trailing latent frames (video extension)"),
                io.Int.Input("context_frames", default=2, min=1, max=64, tooltip="Only used when context_latent is connected: trailing latent frames of context_latent carried over as context"),
                io.Float.Input("context_strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Only used when context_latent is connected: 1.0 = context frames stay clean/hard-pinned; lower blends in noise for a softer handoff"),
                io.Boolean.Input("context_static_time", default=False, tooltip="Only used when context_latent is connected: pin every context frame to zero RoPE distance instead of stepping them back through real per-frame time spacing. Turn on when context_latent isn't actually a prior moment of the same continuous clip (e.g. a rendered space reference) -- the default spacing reads as 'these are sequential instants' and the model imitates that as motion (e.g. inheriting a rendered orbit's spin)."),
                io.Float.Input("temporal_stretch", default=1.0, min=1.0, max=100.0, step=0.5, tooltip="Spread the generated frames' RoPE time-positions apart by this factor, so a short clip occupies the time-span of a longer one -- its frames read as sparse keyframes of a long video instead of a dense burst (ChronoEdit-style skip-RoPE). 1.0 = normal. Experimental: intended for few-frame still/transition generation at short lengths, e.g. length 5 with stretch ~30 spans what ~124 frames normally would."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, length,
                first_frame=None, last_frame=None,
                context_latent=None, context_frames=2, context_strength=1.0,
                context_static_time=False, temporal_stretch=1.0) -> io.NodeOutput:
        keyframes = []
        if context_latent is not None:
            width, height, keyframes = _context_keyframes(context_latent, context_frames, context_strength,
                                                            static_time=context_static_time)

        latent, frame_count = _empty_av_latent(width, height, length)

        images = []
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

        values = {}
        if keyframes:
            for kf in keyframes:
                if "image" in kf:
                    kf["latent"] = vae.encode(kf.pop("image"))
            values["minimax_keyframes"] = keyframes
            values["minimax_frame_count"] = frame_count
        if temporal_stretch > 1.0:
            values["minimax_temporal_stretch"] = temporal_stretch
        if values:
            cond = node_helpers.conditioning_set_values(cond, values)
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
                      ref_images, ref_videos, ref_video_audios, ref_audios, ref_spacing=1.0, ref_strength=1.0):
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
        ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z,
                          "spacing": ref_spacing, "aug": ref_strength})

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
                           "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent,
                           "aug": ref_strength})

    for audio in (ref_audios or {}).values():
        if audio is None:
            continue
        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent,
                          "aug": ref_strength})

    return ref_items, ref_blocks


_REF_INPUTS = [
    io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
        tooltip="Reference image sizing. 'match' scales each ref (down only, keeping aspect) to the generation's pixel area; 'max' uses the reference pipeline's 2048px short edge for best identity fidelity. Reference tokens ride through every sampling step, so 'max' can be several times slower."),
    io.Float.Input("ref_spacing", default=1.0, min=0.0, max=50.0, step=0.5,
        tooltip="RoPE-distance separation between each image ref and the target/context origin. Higher reads to the model as more 'distant reference' vs 'recent/competing' content -- try raising this if refs seem to be drowning out context_latent continuation."),
    io.Float.Input("ref_strength", default=1.0, min=0.0, max=1.0, step=0.01,
        tooltip="Independent of ref_spacing: 1.0 = refs stay clean/hard-pinned; lower blends in noise for a softer identity nudge instead of a hard override. Tune this (not spacing) if refs need to keep asserting identity even when spaced far from context."),
    io.Float.Input("ref_decay", default=0.0, min=0.0, max=1.0, step=0.01,
        tooltip="Attention suppression strength for refs within ref_ramp of the context/target handoff (0 = no suppression -- refs influence the whole clip uniformly). Use with ref_ramp to let context own the transition and refs own identity for the rest of the clip."),
    io.Float.Input("ref_ramp", default=0.0, min=0.0, max=50.0, step=0.5,
        tooltip="Width (in the same units as ref_spacing) of the ref-suppression window centered on the context/target handoff. 0 disables the window (ref_decay has no effect). Try a few units first -- refs ramp back to full strength once a query frame is this far past the handoff."),
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
                io.Image.Input("first_frame", optional=True, tooltip="Pins this call's own frame 0 -- zero RoPE distance, the strongest anchor available (same mechanism as MiniMaxH3ImageToVideo's first_frame). Independent of the refs below, which use offset RoPE positioning instead."),
                io.Image.Input("last_frame", optional=True, tooltip="Pins this call's own last frame -- same zero-RoPE-distance anchor as first_frame, applied to the end instead."),
                io.Float.Input("temporal_stretch", default=1.0, min=1.0, max=100.0, step=0.5, tooltip="Spread the generated frames' RoPE time-positions apart by this factor, so a short clip occupies the time-span of a longer one -- its frames read as sparse keyframes of a long video instead of a dense burst (ChronoEdit-style skip-RoPE). 1.0 = normal. Experimental: intended for few-frame still/transition generation at short lengths, e.g. length 5 with stretch ~30 spans what ~124 frames normally would."),
                io.Latent.Input("world_latent", optional=True, tooltip="World-grounding context latent (e.g. MiniMaxH3SceneToContextLatent's output, or a VAE-encoded room render) pinned as context rows at the target's RoPE origin -- the latent-input half of the grounding recipe, in addition to refs conditioning. Must share this node's exact canvas (width/height)."),
                io.Int.Input("world_frames", default=1, min=1, max=64, tooltip="Only used when world_latent is connected: trailing latent frames carried over as context"),
                io.Float.Input("world_strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Only used when world_latent is connected: 1.0 = context frames stay clean/hard-pinned; lower blends in noise"),
                io.Boolean.Input("world_static_time", default=True, tooltip="Only used when world_latent is connected: pin every context frame to a single zero-RoPE-distance point instead of stepping back through per-frame time. Keep True for spatial references (a room), False only for genuine prior-clip footage."),
                *_REF_INPUTS,
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length, first_frame=None, last_frame=None,
                temporal_stretch=1.0,
                world_latent=None, world_frames=1, world_strength=1.0, world_static_time=True,
                ref_image_size="match", ref_spacing=1.0, ref_strength=1.0, ref_decay=0.0, ref_ramp=0.0,
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        keyframes = []
        keyframe_items = []
        if first_frame is not None:
            # geometry anchor: plain stretch to canvas, same convention as MiniMaxH3ImageToVideo
            img = _resize(first_frame[:1], width, height, "disabled")
            keyframe_items.append({"type": "image", "data": img})
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if last_frame is not None:
            # follower: aspect-preserving cover-crop, same convention as MiniMaxH3ImageToVideo
            img = _resize(last_frame[:1], width, height, "center")
            keyframe_items.append({"type": "image", "data": img})
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

        world_keyframes = []
        if world_latent is not None:
            # latent-input grounding: pinned context rows at the target's RoPE
            # origin, same slot machinery as VideoExtend's world_latent
            _, _, world_keyframes = _context_keyframes(world_latent, world_frames, world_strength,
                                                       static_time=world_static_time,
                                                       expect_hw=(height // 16, width // 16))

        ref_items, ref_blocks = _build_ref_blocks(vae, audio_vae, width, height, frame_count, ref_image_size,
                                                  ref_images, ref_videos, ref_video_audios, ref_audios,
                                                  ref_spacing, ref_strength)

        # tokenize_with_weights treats images= and minimax_ref_items= as
        # mutually exclusive (an if/else, not additive) -- merge first/last
        # frame into the SAME ref_items list so both get proper <Picture N>
        # tags in one consistent numbering, rather than dropping one set
        # silently. The DiT-conditioning side stays separate: keyframes get
        # zero-RoPE-distance pinning, refs get offset ref_spacing -- distinct
        # mechanisms PackedLayout already accepts simultaneously.
        tokens = clip.tokenize(prompt, minimax_ref_items=keyframe_items + ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        values = {}
        if keyframes or world_keyframes:
            for kf in keyframes:
                if "image" in kf:
                    kf["latent"] = vae.encode(kf.pop("image"))
            values["minimax_keyframes"] = keyframes + world_keyframes
            values["minimax_frame_count"] = frame_count
        if ref_blocks:
            values["minimax_refs"] = ref_blocks
            values["minimax_ref_decay"] = ref_decay
            values["minimax_ref_ramp"] = ref_ramp
        if temporal_stretch > 1.0:
            values["minimax_temporal_stretch"] = temporal_stretch
        if values:
            cond = node_helpers.conditioning_set_values(cond, values)
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
                io.Int.Input("context_frames", default=2, min=1, max=64, tooltip="Trailing latent frames of context_latent carried over as context (1 latent frame covers 1-4 pixel frames)"),
                io.Boolean.Input("context_static_time", default=False, tooltip="Pin every context frame to zero RoPE distance instead of stepping them back through real per-frame time spacing. Turn on when context_latent isn't actually a prior moment of the same continuous clip (e.g. a rendered space reference) -- the default spacing reads as 'these are sequential instants' and the model imitates that as motion (e.g. inheriting a rendered orbit's spin)."),
                io.Float.Input("context_strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="1.0 = context frames stay clean/hard-pinned; lower blends in noise for a softer handoff"),
                io.Boolean.Input("pin_last_frame", default=True, tooltip="Decode context_latent's true trailing pixel frame and pin it as this call's own frame 0 (same zero-RoPE-distance anchor as first_frame below). context_frames alone only carries whole latent frames (each spanning 1-4 pixel frames), leaving the exact handoff state a little ambiguous -- pinning the real decoded pixels removes that ambiguity, which otherwise can show up as the continuation re-playing a moment that already happened. Ignored if first_frame is connected."),
                io.Image.Input("first_frame", optional=True, tooltip="Hard-pin this call's own frame 0 to an exact image you supply (e.g. the prior clip's actual last output frame), instead of pin_last_frame's own decode of context_latent. Use this if pin_last_frame's decode-of-a-latent-slice roundtrip is introducing a color-grade mismatch at the handoff -- feeding the real pixels directly skips that roundtrip entirely."),
                io.Latent.Input("world_latent", optional=True, tooltip="Second, independent context block -- e.g. a rendered room/scene reference -- kept separate from context_latent so that slot stays free for real prior-clip continuation. Must share context_latent's exact canvas. Chained immediately behind context_latent's own context frames in RoPE time (not colliding with them)."),
                io.Int.Input("world_frames", default=2, min=1, max=64, tooltip="Only used when world_latent is connected: trailing latent frames of world_latent carried over as context"),
                io.Float.Input("world_strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Only used when world_latent is connected: 1.0 = context frames stay clean/hard-pinned; lower blends in noise for a softer handoff"),
                io.Boolean.Input("world_static_time", default=False, tooltip="Only used when world_latent is connected: pin every world_latent context frame to a single zero-RoPE-distance point instead of stepping them back through real per-frame time spacing."),
                *_REF_INPUTS,
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, context_latent, prompt, length, context_frames=2, context_strength=1.0,
                context_static_time=False, pin_last_frame=True, first_frame=None,
                world_latent=None, world_frames=2, world_strength=1.0, world_static_time=False,
                audio_vae=None, ref_image_size="match", ref_spacing=1.0, ref_strength=1.0,
                ref_decay=0.0, ref_ramp=0.0, ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None) -> io.NodeOutput:
        width, height, keyframes = _context_keyframes(context_latent, context_frames, context_strength,
                                                        static_time=context_static_time)
        if first_frame is not None:
            # exact pixels supplied directly -- skips the decode-of-a-latent-slice
            # roundtrip pin_last_frame does, which can shift color grade slightly
            img = _resize(first_frame[:1], width, height, "disabled")
            keyframes.append({"resolved_frame_index": 0, "image": img})
        elif pin_last_frame:
            keyframes.append(_pin_last_context_frame(vae, context_latent, width, height))
        if world_latent is not None:
            # chained behind context_latent's own context block (appended after it in
            # the list) -- see PackedLayout's context_cursor for the chaining mechanism
            _, _, world_keyframes = _context_keyframes(world_latent, world_frames, world_strength,
                                                        static_time=world_static_time,
                                                        expect_hw=(height // 16, width // 16))
            keyframes = keyframes + world_keyframes
        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items, ref_blocks = ([], [])
        if any((ref_images, ref_videos, ref_audios)):
            if audio_vae is None and (ref_video_audios or ref_audios):
                raise ValueError("audio_vae is required when ref_video_audios or ref_audios are supplied")
            ref_items, ref_blocks = _build_ref_blocks(vae, audio_vae, width, height, frame_count, ref_image_size,
                                                      ref_images, ref_videos, ref_video_audios, ref_audios,
                                                      ref_spacing, ref_strength)

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        for kf in keyframes:
            if "image" in kf:
                kf["latent"] = vae.encode(kf.pop("image"))

        values = {"minimax_keyframes": keyframes, "minimax_frame_count": frame_count}
        if ref_blocks:
            values["minimax_refs"] = ref_blocks
            values["minimax_ref_decay"] = ref_decay
            values["minimax_ref_ramp"] = ref_ramp
        cond = node_helpers.conditioning_set_values(cond, values)
        return io.NodeOutput(cond, latent)


_SCENE_ADAPTER_REPO = "/weka/home-kateriw/h3-3d-adapter"


class MiniMaxH3SceneFromPLY(io.ComfyNode):
    """Research prototype (untrained by default): encode a raw gaussian-splat
    .ply directly -- no rendering step -- into a spatial latent grid shaped
    like a real VAE keyframe latent, via the GaussianSceneEncoder adapter
    (per-splat MLP featurizer + distance-biased cross-attention pooling into a
    h_lat x w_lat grid + self-attention refine + linear-to-24-channels,
    developed in h3-3d-adapter). Because the output is shaped like a real
    keyframe latent, it rides through H3's *existing* "ref_img" pathway --
    real frame_grid RoPE placement, video_patch_proj, ref_spacing/ref_strength
    -- as a synthetic reference image, skipping only the vae.encode() step
    since we already have the latent. Appends to whatever refs the upstream
    ReferenceToVideo/VideoExtend node already set, rather than replacing them.

    Without adapter_checkpoint this is plumbing-only: a randomly initialized
    adapter cannot meaningfully steer generation, it only proves the latent
    flows through the real model without shape/dtype errors (the spatial
    structure from the distance bias exists even untrained, but what content
    gets encoded there is what training teaches).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SceneFromPLY",
            display_name="MiniMax H3 Scene From PLY (experimental adapter)",
            category="model/conditioning/minimax",
            description="Encode a gaussian-splat .ply into a synthetic reference latent grounding H3 in that 3D space (3D-scene adapter prototype, untrained unless adapter_checkpoint is set).",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input("ply_path", tooltip="Path to a 3DGS .ply (needs x,y,z,f_dc_0-2,opacity,scale_0-2,rot_0-3 vertex fields)"),
                io.Int.Input("h_lat", default=16, min=4, max=64, tooltip="Must match adapter_checkpoint's own h_lat if one is given"),
                io.Int.Input("w_lat", default=24, min=4, max=64, tooltip="Must match adapter_checkpoint's own w_lat if one is given"),
                io.Int.Input("subsample", default=200000, min=1000, max=2000000, tooltip="Random subsample of gaussians (CPU/memory tradeoff); scenes run in the hundreds of thousands of splats"),
                io.String.Input("adapter_checkpoint", default="", optional=True, tooltip="Path to a trained GaussianSceneEncoder state_dict; empty = random-init (plumbing test only, no real conditioning effect)"),
                io.Float.Input("scene_strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Noise-blend strength (aug), same convention as ref_strength"),
                io.Float.Input("scene_spacing", default=1.0, min=0.0, max=50.0, step=0.5, tooltip="RoPE-distance separation from target/context origin, same convention as ref_spacing"),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning, ply_path, h_lat=16, w_lat=24, subsample=200000,
                adapter_checkpoint="", scene_strength=1.0, scene_spacing=1.0) -> io.NodeOutput:
        import sys
        if _SCENE_ADAPTER_REPO not in sys.path:
            sys.path.insert(0, _SCENE_ADAPTER_REPO)
        import numpy as np
        from plyfile import PlyData
        from encoder.gaussian_encoder import GaussianSceneEncoder

        ply = PlyData.read(ply_path)
        v = ply["vertex"].data
        n = len(v)
        idx = np.random.choice(n, size=min(subsample, n), replace=False) if 0 < subsample < n else np.arange(n)

        def field(name):
            return torch.from_numpy(np.ascontiguousarray(v[name][idx]).astype(np.float32))

        xyz = torch.stack([field("x"), field("y"), field("z")], dim=1)
        f_dc = torch.stack([field("f_dc_0"), field("f_dc_1"), field("f_dc_2")], dim=1)
        opacity = field("opacity").unsqueeze(1)
        scale = torch.stack([field("scale_0"), field("scale_1"), field("scale_2")], dim=1)
        rot = torch.stack([field("rot_0"), field("rot_1"), field("rot_2"), field("rot_3")], dim=1)

        device = comfy.model_management.get_torch_device()
        encoder = GaussianSceneEncoder(h_lat=h_lat, w_lat=w_lat)
        if adapter_checkpoint:
            encoder.load_state_dict(torch.load(adapter_checkpoint, map_location="cpu"))
        else:
            logging.warning("MiniMaxH3SceneFromPLY: no adapter_checkpoint given -- scene latent is random-init, plumbing test only")
        encoder.to(device).eval()
        with torch.no_grad():
            latent = encoder(xyz.to(device), f_dc.to(device), opacity.to(device), scale.to(device), rot.to(device))

        ref_block = {"kind": "image", "latent_h": h_lat, "latent_w": w_lat,
                    "latent": latent.to(comfy.model_management.intermediate_device()),
                    "aug": scene_strength, "spacing": scene_spacing}
        cond = node_helpers.conditioning_set_values(conditioning, {"minimax_refs": [ref_block]}, append=True)
        return io.NodeOutput(cond)


class MiniMaxH3RenderPLYWalkthrough(io.ComfyNode):
    """Render a real, VAE-ready video from a complete 3DGS .ply (e.g. a pano
    reconstructed via midi3d-spike/pano_to_ply.py) by panning in place around
    its capture origin -- NOT translating, since coverage was only validated
    to be complete (~98-99.97%) exactly at that point; moving away reveals the
    same single-viewpoint gaps a raw render always has.

    Output is a plain IMAGE batch (real gsplat renders, no learned encoder,
    no distribution-mismatch risk) -- wire it directly into
    MiniMaxH3ReferenceToVideo's ref_video input, which VAE-encodes it the
    same way it would any other reference video.

    Assumes the .ply's own coordinate frame is Y-up with the capture point at
    the origin (pano_to_ply.py's convention) -- NOT DA3's raw Y-down camera
    frame. Check the source of your .ply if results look upside-down.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RenderPLYWalkthrough",
            display_name="MiniMax H3 Render PLY Walkthrough",
            category="model/conditioning/minimax",
            description="Pan-in-place render of a complete gaussian-splat .ply (e.g. a pano shell) into a real video, for use as an ordinary ref_video reference.",
            inputs=[
                io.String.Input("ply_path", tooltip="Path to a complete 3DGS .ply, capture origin at (0,0,0), Y-up (pano_to_ply.py convention)"),
                io.Int.Input("n_frames", default=48, min=5, max=362, tooltip="Output frame count at 24fps"),
                io.Float.Input("sweep_deg", default=360.0, min=1.0, max=720.0, tooltip="Total yaw rotation across the clip"),
                io.Float.Input("start_yaw_deg", default=0.0, min=-360.0, max=360.0),
                io.Float.Input("fov_deg", default=80.0, min=10.0, max=170.0),
                io.Int.Input("render_h", default=512, min=64, max=2048, step=16),
                io.Int.Input("render_w", default=512, min=64, max=2048, step=16),
                io.Int.Input("max_gaussians", default=10_000_000, min=10_000, max=20_000_000, tooltip="Random subsample cap for render speed/memory -- keep above the .ply's actual point count (e.g. pano_to_ply.py's full-density output is ~8.4M) or its splats, sized for that density, will be too small for the sparser subsample and speckle/holes reappear"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, ply_path, n_frames=48, sweep_deg=360.0, start_yaw_deg=0.0,
                fov_deg=80.0, render_h=512, render_w=512, max_gaussians=10_000_000) -> io.NodeOutput:
        import sys
        _filmworld_repo = "/weka/home-kateriw/ComfyUI/custom_nodes/ComfyUI-Filmworld"
        if _filmworld_repo not in sys.path:
            sys.path.insert(0, _filmworld_repo)
        import numpy as np
        from plyfile import PlyData
        from da3_helper import render_gaussians_at_poses

        C0 = 0.28209479177387814
        device = comfy.model_management.get_torch_device()

        p = PlyData.read(ply_path)
        v = p["vertex"].data
        n = len(v)
        idx = np.random.default_rng(0).choice(n, size=min(max_gaussians, n), replace=False) if n > max_gaussians else np.arange(n)

        def field(name):
            return torch.from_numpy(np.ascontiguousarray(v[name][idx]).astype(np.float32))

        xyz = torch.stack([field("x"), field("y"), field("z")], dim=1).to(device)
        f_dc = torch.stack([field("f_dc_0"), field("f_dc_1"), field("f_dc_2")], dim=1).to(device)
        opacity_logit = field("opacity").to(device)
        scale_log = torch.stack([field("scale_0"), field("scale_1"), field("scale_2")], dim=1).to(device)
        rot = torch.stack([field("rot_0"), field("rot_1"), field("rot_2"), field("rot_3")], dim=1).to(device)

        scales = torch.exp(scale_log)
        opacities = torch.sigmoid(opacity_logit)
        rgb_equiv = (0.5 + C0 * f_dc).clamp(1e-6, 1 - 1e-6)
        dc_presigmoid = torch.logit(rgb_equiv)

        class _GaussiansLike:
            def __init__(self, means, scales, rotations, opacities, harmonics):
                self.means, self.scales, self.rotations = means, scales, rotations
                self.opacities, self.harmonics = opacities, harmonics

        gaussians = _GaussiansLike(
            means=xyz.unsqueeze(0), scales=scales.unsqueeze(0), rotations=rot.unsqueeze(0),
            opacities=opacities.unsqueeze(0), harmonics=dc_presigmoid.unsqueeze(0).unsqueeze(-1),
        )

        # pan in place: fixed position at the capture origin, yaw sweeps
        # start_yaw_deg -> start_yaw_deg + sweep_deg across n_frames
        w2cs = []
        for i in range(n_frames):
            yaw = math.radians(start_yaw_deg + sweep_deg * (i / max(n_frames - 1, 1)))
            forward = np.array([math.cos(yaw), 0.0, math.sin(yaw)], dtype=np.float32)
            down = np.array([0.0, -1.0, 0.0], dtype=np.float32)  # pano_to_ply.py's .ply is Y-up
            f = forward / max(np.linalg.norm(forward), 1e-9)
            d = down - np.dot(down, f) * f
            d /= max(np.linalg.norm(d), 1e-9)
            r = np.cross(d, f)
            r /= max(np.linalg.norm(r), 1e-9)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = r, d, f
            w2cs.append(np.linalg.inv(c2w))
        w2cs = torch.from_numpy(np.stack(w2cs).astype(np.float32)).to(device)

        fx = (render_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        K = torch.zeros(3, 3, dtype=torch.float32, device=device)
        K[0, 0] = fx; K[1, 1] = fx; K[0, 2] = render_w / 2.0; K[1, 2] = render_h / 2.0; K[2, 2] = 1.0
        Ks = K.unsqueeze(0).repeat(n_frames, 1, 1)

        rgb, _depth, alpha = render_gaussians_at_poses(gaussians, w2cs, Ks, render_h, render_w, use_sh=False, chunk_size=8)
        coverage = alpha.mean().item()
        logging.info(f"MiniMaxH3RenderPLYWalkthrough: {n} gaussians (subsampled to {min(max_gaussians, n)}), "
                     f"mean alpha coverage {coverage:.4f} across {n_frames} frames")

        images = rgb.clamp(0, 1).to(comfy.model_management.intermediate_device())
        return io.NodeOutput(images)


class MiniMaxH3RenderPLYCutoutViews(io.ComfyNode):
    """Experimental alternative to MiniMaxH3RenderPLYWalkthrough: instead of one
    smooth panning sweep (a real camera move, which the model reads as motion to
    continue -- see context_static_time's failure mode), render a batch of
    DISCRETE still positions around the .ply's capture origin -- a yaw ring at
    each of several pitch rows (looking level, up, down) -- spaced so each
    neighbor overlaps by a set fraction of the field of view.

    The idea being tested: since these views aren't consecutive frames of one
    continuous move, feeding them as ordinary ref_images (Autogrow, each its own
    independent spatial grid and RoPE position -- no shared frame_grid like
    context/world_latent's "cond" mechanism has) might let the model build a
    coherent sense of the room's layout from several independent stills, without
    inheriting a fabricated camera motion the way a swept video can.

    Deliberately a separate node from MiniMaxH3RenderPLYWalkthrough -- same
    gaussian-loading and rendering code, just a different, discrete camera
    schedule -- so the validated walkthrough+context_latent recipe is untouched
    while this alternative gets tried.

    Same .ply coordinate-frame assumption as MiniMaxH3RenderPLYWalkthrough:
    Y-up, capture origin at (0,0,0) (pano_to_ply.py's convention).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RenderPLYCutoutViews",
            display_name="MiniMax H3 Render PLY Cutout Views",
            category="model/conditioning/minimax",
            is_experimental=True,
            description="Discrete, overlapping still views around a .ply's capture origin (not a smooth sweep) -- wire the output batch into separate ref_image slots to test whether several static cutouts ground the model better than a swept reference video.",
            inputs=[
                io.String.Input("ply_path", tooltip="Path to a complete 3DGS .ply, capture origin at (0,0,0), Y-up (pano_to_ply.py convention)"),
                io.Float.Input("overlap_percent", default=25.0, min=0.0, max=90.0, tooltip="Target overlap between each view and its neighbor, as a fraction of fov_deg. Together with fov_deg this determines how many views are needed to cover the full 360 degrees (ignored if num_views is set > 0)."),
                io.Int.Input("num_views", default=0, min=0, max=32, tooltip="Explicit view count per pitch row, evenly spaced across 360 degrees of yaw. 0 = auto-compute from overlap_percent and fov_deg."),
                io.Int.Input("num_pitch_rows", default=3, min=1, max=9, tooltip="How many pitch rows to render a full yaw ring at (1 = level only, like the old behavior). E.g. 3 = looking down, level, and up."),
                io.Float.Input("pitch_max_deg", default=45.0, min=0.0, max=85.0, tooltip="Extreme up/down tilt for the outermost pitch rows (rows are evenly spaced from -pitch_max_deg to +pitch_max_deg). Ignored if num_pitch_rows is 1."),
                io.Float.Input("start_yaw_deg", default=0.0, min=-360.0, max=360.0),
                io.Float.Input("fov_deg", default=80.0, min=10.0, max=170.0),
                io.Int.Input("render_h", default=512, min=64, max=2048, step=16),
                io.Int.Input("render_w", default=512, min=64, max=2048, step=16),
                io.Int.Input("max_gaussians", default=10_000_000, min=10_000, max=20_000_000, tooltip="Random subsample cap for render speed/memory -- keep above the .ply's actual point count or splats will be too small for the sparser subsample and speckle/holes reappear"),
            ],
            outputs=[io.Image.Output(tooltip="One still per view, in yaw order -- wire individual frames (e.g. via ImageFromBatch) into separate ref_image Autogrow slots")],
        )

    @classmethod
    def execute(cls, ply_path, overlap_percent=25.0, num_views=0, num_pitch_rows=3, pitch_max_deg=45.0,
                start_yaw_deg=0.0, fov_deg=80.0, render_h=512, render_w=512, max_gaussians=10_000_000) -> io.NodeOutput:
        import sys
        _filmworld_repo = "/weka/home-kateriw/ComfyUI/custom_nodes/ComfyUI-Filmworld"
        if _filmworld_repo not in sys.path:
            sys.path.insert(0, _filmworld_repo)
        import numpy as np
        from plyfile import PlyData
        from da3_helper import render_gaussians_at_poses

        C0 = 0.28209479177387814
        device = comfy.model_management.get_torch_device()

        if num_views > 0:
            n_views_per_row = num_views
        else:
            step_deg = fov_deg * (1.0 - overlap_percent / 100.0)
            n_views_per_row = max(1, math.ceil(360.0 / max(step_deg, 1e-6)))

        if num_pitch_rows <= 1:
            pitches_deg = [0.0]
        else:
            pitches_deg = [-pitch_max_deg + 2.0 * pitch_max_deg * (i / (num_pitch_rows - 1))
                          for i in range(num_pitch_rows)]
        n_views = n_views_per_row * len(pitches_deg)

        p = PlyData.read(ply_path)
        v = p["vertex"].data
        n = len(v)
        idx = np.random.default_rng(0).choice(n, size=min(max_gaussians, n), replace=False) if n > max_gaussians else np.arange(n)

        def field(name):
            return torch.from_numpy(np.ascontiguousarray(v[name][idx]).astype(np.float32))

        xyz = torch.stack([field("x"), field("y"), field("z")], dim=1).to(device)
        f_dc = torch.stack([field("f_dc_0"), field("f_dc_1"), field("f_dc_2")], dim=1).to(device)
        opacity_logit = field("opacity").to(device)
        scale_log = torch.stack([field("scale_0"), field("scale_1"), field("scale_2")], dim=1).to(device)
        rot = torch.stack([field("rot_0"), field("rot_1"), field("rot_2"), field("rot_3")], dim=1).to(device)

        scales = torch.exp(scale_log)
        opacities = torch.sigmoid(opacity_logit)
        rgb_equiv = (0.5 + C0 * f_dc).clamp(1e-6, 1 - 1e-6)
        dc_presigmoid = torch.logit(rgb_equiv)

        class _GaussiansLike:
            def __init__(self, means, scales, rotations, opacities, harmonics):
                self.means, self.scales, self.rotations = means, scales, rotations
                self.opacities, self.harmonics = opacities, harmonics

        gaussians = _GaussiansLike(
            means=xyz.unsqueeze(0), scales=scales.unsqueeze(0), rotations=rot.unsqueeze(0),
            opacities=opacities.unsqueeze(0), harmonics=dc_presigmoid.unsqueeze(0).unsqueeze(-1),
        )

        # discrete, evenly-spaced yaw x pitch positions -- each an independent
        # still, not frames of one continuous move (contrast with
        # MiniMaxH3RenderPLYWalkthrough). +pitch looks up, -pitch looks down
        # (Y-up convention, matching down=[0,-1,0] below).
        w2cs = []
        for pitch_deg in pitches_deg:
            pitch = math.radians(pitch_deg)
            for i in range(n_views_per_row):
                yaw = math.radians(start_yaw_deg + 360.0 * (i / n_views_per_row))
                forward = np.array([math.cos(pitch) * math.cos(yaw), math.sin(pitch),
                                    math.cos(pitch) * math.sin(yaw)], dtype=np.float32)
                down = np.array([0.0, -1.0, 0.0], dtype=np.float32)  # pano_to_ply.py's .ply is Y-up
                f = forward / max(np.linalg.norm(forward), 1e-9)
                d = down - np.dot(down, f) * f
                d /= max(np.linalg.norm(d), 1e-9)
                r = np.cross(d, f)
                r /= max(np.linalg.norm(r), 1e-9)
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = r, d, f
                w2cs.append(np.linalg.inv(c2w))
        w2cs = torch.from_numpy(np.stack(w2cs).astype(np.float32)).to(device)

        fx = (render_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        K = torch.zeros(3, 3, dtype=torch.float32, device=device)
        K[0, 0] = fx; K[1, 1] = fx; K[0, 2] = render_w / 2.0; K[1, 2] = render_h / 2.0; K[2, 2] = 1.0
        Ks = K.unsqueeze(0).repeat(n_views, 1, 1)

        rgb, _depth, alpha = render_gaussians_at_poses(gaussians, w2cs, Ks, render_h, render_w, use_sh=False, chunk_size=8)
        coverage = alpha.mean().item()
        logging.info(f"MiniMaxH3RenderPLYCutoutViews: {n_views} discrete views ({len(pitches_deg)} pitch rows x "
                     f"{n_views_per_row} yaw steps of {360.0 / n_views_per_row:.1f} deg, pitches {pitches_deg}), "
                     f"fov {fov_deg} deg, {n} gaussians (subsampled to {min(max_gaussians, n)}), "
                     f"mean alpha coverage {coverage:.4f}")

        images = rgb.clamp(0, 1).to(comfy.model_management.intermediate_device())
        return io.NodeOutput(images)


class MiniMaxH3LatentUpsample(io.ComfyNode):
    """2x spatial upscale of an H3 video latent via a trained latent upscaler.

    H3 latents are already normalized inside the VAE, so unlike the LTX
    equivalent there is no per-channel-statistics round-trip and no VAE input.
    The audio stream of an AV NestedTensor passes through untouched. Follow
    with a low-denoise sampling pass at the new canvas to restore detail.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LatentUpsample",
            display_name="MiniMax H3 Latent Upscale",
            category="model/latent/minimax",
            is_experimental=True,
            description="2x spatial upscale of an H3 video latent (audio passes through). Checkpoint from h3-latent-upscaler via Load Latent Upscale Model.",
            inputs=[
                io.Latent.Input("samples"),
                io.LatentUpscaleModel.Input("upscale_model"),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, samples, upscale_model) -> io.NodeOutput:
        device = upscale_model.load_device
        model = upscale_model.model
        model_dtype = upscale_model.model_dtype()
        latents = samples["samples"]
        audio = None
        if latents.is_nested:
            video, audio = latents.tensors
        else:
            video = latents
        input_dtype = video.dtype
        logging.info(
            f"MiniMaxH3LatentUpsample: input nested={latents.is_nested} "
            f"video={tuple(video.shape)} audio={tuple(audio.shape) if audio is not None else None}"
        )
        if video.ndim != 5 or video.shape[1] != 24:
            raise ValueError(
                f"MiniMaxH3LatentUpsample expects a video latent [B,24,T,H,W], got {tuple(video.shape)} "
                f"(nested={latents.is_nested}). Something upstream flattened or reordered the AV latent -- "
                f"use MiniMaxH3SplitAV to inspect."
            )

        memory_required = math.prod(video.shape) * 3000.0
        comfy.model_management.load_models_gpu([upscale_model], memory_required=memory_required)

        upscaled = model(video.to(dtype=model_dtype, device=device))
        upscaled = upscaled.to(dtype=input_dtype, device=comfy.model_management.intermediate_device())

        out = samples.copy()
        if audio is not None:
            out["samples"] = comfy.nested_tensor.NestedTensor((upscaled, audio))
        else:
            out["samples"] = upscaled
        out.pop("noise_mask", None)
        return io.NodeOutput(out)


class MiniMaxH3SplitAV(io.ComfyNode):
    """Split an H3 AV NestedTensor latent into separate video and audio latents.

    Diagnostic + plumbing: lets you route the video stream alone (e.g. into a
    latent upscaler or inspection nodes) and rejoin with MiniMaxH3JoinAV.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SplitAV",
            display_name="MiniMax H3 Split AV Latent",
            category="model/latent/minimax",
            inputs=[io.Latent.Input("samples")],
            outputs=[
                io.Latent.Output(display_name="video"),
                io.Latent.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(cls, samples) -> io.NodeOutput:
        latents = samples["samples"]
        if latents.is_nested:
            video, audio = latents.tensors
        else:
            video, audio = latents, None
        logging.info(
            f"MiniMaxH3SplitAV: nested={latents.is_nested} video={tuple(video.shape)} "
            f"audio={tuple(audio.shape) if audio is not None else None}"
        )
        audio_out = {"samples": audio} if audio is not None else {"samples": torch.zeros(1, 32, 2, 1)}
        return io.NodeOutput({"samples": video}, audio_out)


class MiniMaxH3JoinAV(io.ComfyNode):
    """Rejoin video and audio latents into an H3 AV NestedTensor latent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3JoinAV",
            display_name="MiniMax H3 Join AV Latent",
            category="model/latent/minimax",
            inputs=[
                io.Latent.Input("video"),
                io.Latent.Input("audio", optional=True),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, video, audio=None) -> io.NodeOutput:
        v = video["samples"]
        if audio is None:
            return io.NodeOutput({"samples": v})
        return io.NodeOutput({"samples": comfy.nested_tensor.NestedTensor((v, audio["samples"]))})


# --- loading saved latents back in -------------------------------------------
# Core's LoadLatent lists input/ only (top level, non-recursive) while SaveLatent
# writes to output/latents/, so a saved latent can never appear in its dropdown,
# and naming one fails VALIDATE_INPUTS because an un-annotated name resolves
# against input/. These read from wherever the file actually is.

_LATENT_KEYS = ("samples", "latent_tensor", "lr_latent", "hr_latent", "latent", "video")
_SD_SCALE = 1.0 / 0.18215
_SCAN_CACHE = {"at": 0.0, "files": []}
_SCAN_TTL = 15.0        # define_schema runs on every /object_info; don't re-walk output/
_SCAN_DEPTH = 4         # latents live near the top; the rest of output/ can be huge


def _scan_latents(limit=400):
    """Latent files under output/ and input/, newest first, deduped by real path.

    `.latent` anywhere; `.safetensors` only inside a directory whose name mentions
    "latent", so exported LoRAs don't flood the dropdown.
    """
    now = time.time()
    if now - _SCAN_CACHE["at"] < _SCAN_TTL:
        return list(_SCAN_CACHE["files"])

    base = folder_paths.base_path
    found, seen = [], set()
    for root in (folder_paths.get_output_directory(), folder_paths.get_input_directory()):
        if not os.path.isdir(root):
            continue
        root_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            if dirpath.count(os.sep) - root_depth >= _SCAN_DEPTH:
                dirnames[:] = []
            else:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            in_latent_dir = "latent" in os.path.relpath(dirpath, root).lower()
            for fn in filenames:
                if not (fn.endswith(".latent") or (in_latent_dir and fn.endswith(".safetensors"))):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    real = os.path.realpath(full)
                    if real in seen:
                        continue
                    mtime = os.path.getmtime(full)
                except OSError:      # broken symlink, or vanished mid-walk
                    continue
                seen.add(real)
                found.append((mtime, os.path.relpath(full, base)))
    found.sort(reverse=True)
    files = [name for _, name in found[:limit]]
    _SCAN_CACHE.update(at=now, files=files)
    return list(files)


def _resolve_latent_path(latent, path=""):
    """Absolute path for the chosen file, or None. `path` wins when set."""
    path = (path or "").strip()
    roots = [folder_paths.base_path, folder_paths.get_output_directory(),
             folder_paths.get_input_directory()]
    if path:
        cand = os.path.expanduser(path)
        for root in [""] + roots:
            full = cand if os.path.isabs(cand) else os.path.join(root, cand)
            if os.path.isfile(full):
                return full
        return None
    if not latent:
        return None
    for root in roots:
        full = os.path.join(root, latent)
        if os.path.isfile(full):
            return full
    return None


def _extract_latent(sd, key=""):
    """Pull the video tensor out of a loaded state dict as [B, C, T, H, W]."""
    key = (key or "").strip()
    if key:
        if key not in sd:
            raise ValueError("key '{}' not in file (has: {})".format(key, ", ".join(sd.keys())))
        name = key
    else:
        name = next((k for k in _LATENT_KEYS if k in sd), None)
        if name is None:
            tensors = [k for k, v in sd.items() if v.numel() > 0]
            if len(tensors) != 1:
                raise ValueError("no known latent key; found {}. Set 'key' to pick one."
                                 .format(", ".join(sd.keys()) or "nothing"))
            name = tensors[0]

    t = sd[name].float()
    if t.ndim == 4:                  # [C, T, H, W] -> add batch
        t = t.unsqueeze(0)
    if t.ndim != 5:
        raise ValueError("'{}' has shape {}; expected 5 dims [B,C,T,H,W]"
                         .format(name, tuple(t.shape)))

    # Core SaveLatent's legacy format is SD-scaled; its v0 marker means "not scaled".
    # Files written with a 'samples' key are never scaled.
    if name == "latent_tensor" and "latent_format_version_0" not in sd:
        t = t * _SD_SCALE

    if t.shape[1] != 24:
        logging.warning("MiniMaxH3LoadVideoLatent: %d channels, H3 latents have 24", t.shape[1])
    return t, name


class MiniMaxH3LoadVideoLatent(io.ComfyNode):
    """Load a saved H3 video latent from output/, input/, or any path."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LoadVideoLatent",
            display_name="MiniMax H3 Load Video Latent",
            category="model/latent/minimax",
            description="Load a saved H3 video latent by browsing output/ and input/, or by "
                        "pasting a path. Returns the video stream -- add Join AV Latent for a "
                        "sampler-ready AV latent.",
            inputs=[
                io.Combo.Input("latent", options=_scan_latents(),
                               tooltip="Latent files under output/ and input/, newest first. "
                                       "Ignored when 'path' is set."),
                io.String.Input("path", default="", optional=True,
                                tooltip="Absolute path, or one relative to the ComfyUI root / "
                                        "output / input. Overrides the dropdown, so you can "
                                        "paste a path instead of browsing."),
                io.String.Input("key", default="", optional=True,
                                tooltip="Tensor key to read. Blank auto-detects (samples, "
                                        "latent_tensor, lr_latent, hr_latent, ...). Use it to "
                                        "pick a side of a training pair."),
            ],
            outputs=[io.Latent.Output(display_name="video")],
        )

    @classmethod
    def execute(cls, latent, path="", key="") -> io.NodeOutput:
        full = _resolve_latent_path(latent, path)
        if full is None:
            raise FileNotFoundError("latent not found: {!r}".format((path or "").strip() or latent))
        sd = safetensors.torch.load_file(full, device="cpu")
        samples, name = _extract_latent(sd, key)
        logging.info("MiniMaxH3LoadVideoLatent: %s key='%s' shape=%s",
                     os.path.basename(full), name, tuple(samples.shape))
        return io.NodeOutput({"samples": samples})

    @classmethod
    def fingerprint_inputs(cls, latent, path="", key=""):
        full = _resolve_latent_path(latent, path)
        if full is None:
            return float("nan")      # unresolvable: never cache
        st = os.stat(full)
        return "{}:{}:{}:{}".format(full, st.st_mtime_ns, st.st_size, key)

    @classmethod
    def validate_inputs(cls, latent, path="", key=""):
        # Naming 'latent' here also skips the built-in combo-membership check, so a
        # pasted path still validates when the dropdown is empty or stale.
        if _resolve_latent_path(latent, path) is None:
            return "Latent file not found: {}".format(
                (path or "").strip() or latent or "(nothing selected)")
        return True


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


class _SceneTokenProj(torch.nn.Module):
    """[N, in_ch] scene tokens -> [N, hidden]; mirrors the training-side
    SceneProj (h3-3d-adapter/scripts/train_scene3d_smoke.py). Kept fp32 and
    dtype/device-robust regardless of what the surrounding model was cast to."""

    def __init__(self, in_ch, hidden):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(in_ch, 1024), torch.nn.GELU(),
                                       torch.nn.Linear(1024, hidden))

    def forward(self, x):
        w = self.net[0].weight
        return self.net(x.to(device=w.device, dtype=w.dtype))


class MiniMaxH3SceneAdapterLoader(io.ComfyNode):
    """Attach a trained scene3d adapter (token projector + attention LoRA,
    trained by h3-3d-adapter/scripts/train_scene3d_*.py) onto the H3 DiT so
    MiniMaxH3SceneLatent conditioning has a pathway into the model.

    Research-grade: mutates the loaded diffusion model in place (LoRA is a
    forward-wrap, not a ComfyUI weight patch), so it affects every workflow
    sharing that loaded UNet until strength is set to 0 or the UNet is
    reloaded. Re-executing with a different checkpoint/strength updates the
    attached adapter rather than stacking."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SceneAdapterLoader",
            display_name="MiniMax H3 Scene Adapter Loader (experimental)",
            category="model/patch/minimax",
            description="Load a trained scene3d adapter checkpoint (proj + LoRA) onto the H3 model. In-place research patch: reload the UNet to fully remove.",
            inputs=[
                io.Model.Input("model"),
                io.String.Input("adapter_path", tooltip="Path to a scene3d adapter .pt ({proj, lora, rank, in_ch}), e.g. /weka/home-kateriw/h3-3d-adapter/checkpoints/scene3d_adapter_v2.pt"),
                io.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.05, tooltip="LoRA scale multiplier (0 = LoRA off; projector stays attached)"),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, adapter_path, strength=1.0) -> io.NodeOutput:
        ck = torch.load(adapter_path, map_location="cpu")
        dm = model.model.diffusion_model

        proj = _SceneTokenProj(ck["in_ch"], dm.hidden_size)
        proj.load_state_dict(ck["proj"])
        proj.float()
        dm.scene3d_proj = proj

        rank = ck["rank"]
        base_scale = 16.0 / rank  # alpha=16, matches training
        if not hasattr(dm, "_scene3d_lora"):
            state = {"scale": base_scale * strength}
            params = []
            for block in dm.blocks:
                for lin in (block.attn.qkv_proj, block.attn.out_proj):
                    A = torch.nn.Parameter(torch.zeros(rank, lin.in_features), requires_grad=False)
                    B = torch.nn.Parameter(torch.zeros(lin.out_features, rank), requires_grad=False)
                    orig = lin.forward

                    def fwd(x, _orig=orig, _A=A, _B=B, _state=state):
                        y = _orig(x)
                        if _state["scale"] == 0.0:
                            return y
                        if _A.device != x.device:
                            _A.data = _A.data.to(x.device)
                            _B.data = _B.data.to(x.device)
                        return y + ((x.float() @ _A.t()) @ _B.t() * _state["scale"]).to(y.dtype)

                    lin.forward = fwd
                    params += [A, B]
            dm._scene3d_lora = params
            dm._scene3d_lora_state = state
        if len(dm._scene3d_lora) != len(ck["lora"]):
            raise RuntimeError(f"adapter rank mismatch: model has {len(dm._scene3d_lora)} LoRA tensors attached, checkpoint has {len(ck['lora'])}; reload the UNet before switching ranks")
        with torch.no_grad():
            for p, saved in zip(dm._scene3d_lora, ck["lora"]):
                p.data = saved.float().to(p.device)
        dm._scene3d_lora_state["scale"] = base_scale * strength
        logging.info(f"MiniMaxH3SceneAdapterLoader: attached {adapter_path} (rank {rank}, strength {strength}, step {ck.get('step', '?')})")
        return io.NodeOutput(model)


class MiniMaxH3SceneLatent(io.ComfyNode):
    """Condition H3 on a whole room: appends a scene3d refs block carrying
    XCube scene tokens ([N, 11] = 8ch geometry latent + 3ch pooled RGB,
    exported by scene3d_eval/export_scene3d_tokens.py). Requires the scene3d
    adapter to be attached via MiniMaxH3SceneAdapterLoader -- tokens without
    the adapter raise at sampling time by design."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SceneLatent",
            display_name="MiniMax H3 Scene Latent (experimental)",
            category="model/conditioning/minimax",
            description="Append XCube scene-3D tokens (the 4th modality) to the conditioning. Pair with MiniMax H3 Scene Adapter Loader.",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input("tokens_path", tooltip="Path to a scene tokens .pt ({geom, rgb}), e.g. /weka/home-kateriw/scene3d_eval/out/hotel_scene3d_tokens.pt"),
                io.Combo.Input("placement", options=["refs", "context", "both"], default="refs",
                               tooltip="refs = distant reference material on the refs cursor chain (inside ref_decay/ref_ramp). context = world-latent-style anchor at the target's own RoPE origin, outside the ref window (spacing ignored -- the context-chain slot IS the position). both = one block each. Must match how the adapter was trained."),
                io.Float.Input("scene_strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="aug value on the refs block, same convention as ref_strength"),
                io.Float.Input("scene_spacing", default=1.0, min=0.0, max=50.0, step=0.5, tooltip="RoPE-distance separation on the refs cursor chain (refs placement only), same convention as ref_spacing"),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning, tokens_path, placement="refs", scene_strength=1.0, scene_spacing=1.0) -> io.NodeOutput:
        tok = torch.load(tokens_path, map_location="cpu")
        tokens = torch.cat([tok["geom"], tok["rgb"]], dim=1).float()
        blocks = []
        if placement in ("refs", "both"):
            blocks.append({"kind": "scene3d", "num_tokens": int(tokens.shape[0]), "tokens": tokens,
                           "spacing": scene_spacing, "aug": scene_strength})
        if placement in ("context", "both"):
            blocks.append({"kind": "scene3d", "num_tokens": int(tokens.shape[0]), "tokens": tokens,
                           "placement": "context", "spacing": 0.0, "aug": scene_strength})
        cond = node_helpers.conditioning_set_values(conditioning, {"minimax_refs": blocks}, append=True)
        logging.info(f"MiniMaxH3SceneLatent: {tokens.shape[0]} tokens x {tokens.shape[1]}ch from {tokens_path} (placement={placement})")
        return io.NodeOutput(cond)


class MiniMaxH3SceneToContextLatent(io.ComfyNode):
    """The latent-input half of the scene grounding recipe: run the adapter's
    trained SceneContextHead over the XCube scene tokens to produce a pseudo
    context latent in the video VAE's latent space, and output it as a plain
    LATENT -- wire it into `world_latent` (set world_static_time=True,
    world_frames = the head's ctx_frames) on ReferenceToVideo/VideoExtend.
    The model reads it through its native, pretrained context machinery
    (video_patch_proj), in addition to the SceneLatent conditioning tokens.

    Requires an adapter checkpoint trained WITH --context-head; older
    refs-only checkpoints have no head and raise."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SceneToContextLatent",
            display_name="MiniMax H3 Scene To Context Latent (experimental)",
            category="model/conditioning/minimax",
            description="Trained head: XCube scene tokens -> pseudo world/context latent. Plug into world_latent (world_static_time=True).",
            inputs=[
                io.String.Input("tokens_path", tooltip="Path to a scene tokens .pt ({geom, rgb})"),
                io.String.Input("adapter_path", tooltip="Adapter .pt trained with --context-head (carries the SceneContextHead weights + ctx_frames)"),
                io.Int.Input("width", default=320, min=32, max=nodes.MAX_RESOLUTION, step=32, tooltip="Must match the generation canvas exactly (world_latent contract). Head was trained at 320x192; other canvases are extrapolation."),
                io.Int.Input("height", default=192, min=32, max=nodes.MAX_RESOLUTION, step=32),
            ],
            outputs=[io.Latent.Output(display_name="world_latent")],
        )

    @classmethod
    def execute(cls, tokens_path, adapter_path, width=320, height=192) -> io.NodeOutput:
        import sys
        scripts = _SCENE_ADAPTER_REPO + "/scripts"
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from scene_context_head import SceneContextHead

        ck = torch.load(adapter_path, map_location="cpu")
        if ck.get("context_head") is None:
            raise ValueError(f"{adapter_path} has no context head -- it was trained without --context-head (the latent path). Use a v3+ adapter.")
        head = SceneContextHead(ck["in_ch"]).eval()
        head.load_state_dict(ck["context_head"])
        tok = torch.load(tokens_path, map_location="cpu")
        tokens = torch.cat([tok["geom"], tok["rgb"]], dim=1).float()
        ctx_frames = ck.get("ctx_frames", 1)
        with torch.no_grad():
            lat = head(tokens, ctx_frames, height // 16, width // 16)
        logging.info(f"MiniMaxH3SceneToContextLatent: {tokens.shape[0]} tokens -> pseudo context latent {tuple(lat.shape)} (wire world_frames={ctx_frames}, world_static_time=True)")
        return io.NodeOutput({"samples": lat})


class MiniMaxH3Extension(ComfyExtension):
    async def get_node_list(self):
        return [
            EmptyMiniMaxH3LatentAV,
            MiniMaxH3EncodeAV,
            MiniMaxH3ImageToVideo,
            MiniMaxH3ReferenceToVideo,
            MiniMaxH3VideoExtend,
            MiniMaxH3SceneFromPLY,
            MiniMaxH3RenderPLYWalkthrough,
            MiniMaxH3RenderPLYCutoutViews,
            MiniMaxH3SigmaShift,
            MiniMaxH3LatentUpsample,
            MiniMaxH3SplitAV,
            MiniMaxH3JoinAV,
            MiniMaxH3LoadVideoLatent,
            MiniMaxH3SceneAdapterLoader,
            MiniMaxH3SceneLatent,
            MiniMaxH3SceneToContextLatent
            ]


async def comfy_entrypoint() -> MiniMaxH3Extension:
    return MiniMaxH3Extension()
