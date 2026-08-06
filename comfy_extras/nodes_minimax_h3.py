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

import torch
import torchaudio

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


def _context_keyframes(context_latent, context_frames, context_strength=1.0):
    """Build context/context_audio keyframe dicts continuing a prior generation's AV
    latent, plus the (width, height) it implies. Shared by MiniMaxH3ImageToVideo and
    MiniMaxH3VideoExtend."""
    ctx_samples = context_latent["samples"]
    # accept either a full AV NestedTensor (from an H3 sampler output) or a
    # plain video-only latent (e.g. a straight VAEEncode of external footage)
    is_av = ctx_samples.is_nested
    ctx_video = ctx_samples.tensors[0] if is_av else ctx_samples
    ctx_audio = ctx_samples.tensors[1] if is_av else None
    if ctx_video.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    ctx_t, ctx_h, ctx_w = ctx_video.shape[2], ctx_video.shape[3], ctx_video.shape[4]
    n_frames = min(context_frames, ctx_t)
    width, height = ctx_w * 16, ctx_h * 16  # inherit the source clip's canvas exactly

    keyframes = [{"kind": "context", "num_frames": n_frames, "aug": context_strength,
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
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, length,
                first_frame=None, last_frame=None,
                context_latent=None, context_frames=2, context_strength=1.0) -> io.NodeOutput:
        keyframes = []
        if context_latent is not None:
            width, height, keyframes = _context_keyframes(context_latent, context_frames, context_strength)

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

        if keyframes:
            for kf in keyframes:
                if "image" in kf:
                    kf["latent"] = vae.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": keyframes, "minimax_frame_count": frame_count})
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
                *_REF_INPUTS,
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length, ref_image_size="match", ref_spacing=1.0,
                ref_strength=1.0, ref_decay=0.0, ref_ramp=0.0, ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items, ref_blocks = _build_ref_blocks(vae, audio_vae, width, height, frame_count, ref_image_size,
                                                  ref_images, ref_videos, ref_video_audios, ref_audios,
                                                  ref_spacing, ref_strength)

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks,
                "minimax_ref_decay": ref_decay, "minimax_ref_ramp": ref_ramp})
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
                io.Float.Input("context_strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="1.0 = context frames stay clean/hard-pinned; lower blends in noise for a softer handoff"),
                *_REF_INPUTS,
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, context_latent, prompt, length, context_frames=2, context_strength=1.0,
                audio_vae=None, ref_image_size="match", ref_spacing=1.0, ref_strength=1.0,
                ref_decay=0.0, ref_ramp=0.0, ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None) -> io.NodeOutput:
        width, height, keyframes = _context_keyframes(context_latent, context_frames, context_strength)
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
            MiniMaxH3EncodeAV,
            MiniMaxH3ImageToVideo,
            MiniMaxH3ReferenceToVideo,
            MiniMaxH3VideoExtend,
            MiniMaxH3SceneFromPLY,
            MiniMaxH3RenderPLYWalkthrough,
            MiniMaxH3SigmaShift
            ]


async def comfy_entrypoint() -> MiniMaxH3Extension:
    return MiniMaxH3Extension()
