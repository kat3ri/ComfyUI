"""MiniMax H3 flow-matching latent upscaler (v3a) as a ComfyUI node.

Unlike the v1/v2 regression upscaler, this model is generative: it integrates a
learned velocity field from noise to the HR latent, conditioned on the LR latent.
That needs a timestep pathway and a sampler, neither of which the stock
LatentUpscaleModelLoader knows about -- hence a dedicated node.

More steps is not automatically better here: on high-detail content the extra
Euler accuracy converges onto a smoother answer, so 4-8 steps often looks better
than 32. Worth sweeping rather than maxing.

Checkpoints are read from models/h3_flow_upscalers/.
"""

import json
import sys

import torch

import comfy.model_management
import comfy.nested_tensor
import folder_paths
from comfy_api.latest import ComfyExtension, io
from safetensors import safe_open

REPO = "/weka/home-kateriw/h3-latent-upscaler"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

folder_paths.add_model_folder_path(
    "h3_flow_upscalers", str(folder_paths.models_dir) + "/h3_flow_upscalers")

_CACHE = {}


def _load(name):
    if name in _CACHE:
        return _CACHE[name]
    import inspect

    from model_flow import FlowUpsampler

    path = folder_paths.get_full_path_or_raise("h3_flow_upscalers", name)
    with safe_open(path, framework="pt", device="cpu") as f:
        cfg = json.loads(f.metadata()["config"])
        sd = {k: f.get_tensor(k).float() for k in f.keys()}
    allowed = set(inspect.signature(FlowUpsampler.__init__).parameters) - {"self"}
    model = FlowUpsampler(**{k: v for k, v in cfg.items() if k in allowed})
    model.load_state_dict(sd)
    model = model.to(comfy.model_management.get_torch_device()).eval()
    _CACHE.clear()  # only keep one resident
    _CACHE[name] = (model, cfg)
    return _CACHE[name]


class MiniMaxH3FlowUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FlowUpscale",
            display_name="MiniMax H3 Flow Latent Upscale (2x)",
            category="model/latent/minimax",
            is_experimental=True,
            description="Generative 2x latent upscale for H3 (flow matching). Audio passes "
                        "through. Try 4-8 steps first: more steps is not always sharper.",
            inputs=[
                io.Latent.Input("samples"),
                io.Combo.Input("model_name",
                               options=folder_paths.get_filename_list("h3_flow_upscalers")),
                io.Int.Input("steps", default=8, min=1, max=64),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFF,
                             control_after_generate=True),
                io.Boolean.Input("one_step_dmd", default=False,
                                 tooltip="For DMD-distilled checkpoints: single Euler step "
                                         "from noise (ignores `steps`)."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, samples, model_name, steps=8, seed=0, one_step_dmd=False) -> io.NodeOutput:
        from model_flow import sample, upsample_cond

        device = comfy.model_management.get_torch_device()
        model, _cfg = _load(model_name)

        latents = samples["samples"]
        audio = None
        if latents.is_nested:
            video, audio = latents.tensors
        else:
            video = latents
        if video.ndim != 5 or video.shape[1] != 24:
            raise ValueError(f"expected an H3 video latent [B,24,T,H,W], got {tuple(video.shape)}")

        in_dtype = video.dtype
        lr = video.float().to(device)
        gen = torch.Generator(device=device).manual_seed(int(seed))

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if one_step_dmd:
                b, c, t, h, w = lr.shape
                cond = upsample_cond(lr, (h * 2, w * 2))
                noise = torch.randn((b, c, t, h * 2, w * 2), device=device, generator=gen)
                out = noise + model(noise, torch.zeros(b, device=device), cond)
            else:
                out = sample(model, lr, steps=int(steps), device=device, generator=gen)

        out = out.to(dtype=in_dtype, device=comfy.model_management.intermediate_device())
        res = samples.copy()
        res["samples"] = comfy.nested_tensor.NestedTensor((out, audio)) if audio is not None else out
        res.pop("noise_mask", None)
        return io.NodeOutput(res)


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3FlowUpscale]


async def comfy_entrypoint() -> Extension:
    return Extension()
