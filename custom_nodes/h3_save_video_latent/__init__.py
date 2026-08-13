"""Save the video half of an H3 AV latent to safetensors.

Core's SaveLatent can't serialize H3's nested (video, audio) tensor, and the
latent upscaler's eval set needs exactly the thing a real workflow produces:
the sampler's own output latent. Drop this before the decode in any H3 workflow
to collect gen-latent val data.
"""

import json
import os

import folder_paths
from comfy_api.latest import ComfyExtension, io
from safetensors.torch import save_file


class MiniMaxH3SaveVideoLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SaveVideoLatent",
            display_name="MiniMax H3 Save Video Latent",
            category="model/latent/minimax",
            description="Save the video stream of an H3 AV latent as fp16 safetensors "
                        "(for latent-upscaler eval sets). Audio is dropped.",
            inputs=[
                io.Latent.Input("samples"),
                io.String.Input("filename_prefix", default="h3_gen_latents/gen"),
                io.String.Input("note", default="", tooltip="free-text tag stored in metadata "
                                                            "(e.g. 't2v base 20step')"),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, samples, filename_prefix="h3_gen_latents/gen", note="") -> io.NodeOutput:
        latents = samples["samples"]
        video = latents.tensors[0] if latents.is_nested else latents

        full_path, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        os.makedirs(full_path, exist_ok=True)
        out = os.path.join(full_path, f"{filename}_{counter:05}_.safetensors")

        save_file({"samples": video.contiguous().to("cpu").half()}, out,
                  metadata={"meta": json.dumps({"note": note, "shape": list(video.shape),
                                                "nested": bool(latents.is_nested)})})
        return io.NodeOutput(ui={"text": [f"saved {out}"]})


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3SaveVideoLatent]


async def comfy_entrypoint() -> Extension:
    return Extension()
