import os
import sys
import gc
import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.cm as cm

from PIL import Image

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file

from .video_depth_anything.video_depth import VideoDepthAnything

class VideoDepthAnythingLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("videodepthanything"), {"tooltip": "These models are loaded from 'ComfyUI/models/videodepthanything'"}),
                "config_name": (["vits", "vitl"], {"tooltip": "choose according to the model: vits=small, vitl=large"})
            }
        }

    RETURN_TYPES = ("VDAMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loadmodel"
    CATEGORY = "VideoDepthAnything"
    DESCRIPTION = "Load a VideoDepthAnything Model from ComfyUI/models/videodepthanything"

    def loadmodel(self, model_name, config_name):
        model_path = folder_paths.get_full_path("videodepthanything", model_name)

        # safe_load might need to be false because of pickle.
        # state = load_torch_file(model_path, safe_load=True)
        state = torch.load(model_path, map_location='cpu')

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        vda = VideoDepthAnything(**model_configs[config_name])

        vda.load_state_dict(state, strict=True)
        return (vda,)

class VideoDepthAnythingProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("VDAMODEL",),
                "frames": ("IMAGE",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 100, "tooltip": "frames-per-second of the video"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depths",)
    FUNCTION = "process"
    CATEGORY = "VideoDepthAnything"
    DESCRIPTION = "Run VideoDepthAnything on the provided input image"

    def process(self, model, frames, fps):
        # VideoDepthAnything expects uint8 pixeldata
        if frames.dtype == torch.float32 and frames.max() <= 1.0:
            frames = (frames * 255).clamp(0, 255).to(torch.uint8)

        # clear memory before sampling
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()

        device = mm.get_torch_device()
        model = model.to(device).eval() 

        # convert video list[Image.Image] into frames, fps
        frames = frames.cpu().numpy()

        # 3. run inference
        input_size = 518 # copied from run.py TODO not sure what this is
        depths, fps = model.infer_video_depth(frames, fps, input_size=input_size, device=device.type)

        # 4. transform back into image list and return
        return (into_frames_list(depths, fps),)

# adapted from dc_utils/save_video
def into_frames_list(frames, fps, grayscale=False):
    colormap = np.array(cm.get_cmap("inferno").colors, dtype=np.float32)  # shape (256, 3)
    d_min, d_max = frames.min(), frames.max()
    depth_norm = (frames - d_min) / (d_max - d_min)  # shape (N, H, W), float32 ∈ [0,1]

    if grayscale:
        depth_rgb = np.stack([depth_norm]*3, axis=-1)  # (N, H, W, 3)
    else:
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        depth_rgb = colormap[depth_uint8]  # (N, H, W, 3), float32 ∈ [0,1]

    return torch.from_numpy(depth_rgb).float()  # (N, H, W, 3)

NODE_CLASS_MAPPINGS = {
    "VideoDepthAnythingLoader": VideoDepthAnythingLoader,
    "VideoDepthAnythingProcess": VideoDepthAnythingProcess,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoDepthAnythingLoader": "VideoDepthAnything Model Loader",
    "VideoDepthAnythingProcess": "VideoDepthAnything Processor",
}
