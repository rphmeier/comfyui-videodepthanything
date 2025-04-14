from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
import folder_paths
import os

if "videodepthanything" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["videodepthanything"] = ([os.path.join(folder_paths.models_dir, "videodepthanything")], [".pth"])

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
