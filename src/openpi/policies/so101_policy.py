import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        # Convert CHW -> HWC if needed
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """
    Converts SO101 LeRobot inputs to OpenPI model inputs.

    Expected (after repack):
    - observation/image: third-person image (front)
    - observation/wrist_image: auxiliary image (above)
    - observation/state: [6] (5 joints + 1 gripper)
    - actions: [H, 6] (optional during training)
    - prompt: str (if present or injected via PromptFromLeRobotTask)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])  # front view
        aux_image = _parse_image(data["observation/wrist_image"])  # above view (used as wrist proxy)

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": aux_image,
                # No right wrist camera; pad with zeros of same shape as base image
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask padding image only for non-FAST models
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Slices model actions back to SO101's 6-dim control."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :6])}
    