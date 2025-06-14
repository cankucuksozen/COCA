from __future__ import annotations  # for type-hinting MultiObjectDataset

from torch import Tensor
from torchvision.transforms import *
from torchvision.transforms.functional import crop
from torchvision.utils import _log_api_usage_once
import numbers
from collections.abc import Sequence

import torch

from data import datasets
from data.dataset_transform import DatasetTransform


class Enlarge(DatasetTransform):
    """."""

    
    def __init__(self, dataset: datasets.MultiObjectDataset, fac: int = 4):
        super().__init__(dataset)
        h, w = dataset.height, dataset.width
        new_h, new_w = int(fac*h), int(fac*w)
        
        self._image_transform = Compose(
            [Resize((new_h, new_w), InterpolationMode.BILINEAR)]
        )
        # Nearest interpolation for the masks to keep partition of the input.
        self._mask_transform = Compose(
            [Resize((new_h, new_w), InterpolationMode.NEAREST)]
        )
    
    def transform_sample(self, sample: dict, idx: int) -> dict:
        sample["image"] = self._transform_image(sample["image"])
        sample["mask"] = self._transform_mask(sample["mask"])

        # Update visibility and num actual objects after changing masks.
        sample["visibility"] = (
            (sample["mask"].sum(dim=(1, 2, 3)) > 0).float().unsqueeze(1)
        )
        sample["num_actual_objects"] = (
            sample["visibility"].sum().long() - self.dataset.num_background_objects
        )
        return sample

    def _transform_image(self, image: Tensor) -> Tensor:
        return self._image_transform(image)

    def _transform_mask(self, mask: Tensor) -> Tensor:
        out = self._mask_transform(mask.flatten(0, 1))
        out = out.view(mask.size(0),mask.size(1),out.size(-2),out.size(-1))
        return out



def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

