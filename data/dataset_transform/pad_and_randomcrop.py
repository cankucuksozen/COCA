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


class PadAndRandomCrop(DatasetTransform):
    """Pads and crops an image and masks."""

    
    def __init__(self, dataset: datasets.MultiObjectDataset, pad: int = 2):
        super().__init__(dataset)
        self.pad = pad
        h, w = dataset.height, dataset.width
        new_h, new_w = int(h+2*pad), int(w+2*pad)
        
        i = torch.randint(0, new_h - h + 1, size=(1,)).item()
        j = torch.randint(0, new_w - w + 1, size=(1,)).item()
                
        pad_op = Pad(pad, padding_mode='edge')
        
        crop_op = MyCrop(i,j,(h,w))
        
        self._transform = Compose(
            [pad_op, crop_op]
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
        return self._transform(image)

    def _transform_mask(self, mask: Tensor) -> Tensor:
        out = self._transform(mask.flatten(0, 1))
        out = out.view_as(mask)
        return out


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

class MyCrop(torch.nn.Module):
    
    def __init__(self, top, left, size):
        super().__init__()
        _log_api_usage_once(self)

        self.top = top
        self.left = left
        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        
        return crop(img, self.top, self.left, self.size[0], self.size[1])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"
    
    
