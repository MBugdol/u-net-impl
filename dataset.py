"""Contains definitions used for the Camouflaged Animals dataset."""

import os
import torch
from pathlib import Path
from torch import nn, Tensor
from torchvision.io import read_image, ImageReadMode
# import torchvision.transforms as T


def _get_filenames_from_path(path: str):
    return [Path(file).stem for file in os.listdir(path)]


def _get_image_path_from_filename(filename: str, directory: str = "./"):
    allowed_extensions = (".png", ".jpg", ".jpeg", ".webp")
    matching_files = Path(directory).glob(filename + ".*")
    first_image_match = next(
        (file for file in matching_files if file.suffix.lower() in allowed_extensions),
        None,
    )
    return first_image_match


class CamouflagedAnimalsDataset(nn.Module):
    def __init__(
        self,
        images_path: str,
        masks_path: str,
        common_transform=None,
        image_transform=None,
        mask_transform=None,
    ):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path
        self.common_transform = common_transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.filenames = _get_filenames_from_path(self.images_path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        filename = self.filenames[idx]

        image_path = _get_image_path_from_filename(filename, self.images_path)
        if image_path is None:
            raise FileNotFoundError(
                f"Can't find matching image file for filename {self.filenames[idx]}"
            )
        mask_path = _get_image_path_from_filename(filename, self.masks_path)
        if mask_path is None:
            raise FileNotFoundError(
                f"Can't find matching mask file for filename {self.filenames[idx]}"
            )

        image = read_image(image_path, ImageReadMode.RGB)
        mask = read_image(mask_path, ImageReadMode.RGB)

        if self.common_transform:
            state = torch.get_rng_state()
            image = self.common_transform(image)
            torch.set_rng_state(state)
            mask = self.common_transform(mask)
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


def getColors() -> list[tuple[int, int, int]]:
    return [
        (255, 0, 0),  # background
        (0, 255, 0),  # masking background
        (0, 0, 255),  # animal
        (255, 255, 255),  # attention foreground
    ]


def colorMaskToOneHot(rgbmask: Tensor) -> Tensor:
    object_colors = torch.tensor(getColors(), device=rgbmask.device)
    # check rgbmask type - if float handle differently
    if rgbmask.dtype in [torch.float, torch.double]:
        object_colors = object_colors.type(rgbmask.dtype) / 255.0

    one_hot_mask = torch.zeros(
        object_colors.size(0), *rgbmask.shape[1:], dtype=torch.bool, device=rgbmask.device
    )

    for i, color in enumerate(object_colors):
        pass
        one_hot_mask[i] = (rgbmask == color.view(3, 1, 1)).all(dim=0)

    return one_hot_mask
