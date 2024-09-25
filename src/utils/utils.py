from typing import List
from pathlib import Path
import os
import random

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


class SingleImageDataset(Dataset):
    def __init__(self, image, transform=None):
        self.image = image
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = self.image
        if self.transform:
            img = self.transform(img)
        return img, 0


def wiener2(
    image: np.ndarray, local_mean_filter_size: int = 3, noise_var: float = None
) -> np.ndarray:
    """
    Applies a Wiener filter to an input image, used for noise reduction and image enhancement.

    This function computes the local mean and variance of the input image and then applies the Wiener filter.
    If noise variance is not provided, it is estimated from the input image. The local mean and local variance
    are computed using a convolution with a kernel of ones, divided by the square of the local_mean_filter_size.

    Args:
        image (np.ndarray): 2D input array representing the grayscale image to be filtered.
        local_mean_filter_size (int, optional): The size of the local mean filter. Default is 3.
        noise_var (float, optional): The noise variance. If None, it will be estimated from the input image.

    Returns:
        np.ndarray: The filtered image, with reduced noise.

    Example:
        >>> import numpy as np
        >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        >>> wiener2(img, local_mean_filter_size=3)
    """

    local_mean_kernel = np.ones(
        (local_mean_filter_size, local_mean_filter_size), dtype=np.float32
    )
    local_mean_kernel /= local_mean_filter_size**2

    # Compute local mean of image
    local_mean = cv2.filter2D(image, -1, local_mean_kernel)

    # Compute local variance of image
    local_var = cv2.filter2D(image**2, -1, local_mean_kernel) - local_mean**2

    # Estimate the noise variance if not provided
    if noise_var is None:
        noise_var = np.var(image - local_mean)

    # Compute result image (Wiener Filter)
    result = local_mean + (
        np.maximum(local_var - noise_var, 0) / (local_var + 1e-10)
    ) * (image - local_mean)

    return result


def find_images_in_path(path: str) -> List:
    """Find all images in path

    Args:
        path: path to folder with images

    Returns:
        List: list of paths to images
    """

    path = Path(path)
    img_ext = [
        ".jpg",
        ".JPG",
        ".jpeg",
        ".JPEG",
        ".png",
        ".PNG",
        ".webp",
        ".WEBP",
        ".tif",
        ".TIF",
        ".tiff",
        ".TIFF",
        ".bmp",
        ".BMP",
    ]
    paths_img = []

    for ext in img_ext:
        paths_img.extend(list(path.glob(f"**/*{ext}")))
    return paths_img


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Setting all seeds to be {seed} to reproduce...")
