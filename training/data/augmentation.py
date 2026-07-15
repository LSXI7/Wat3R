# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict
from torchvision import transforms


def get_image_augmentation(
        color_jitter: Optional[Dict[str, float]] = None,
        gray_scale: bool = True,
        gau_blur: bool = False
) -> Optional[transforms.Compose]:
    """Create a composition of image augmentations.

    Args:
        color_jitter: Dictionary containing color jitter parameters:
            - brightness: float (default: 0.5)
            - contrast: float (default: 0.5)
            - saturation: float (default: 0.5)
            - hue: float (default: 0.1)
            - p: probability of applying (default: 0.9)
            If None, uses default values
        gray_scale: Whether to apply random grayscale (default: True)
        gau_blur: Whether to apply gaussian blur (default: False)

    Returns:
        A Compose object of transforms or None if no transforms are added
    """
    transform_list = []
    default_jitter = {
        "brightness": 0.5,
        "contrast": 0.5,
        "saturation": 0.5,
        "hue": 0.1,
        "p": 0.9
    }
    default_gray_scale = {
        'apply': True,
        'p': 0.05
    }
    default_gau_blur = {
        'apply': False,
        'p': 0.05
    }

    # Handle color jitter
    if color_jitter is not None:
        # Merge with defaults for missing keys
        effective_jitter = {**default_jitter, **color_jitter}
    else:
        effective_jitter = default_jitter

    # Handle color jitter
    if gray_scale is not None:
        # Merge with defaults for missing keys
        effective_gray_scale = {**default_gray_scale, **gray_scale}
    else:
        effective_gray_scale = default_gray_scale

    if gau_blur is not None:
        # Merge with defaults for missing keys
        effective_gau_blur = {**default_gau_blur, **gau_blur}
    else:
        effective_gau_blur = default_gau_blur

    transform_list.append(
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=effective_jitter["brightness"],
                    contrast=effective_jitter["contrast"],
                    saturation=effective_jitter["saturation"],
                    hue=effective_jitter["hue"],
                )
            ],
            p=effective_jitter["p"],
        )
    )

    if effective_gray_scale['apply']:
        transform_list.append(transforms.RandomGrayscale(p=effective_gray_scale['p']))

    if effective_gau_blur['apply']:
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=effective_gau_blur['p']
            )
        )

    return transforms.Compose(transform_list) if transform_list else None
