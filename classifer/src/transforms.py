import random

import cv2
from albumentations.augmentations.crops.transforms import _BaseRandomSizedCrop


class MyRandomSizedCrop(_BaseRandomSizedCrop):
    """Crop a random part of the input and rescale it to some size.
    Args:
        min_max_height_width ((int, int)): crop size limits.
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        min_max_height_width,
        height,
        width,
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=1.0,
    ):
        super(MyRandomSizedCrop, self).__init__(
            height=height,
            width=width,
            interpolation=interpolation,
            always_apply=always_apply,
            p=p,
        )
        self.min_max_height_width = min_max_height_width

    def get_params(self):
        crop_height = random.randint(
            self.min_max_height_width[0], self.min_max_height_width[1]
        )
        crop_width = random.randint(
            self.min_max_height_width[0], self.min_max_height_width[1]
        )

        return {
            "h_start": random.random(),
            "w_start": random.random(),
            "crop_height": crop_height,
            "crop_width": crop_width,
        }

    def get_transform_init_args_names(self):
        return "min_max_height_width", "height", "width", "interpolation"
