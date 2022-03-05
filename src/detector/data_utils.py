import random

from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox
from albumentations.core.transforms_interface import DualTransform


def crop(img, x_min, y_min, x_max, y_max):
    height, width = img.shape[:2]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
            )
        )

    if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
        raise ValueError(
            "Values for crop should be non negative and equal or smaller than image sizes"
            "(x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max}, "
            "height = {height}, width = {width})".format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                height=height,
                width=width,
            )
        )

    return img[int(y_min) : int(y_max), int(x_min) : int(x_max)]


def bbox_crop(bbox, x_min, y_min, x_max, y_max, rows, cols):
    """Crop a bounding box.
    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        x_min (int):
        y_min (int):
        x_max (int):
        y_max (int):
        rows (int): Image rows.
        cols (int): Image cols.
    Returns:
        tuple: A cropped bounding box `(x_min, y_min, x_max, y_max)`.
    """
    crop_coords = x_min, y_min, x_max, y_max
    crop_height = y_max - y_min
    crop_width = x_max - x_min
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)


def crop_bbox_by_coords(
    bbox,
    crop_coords,
    crop_height,
    crop_width,
    rows,
    cols,
):
    """Crop a bounding box using the provided coordinates of bottom-left and top-right corners in pixels and the
    required height and width of the crop.
    Args:
        bbox (tuple): A cropped box `(x_min, y_min, x_max, y_max)`.
        crop_coords (tuple): Crop coordinates `(x1, y1, x2, y2)`.
        crop_height (int):
        crop_width (int):
        rows (int): Image rows.
        cols (int): Image cols.
    Returns:
        tuple: A cropped bounding box `(x_min, y_min, x_max, y_max)`.
    """
    bbox = denormalize_bbox(bbox, rows, cols)
    x_min, y_min, x_max, y_max = bbox[:4]
    x1, y1, _, _ = crop_coords
    cropped_bbox = x_min - x1, y_min - y1, x_max - x1, y_max - y1
    return normalize_bbox(cropped_bbox, crop_height, crop_width)


class RandomCropIncludeBBox(DualTransform):
    """Crop bbox from image with random shift by x,y coordinates
    Args:
        max_part_shift (float): float value in (0.0, 1.0) range. Default 0.3
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, height, width, always_apply=False, p=1.0):
        super(RandomCropIncludeBBox, self).__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return crop(img, x_min, y_min, x_max, y_max)

    def get_params_dependent_on_targets(self, params):
        bbox = random.choice(params["bboxes"])
        max_h, max_w = params["image"].shape[:2]
        x1, y1, x2, y2 = (
            bbox[0] * max_w,
            bbox[1] * max_h,
            bbox[2] * max_w,
            bbox[3] * max_h,
        )

        # Get center point of box
        y_center = (y1 + y2) // 2
        x_center = (x1 + x2) // 2

        # Randomly crop around box
        if x_center < self.width:
            x_shift = random.randint(0, x_center)  # Do not have x_min < 0
        elif max_w - x_center < self.width:
            x_shift = random.randint(
                self.width - (max_w - x_center), self.width  # Do not have x_max > max_w
            )
        else:
            x_shift = random.randint(0, self.width)

        x_min = x_center - x_shift
        x_max = x_center + (self.width - x_shift)

        if y_center < self.height:
            y_shift = random.randint(0, y_center)  # Do not have y_min < 0
        elif max_h - y_center < self.height:
            y_shift = random.randint(
                self.height - (max_h - y_center),
                self.height,  # Do not have y_max > max_h
            )
        else:
            y_shift = random.randint(0, self.height)

        y_min = y_center - y_shift
        y_max = y_center + (self.height - y_shift)

        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def apply_to_bbox(self, bbox, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return bbox_crop(bbox, x_min, y_min, x_max, y_max, **params)

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return ("height", "width")
