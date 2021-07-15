from typing import List

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.ops import nms


def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2):
    TEXT_COLOR = (255, 255, 255)  # White

    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
    )
    cv2.rectangle(
        img,
        (x_min, y_min - int(1.3 * text_height)),
        (x_min + text_width, y_min),
        color,
        -1,
    )
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids):
    category_id_to_name = {0: "positive", 1: "hard negative"}

    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)

    if len(bboxes) > 0:
        plt.figure(figsize=(12, 12))
        plt.axis("off")
        plt.imshow(img)
        plt.show()


def visualize_detections(img, pred_bboxes, pred_labels, target_bboxes, target_labels):
    category_id_to_name = {1: "positive", 2: "hard negative"}

    # Convert to numpy
    img = img.cpu().numpy().transpose(1, 2, 0)
    pred_bboxes = pred_bboxes.cpu().numpy()
    target_bboxes = target_bboxes.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()
    target_labels = target_labels.cpu().numpy()

    img = img.copy()

    # Add predicted boxes
    for bbox, category_id in zip(pred_bboxes, pred_labels):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name, color=(1, 0, 0))

    # Add target boxes
    for bbox, category_id in zip(target_bboxes, target_labels):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name, color=(0, 0, 1))

    return torch.tensor(img).permute(2, 0, 1)


def split_tiles(
    tiles: torch.Tensor,
    tile_size: List[int],
    patch_size: int = 300,
    output_size: int = 250,
    overlap: int = 20,
):
    # tiles: torch.Size([1, 3, 1000, 1000])

    # Compute the locations of outputs, patches, and padding
    _, patch_boxes, padding = compute_boxes_and_padding(
        tile_size, patch_size, output_size, overlap
    )
    pad_top, pad_bot = padding["top"], padding["bot"]
    pad_left, pad_right = padding["left"], padding["right"]

    # Do the necessary padding
    # NOTE: it is necessary to have float for padding
    padded_tiles = F.pad(
        tiles.float(), (pad_left, pad_right, pad_top, pad_bot), mode="reflect"
    )

    # Converting back to the real type
    padded_tiles = padded_tiles.to(dtype=tiles.dtype)

    # Extract the patches
    patches = torch.zeros(
        (len(tiles), len(patch_boxes), 3, patch_size, patch_size),
        dtype=tiles.dtype,
        device=tiles.device,
    )
    pad_top = 0 - patch_boxes[:, 1].min().item()
    pad_left = 0 - patch_boxes[:, 0].min().item()
    for idx, patch_box in enumerate(patch_boxes):
        # put patch_boxes back in proper coordinates
        x, y, h, w = patch_box[0], patch_box[1], patch_box[2], patch_box[3]
        x, y = x + pad_left, y + pad_top

        # Extract patch from padded tile
        patches[:, idx] = padded_tiles[:, :, y : y + h, x : x + w]
    # patches: torch.Size([24, 3, 300, 300])

    return patches


def compute_boxes_and_padding(
    tile_size: List[int], patch_size: int, output_size: int, overlap: int
):
    # Tile size
    th, tw = tile_size
    # th, tw: (1200, 800)

    # Calculate output stride
    output_stride = output_size - overlap
    # output_stride: 230

    # Accordingly calculate output grid
    output_x = torch.arange(0, tw, step=output_stride)
    output_y = torch.arange(0, th, step=output_stride)
    # output_x: tensor([   0,  230,  460,  690,  920, 1150])
    # output_y: tensor([  0, 230, 460, 690])

    # Compute output boxes from top lefts
    grid = torch.meshgrid(output_y, output_x)
    output_top_lefts = torch.cat(
        [grid[1].flatten().unsqueeze(1), grid[0].flatten().unsqueeze(1)], dim=1
    )
    # output_top_lefts: torch.Size([24, 2])

    # Format boxes as x, y, h, w
    output_boxes = torch.zeros((len(output_top_lefts), 4), dtype=torch.int64)
    output_boxes[:, :2] = output_top_lefts
    output_boxes[:, 2:] = output_size
    # output_boxes: torch.Size([24, 4])

    # Corresponding input grid
    # Assumes: patch_size > output_size
    output_patch_border = (patch_size - output_size) // 2
    # output_patch_border: 25

    # Make similar patch_boxes
    patch_boxes = torch.zeros_like(output_boxes)
    patch_boxes[:, :2] = output_boxes[:, :2] - output_patch_border
    patch_boxes[:, 2:] = patch_size
    # patch_boxes: torch.Size([24, 4])

    # Compute necessary padding for sampling patches from
    pad_top = pad_left = output_patch_border
    pad_right = int(patch_boxes[:, 0].max().item() + patch_size - tw)
    pad_bot = int(patch_boxes[:, 1].max().item() + patch_size - th)
    # pad_left, pad_top, pad_right, pad_bot: (25, 25, 225, 165)

    # Compute padded tile size of output
    output_padded_h = int(output_boxes[:, 1].max().item() + output_size)
    output_padded_w = int(output_boxes[:, 0].max().item() + output_size)

    # Store all kinds of sizes for debugging as well
    padding = {
        "top": pad_top,  # Padding
        "bot": pad_bot,
        "left": pad_left,
        "right": pad_right,
        "th": th,  # tile size
        "tw": tw,
        "tph": th + pad_top + pad_bot,  # padded tile size
        "tpw": tw + pad_left + pad_right,
        "oh": th,  # output_tile size
        "ow": tw,
        "oph": output_padded_h,  # padded output_tile size
        "opw": output_padded_w,
    }

    return output_boxes, patch_boxes, padding


def stitch_boxes(
    boxes: List[torch.Tensor],
    tile_size: List[int],
    patch_size: int = 300,
    output_size: int = 250,
    overlap: int = 20,
    iou_threshold: float = 0.5,
):

    # Compute the locations of outputs, patches, and padding
    output_boxes, _, _ = compute_boxes_and_padding(
        tile_size=tile_size,
        patch_size=patch_size,
        output_size=output_size,
        overlap=overlap,
    )

    output_boxes = output_boxes.to(boxes[0].device)

    # Return as list
    all_boxes = []

    # For each image in the batch
    for batch_boxes in boxes:
        tile_boxes = []
        for idx, output_box in enumerate(output_boxes):
            # Choose boxes from this patch
            patch_boxes = batch_boxes[batch_boxes[:, -1] == idx]

            # Convert to x1, y1, h, w
            patch_boxes[:, 2:4] = patch_boxes[:, 2:4] - patch_boxes[:, :2]

            # Add offset of output_box
            patch_boxes[:, :2] = patch_boxes[:, :2] + output_box[:2]
            # patch_boxes[:, 0] = patch_boxes[:, 0] + output_box[1]
            # patch_boxes[:, 1] = patch_boxes[:, 1] + output_box[0]

            # Discard batch info, keep score info ( xyhw,conf )
            patch_boxes = patch_boxes[:, :-1]

            # Aggregate
            tile_boxes.append(patch_boxes)

        # Concatenate into one list
        tile_boxes = torch.cat(tile_boxes, dim=0)

        # Remove boxes with centres outside tile
        tile_centres = tile_boxes[:, :2] + tile_boxes[:, 2:4] / 2
        inside = (
            (tile_centres[:, 0] < tile_size[1])
            & (tile_centres[:, 1] < tile_size[0])
            & (tile_centres[:, 0] > 0)
            & (tile_centres[:, 1] > 0)
        )

        tile_boxes = tile_boxes[inside]

        # Apply nms
        # Format to x1, y1, x2, y2
        tile_boxes[:, 2:4] = tile_boxes[:, :2] + tile_boxes[:, 2:4]

        # Apply nms, tile_boxes - x1, y1, x2, y2, score
        scores = tile_boxes[:, 4]
        nms_sel = nms(tile_boxes[:, :4], scores, iou_threshold=iou_threshold)
        nms_boxes = tile_boxes[nms_sel]

        # Take only the nms boxes
        all_boxes.append(nms_boxes)

    return all_boxes
