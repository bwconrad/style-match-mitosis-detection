import cv2
import matplotlib.pyplot as plt
import torch

BOX_COLOR = (255, 0, 0)  # Red
BOX_COLOR2 = (0, 0, 255)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
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
