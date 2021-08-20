import os
from itertools import cycle
from typing import List

import pytorch_lightning as pl
import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
from torchvision.ops import nms
from torchvision.utils import make_grid, save_image

from .metrics import iou_and_acc, mean_average_precision
from .style_transfer.model import AdaInModel
from .utils import split_tiles, stitch_boxes, visualize_detections


class DetectionModel(pl.LightningModule):
    def __init__(
        self,
        arch: str = "faster_rcnn",
        n_classes: int = 2,
        optimizer: str = "sgd",
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        schedule: str = 'none',
        steps: List[int] = [50],
        gamma: float = 0.1,
        n_samples: int = 10,
        crop_size: int = 256,
        overlap: int = 32,
        nms_threshold: float = 0.2,
        score_threshold: float = 0.8,
        style_checkpoint: str = None,
        eval_only_positives: bool = False,
    ):
        """Midog detection model

        Args:
            arch: Detection model architecture (faster_rcnn | retinanet)
            n_classes: Number of classes
            optimizer: Name of optimizer (sgd | adam)
            lr: Learning rate
            momentum: Momentum for SGD
            weight_decay: Weight decay
            schedule: Learning rate schedule (none | step)
            steps: Step schedule reduction epochs
            gamma: Step schedule reduction factor
            n_samples: Number of validation samples to save detection visualizations of
            crop_size: Size of image patches
            overlap: Number of overlapping pixels when patching image during testing
            nms_threshold: Threshold for non-max supression
            score_threshold: Box score threshold during inference
            style_checkpoint: Checkpoint to style transfer model
            eval_only_positives: only evaluate on positives (ignore hard negatives)
        """
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.momentum = momentum
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.schedule = schedule
        self.steps = steps
        self.gamma = gamma
        self.n_samples = n_samples
        self.crop_size = crop_size
        self.overlap = overlap
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.eval_only_positives = eval_only_positives

        if arch == "faster_rcnn":
            self.net = fasterrcnn_resnet50_fpn(
                pretrained=True, pretrained_backbone=True
            )
            in_features = self.net.roi_heads.box_predictor.cls_score.in_features
            head = FastRCNNPredictor(in_features, n_classes + 1)
            self.net.roi_heads.box_predictor = head
        elif arch == "retinanet":
            self.net = retinanet_resnet50_fpn(pretrained=True, pretrained_backbone=True)
            self.net.head = RetinaNetHead(
                in_channels=self.net.backbone.out_channels,
                num_anchors=self.net.head.classification_head.num_anchors,
                num_classes=n_classes + 1,
            )
        else:
            raise NotImplementedError(f"{arch} is not an available architecture")

        if style_checkpoint:
            self.style_net = AdaInModel().load_from_checkpoint(style_checkpoint)
            self.style_net.freeze()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        if hasattr(self, "style_net"):
            imgs, targets, imgs_s = batch
            imgs, _, _ = self.style_net(imgs, imgs_s)
        else:
            imgs, targets = batch

        # Pass through model
        loss_dict = self.net(imgs, targets)
        loss = sum(loss_dict.values())

        # Log
        self.log_dict(
            {f"{k}": v for k, v in loss_dict.items()},
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        if hasattr(self, "style_net"):
            imgs, targets, imgs_s = batch
            imgs, _, _ = self.style_net(imgs, imgs_s)
        else:
            imgs, targets = batch

        # Pass through model
        out = self(imgs)

        map = []
        accs = []
        ious = []
        for i in range(len(out)):
            if not (len(targets[i]["boxes"]) == len(out[i]["boxes"]) == 0):
                # Apply NMS
                keep_idxs = nms(out[i]["boxes"], out[i]["scores"], self.nms_threshold)
                out[i]["boxes"] = out[i]["boxes"][keep_idxs]
                out[i]["scores"] = out[i]["scores"][keep_idxs]
                out[i]["labels"] = out[i]["labels"][keep_idxs]

                # Filter out low scoring boxes
                keep_idxs = (
                    torch.where(out[i]["scores"] > self.score_threshold, 1, 0)
                    .nonzero()
                    .flatten()
                )
                out[i]["boxes"] = out[i]["boxes"][keep_idxs]
                out[i]["scores"] = out[i]["scores"][keep_idxs]
                out[i]["labels"] = out[i]["labels"][keep_idxs]

                # Filter out hard negatives
                if self.eval_only_positives:
                    keep_idxs = (
                        torch.where(out[i]["labels"] == 1, 1, 0).nonzero().flatten()
                    )
                    out[i]["boxes"] = out[i]["boxes"][keep_idxs]
                    out[i]["scores"] = out[i]["scores"][keep_idxs]
                    out[i]["labels"] = out[i]["labels"][keep_idxs]

                # Prepare outputs and targets for MAP function
                pred = []
                for j, (box, c, score) in enumerate(
                    zip(out[i]["boxes"], out[i]["labels"], out[i]["scores"])
                ):
                    pred.append(
                        [
                            0,
                            c.item(),
                            score.item(),
                            box[0].item(),
                            box[1].item(),
                            box[2].item(),
                            box[3].item(),
                        ]
                    )

                gt = []
                for j, (box, c) in enumerate(
                    zip(targets[i]["boxes"], targets[i]["labels"])
                ):
                    if self.eval_only_positives and c == 2:
                        continue

                    gt.append(
                        [
                            0,
                            c.item(),
                            1.0,
                            box[0].item(),
                            box[1].item(),
                            box[2].item(),
                            box[3].item(),
                        ]
                    )

                # Calculate metrics
                map.append(mean_average_precision(pred, gt))
                iou, acc = iou_and_acc(pred, gt)
                ious.append(iou)
                accs.append(acc)

        # Log
        results = dict()
        if len(map) > 0:
            map = sum(map) / len(map)
            self.log("val_map", map)
            results["val_map"] = map

            iou = sum(ious) / len(ious)
            self.log("val_iou", iou)
            results["val_iou"] = iou

            acc = sum(accs) / len(accs)
            self.log("val_acc", acc)
            results["val_acc"] = acc
        else:
            results["val_map"] = None
            results["val_iou"] = None
            results["val_acc"] = None

        # Save some samples detection results
        results["sample_img"] = imgs[0] if batch_idx < self.n_samples else None
        results["sample_pred_boxes"] = (
            out[0]["boxes"] if batch_idx < self.n_samples else None
        )
        results["sample_pred_labels"] = (
            out[0]["labels"] if batch_idx < self.n_samples else None
        )
        results["sample_target_boxes"] = (
            targets[0]["boxes"] if batch_idx < self.n_samples else None
        )
        results["sample_target_labels"] = (
            targets[0]["labels"] if batch_idx < self.n_samples else None
        )

        return results

    def validation_epoch_end(self, outputs):
        # Print validation metrics
        avg_map = torch.stack(
            [x["val_map"] for x in outputs if x["val_map"] is not None]
        ).mean()
        avg_iou = torch.stack(
            [x["val_iou"] for x in outputs if x["val_iou"] is not None]
        ).mean()
        avg_acc = torch.stack(
            [x["val_acc"] for x in outputs if x["val_acc"] is not None]
        ).mean()
        print(f"Validation MAP: {avg_map} IOU: {avg_iou} Acc: {avg_acc}")

        # Save sample outputs
        imgs = torch.stack(
            [
                visualize_detections(
                    x["sample_img"],
                    x["sample_pred_boxes"],
                    x["sample_pred_labels"],
                    x["sample_target_boxes"],
                    x["sample_target_labels"],
                    only_pos=self.eval_only_positives,
                )
                for x in outputs
                if x["sample_img"] is not None
            ],
        )
        grid = make_grid(imgs, nrow=1)
        tensorboard = self.logger.experiment
        tensorboard.add_image("val_samples", grid, self.current_epoch + 1)

    def test_step(self, batch, batch_idx):
        if hasattr(self, "style_net"):
            imgs, targets, imgs_s = batch

            # Split image into patches
            patches = split_tiles(
                imgs,
                tile_size=list(imgs.shape[-2:]),
                patch_size=self.crop_size,
                output_size=self.crop_size,
                overlap=self.overlap,
            )
            patches_s = split_tiles(
                imgs_s,
                tile_size=list(imgs_s.shape[-2:]),
                patch_size=self.crop_size,
                output_size=self.crop_size,
                overlap=self.overlap,
            )

            # Batch the patches
            split_patches = torch.split(patches.squeeze(0), 8, dim=0)
            split_patches_s = torch.split(patches_s.squeeze(0), 8, dim=0)[:-1]

            # Pass patchs through model
            out = []
            for patch_batch, patch_batch_s in zip(
                split_patches, cycle(split_patches_s)
            ):
                # Apply style transfer
                if patch_batch.shape[0] != patch_batch_s.shape[0]:
                    patch_batch_s = patch_batch_s[: patch_batch.shape[0]]
                patch_batch_st, _, _ = self.style_net(patch_batch, patch_batch_s)

                out.extend(self(patch_batch_st))

        else:
            imgs, targets = batch

            # Split image into patches
            patches = split_tiles(
                imgs,
                tile_size=list(imgs.shape[-2:]),
                patch_size=self.crop_size,
                output_size=self.crop_size,
                overlap=self.overlap,
            )

            # Batch the patches
            split_patches = torch.split(patches.squeeze(0), 8, dim=0)

            # Pass patch batches through model
            out = []
            for patch_batch in split_patches:
                out.extend(self(patch_batch))

        # Convert box predictions into correct format for stitching
        new_out = []
        for patch_idx, patch_out in enumerate(out):
            # If patch has box predictions
            if not patch_out["boxes"].size(0) == 0:
                # Apply NMS
                keep_idxs = nms(
                    patch_out["boxes"], patch_out["scores"], self.nms_threshold
                )
                patch_out["boxes"] = patch_out["boxes"][keep_idxs]
                patch_out["scores"] = patch_out["scores"][keep_idxs]
                patch_out["labels"] = patch_out["labels"][keep_idxs]

                # Filter out low scoring boxes
                keep_idxs = (
                    torch.where(patch_out["scores"] > self.score_threshold, 1, 0)
                    .nonzero()
                    .flatten()
                )
                patch_out["boxes"] = patch_out["boxes"][keep_idxs]
                patch_out["scores"] = patch_out["scores"][keep_idxs]
                patch_out["labels"] = patch_out["labels"][keep_idxs]

                # Filter out hard negatives
                if self.eval_only_positives:
                    keep_idxs = (
                        torch.where(patch_out["labels"] == 1, 1, 0).nonzero().flatten()
                    )
                    patch_out["boxes"] = patch_out["boxes"][keep_idxs]
                    patch_out["scores"] = patch_out["scores"][keep_idxs]
                    patch_out["labels"] = patch_out["labels"][keep_idxs]

                for j, (box, label, score) in enumerate(
                    zip(patch_out["boxes"], patch_out["labels"], patch_out["scores"])
                ):
                    new_out.append(
                        [
                            box[0].item(),
                            box[1].item(),
                            box[2].item(),
                            box[3].item(),
                            score.item(),
                            label.item(),
                            patch_idx,
                        ]
                    )
        new_out = [torch.tensor(new_out)]

        if len(new_out[0] != 0):
            stitched_boxes = stitch_boxes(
                new_out,
                tile_size=list(imgs.shape[-2:]),
                patch_size=self.crop_size,
                output_size=self.crop_size,
                overlap=self.overlap,
                iou_threshold=self.nms_threshold,
            )
        else:
            stitched_boxes = new_out

        # Calculate metrics
        stitched_boxes = stitched_boxes[0].numpy()
        if not len(stitched_boxes) == 0:
            # Prepare outputs and targets for MAP function
            pred = []
            for i, (x1, y1, x2, y2, score, label) in enumerate(stitched_boxes):
                pred.append(
                    [
                        0,
                        int(label),
                        score,
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                )

            gt = []
            for j, (box, label) in enumerate(
                zip(targets[0]["boxes"], targets[0]["labels"])
            ):
                if self.eval_only_positives and label == 2:
                    continue

                gt.append(
                    [
                        0,
                        label.item(),
                        1.0,
                        box[0].item(),
                        box[1].item(),
                        box[2].item(),
                        box[3].item(),
                    ]
                )

            # Calculate metrics
            map = mean_average_precision(pred, gt)
            iou, acc = iou_and_acc(pred, gt)

            # Log
            results = {}
            results["test_map"] = map
            results["test_iou"] = iou
            results["test_acc"] = acc

        else:
            results = {}
            results["test_map"] = None
            results["test_iou"] = None
            results["test_acc"] = None

        # Save some samples detection results
        sample_out = {}
        boxes = []
        labels = []
        for sb in stitched_boxes:
            boxes.append(sb[:4])
            labels.append(sb[-1])
        sample_out["boxes"] = torch.tensor(boxes)
        sample_out["labels"] = torch.tensor(labels)

        # Save sample outputs
        out_img = visualize_detections(
            imgs[0],
            sample_out["boxes"],
            sample_out["labels"],
            targets[0]["boxes"],
            targets[0]["labels"],
            only_pos=self.eval_only_positives,
        )

        if not os.path.exists("./test_out/"):
            os.makedirs("./test_out/")
        save_image(out_img, os.path.join("./test_out/", f"{batch_idx}.jpg"))

        return results

    def test_epoch_end(self, outputs):
        # Print metrics
        avg_map = torch.stack(
            [x["test_map"] for x in outputs if x["test_map"] is not None]
        ).mean()
        avg_iou = torch.stack(
            [x["test_iou"] for x in outputs if x["test_iou"] is not None]
        ).mean()
        avg_acc = torch.stack(
            [x["test_acc"] for x in outputs if x["test_acc"] is not None]
        ).mean()
        print(f"\nTest MAP: {avg_map} IOU: {avg_iou} Acc: {avg_acc}")

    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = SGD(
                self.net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "adam":
            optimizer = Adam(
                self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise NotImplementedError(f"{self.optimizer} is not an available optimizer")

        if self.schedule == 'step':
            scheduler = MultiStepLR(
                    optimizer,
                    milestones=self.steps,
                    gamma=self.gamma,
            ) 

            return [optimizer], [scheduler]
        else:
            return [optimizer]
