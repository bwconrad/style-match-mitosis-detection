import pytorch_lightning as pl
import torch
from torch.optim import SGD, Adam
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
from torchvision.ops import nms
from torchvision.utils import make_grid, save_image

from .metrics import mean_average_precision
from .visualize import visualize_detections


class DetectionModel(pl.LightningModule):
    def __init__(
        self,
        arch: str = "faster_rcnn",
        n_classes: int = 2,
        optimizer: str = "sgd",
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        """Midog detection model

        Args:
            arch: Detection model architecture (faster_rcnn | retinanet)
            n_classes: Number of classes
            optimizer: Name of optimizer (sgd | adam)
            lr: Learning rate
            momentum: Momentum for SGD
            weight_decay: Weight decay
        """
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.momentum = momentum
        self.optimizer = optimizer
        self.weight_decay = weight_decay

        if arch == "faster_rcnn":
            self.net = fasterrcnn_resnet50_fpn(
                pretrained=True, pretrained_backbone=True
            )
            in_features = self.net.roi_heads.box_predictor.cls_score.in_features
            head = FastRCNNPredictor(in_features, n_classes + 1)
            self.net.roi_heads.box_predictor = head
        else:
            self.net = retinanet_resnet50_fpn(pretrained=True, pretrained_backbone=True)
            self.net.head = RetinaNetHead(
                in_channels=self.net.backbone.out_channels,
                num_anchors=self.net.head.classification_head.num_anchors,
                num_classes=n_classes + 1,
            )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
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
        imgs, targets = batch

        # Pass through model
        out = self(imgs)

        map = 0
        n_counted = 0
        for i in range(len(out)):
            if not (len(targets[i]["boxes"]) == len(out[i]["boxes"]) == 0):
                # Perform NMS
                keep_idxs = nms(out[i]["boxes"], out[i]["scores"], 0.2)
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
                            j,
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
                    gt.append(
                        [
                            j,
                            c.item(),
                            1.0,
                            box[0].item(),
                            box[1].item(),
                            box[2].item(),
                            box[3].item(),
                        ]
                    )

                map += mean_average_precision(pred, gt)
                n_counted += 1

        # Log
        results = dict()
        if n_counted > 0:
            map /= n_counted
            self.log("val_map", map)
            results["val_map"] = map
        else:
            results["val_map"] = None

        results["sample_img"] = imgs[0] if batch_idx < 10 else None
        results["sample_pred_boxes"] = out[0]["boxes"] if batch_idx < 10 else None
        results["sample_pred_labels"] = out[0]["labels"] if batch_idx < 10 else None
        results["sample_target_boxes"] = targets[0]["boxes"] if batch_idx < 10 else None
        results["sample_target_labels"] = (
            targets[0]["labels"] if batch_idx < 10 else None
        )

        return results

    def validation_epoch_end(self, outputs):
        # Print validation MAP
        avg_map = torch.stack(
            [x["val_map"] for x in outputs if x["val_map"] is not None]
        ).mean()
        print(f"Validation MAP: {avg_map}")

        # Save sample outputs
        imgs = torch.stack(
            [
                visualize_detections(
                    x["sample_img"],
                    x["sample_pred_boxes"],
                    x["sample_pred_labels"],
                    x["sample_target_boxes"],
                    x["sample_target_labels"],
                )
                for x in outputs
                if x["sample_img"] is not None
            ],
        )
        grid = make_grid(imgs, nrow=1)
        # save_image(grid, "a.png")
        tensorboard = self.logger.experiment
        tensorboard.add_image("val_samples", grid, self.current_epoch + 1)

    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = SGD(
                self.net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = Adam(
                self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        return [optimizer]
