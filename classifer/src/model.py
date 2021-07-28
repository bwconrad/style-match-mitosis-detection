import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, ConfusionMatrix
from torchvision.models import resnet50
from torchvision.utils import save_image

from .style_transfer.model import AdaInModel


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        optimizer: str = "adam",
        style_checkpoint: str = None,
        n_classes: int = 4,
        smoothing: float = 0,
        mix_alpha: float = 0,
        cutmix_alpha: float = 0,
        arch: str = "resnet50",
    ):
        """Scanner Classifer

        Args:
            lr: learning rate
            optimizer: name of optimizer (adam | sgd)
            style_checkpoint: checkpoint of style transfer model
            n_classes: number of classes
            smoothing: label smoothing factor
            mix_alpha: alpha value for mixup
            cutmix_alpha: alpha value for cutmix
            arch: name of architecture
        """
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.optimizer = optimizer
        self.smoothing = smoothing
        self.mix_alpha = mix_alpha
        self.cutmix_alpha = cutmix_alpha
        self.n_classes = n_classes

        # Initalize network
        if arch == "resnet50":
            self.net = resnet50(pretrained=True)
            self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)
        elif arch == "vit":
            self.net = timm.models.vision_transformer.VisionTransformer(
                img_size=64, patch_size=8, num_classes=3
            )
        else:
            raise NotImplementedError(
                f"{arch} is not an available network architecture"
            )
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_confusion = ConfusionMatrix(num_classes=n_classes)

        if style_checkpoint:
            self.style_net = AdaInModel().load_from_checkpoint(style_checkpoint)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        y = F.one_hot(y, num_classes=self.n_classes).float()

        # Apply label smoothing
        if self.smoothing > 0:
            y = self._smooth(y)

        # Apply mixup
        if self.mix_alpha > 0:
            x, y = self._mixup(x, y)

        # Apply cutmix
        if self.cutmix_alpha > 0:
            x, y = self._cutmix(x, y)

        # Pass through model
        pred = self(x)

        # Calculate loss and accuracy
        loss = F.binary_cross_entropy_with_logits(pred, y)
        acc = self.train_acc(pred.max(1).indices, y.max(1).indices)

        # Log
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y = F.one_hot(y, num_classes=self.n_classes).float()

        # Pass through model
        pred = self(x)

        # Calculate loss and accuracy
        loss = F.binary_cross_entropy_with_logits(pred, y)
        acc = self.val_acc(pred.max(1).indices, y.max(1).indices)

        # Log
        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return {
            "val_acc": acc,
            "val_loss": loss,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        print(f"Validation Loss: {avg_loss} Validation Accuracy: {avg_acc}")

    def test_step(self, batch, _):
        x, y = batch["content"]
        img_s = batch["style"]

        # Apply style transfer
        x_s, _, _ = self.style_net(x, img_s)
        # x_s = x

        # Pass through model
        pred = self(x_s)

        max_prob = F.softmax(pred, dim=1).max(1).values

        # Calculate loss and accuracy
        loss = F.cross_entropy(pred, y)
        acc = self.val_acc(pred.max(1).indices, y)
        conf = self.test_confusion(pred.max(1).indices, y)

        # Log
        self.log("test_loss", loss)
        self.log("test_acc", acc)

        return {
            "confs": conf,
            "max_probs": max_prob,
        }

    def test_epoch_end(self, outputs):
        confs = torch.stack([x["confs"] for x in outputs], axis=0).sum(dim=0)
        max_probs = torch.stack([x["max_probs"] for x in outputs], axis=0).mean(dim=0)

        print("\nTest Confusion Matrix:")
        print(confs)
        print(f"Average Confidence: {max_probs.item():.3}")

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = Adam(self.net.parameters(), lr=self.lr)
        else:
            optimizer = SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]

    def _smooth(self, y):
        smoothing = self.smoothing
        confidence = 1.0 - smoothing  # Confidence for target class
        label_shape = y.size()
        other = smoothing / (label_shape[1] - 1)  # Confidence for non-target classes

        # Create new smoothed target vector
        smooth_y = torch.empty(size=label_shape, device=self.device)
        smooth_y.fill_(other)
        smooth_y.add_(y * confidence).sub_(y * other)

        return smooth_y

    def _mixup(self, x, y):
        lam = np.random.beta(self.mix_alpha, self.mix_alpha)
        indices = np.random.permutation(x.size(0))
        x_mix = x * lam + x[indices] * (1 - lam)
        y_mix = y * lam + y[indices] * (1 - lam)
        return x_mix, y_mix

    def _cutmix(self, x, y):
        def rand_bbox(size, lam):
            """ From: https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py """
            W = size[2]
            H = size[3]
            cut_rat = np.sqrt(1.0 - lam)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

            cx = np.random.randint(W)
            cy = np.random.randint(H)

            x1 = np.clip(cx - cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y2 = np.clip(cy + cut_h // 2, 0, H)

            return x1, y1, x2, y2

        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        indices = np.random.permutation(x.size(0))

        # Perform cutmix
        x1, y1, x2, y2 = rand_bbox(x.size(), lam)  # Select a random rectangle
        x[:, :, x1:x2, y1:y2] = x[
            indices, :, x1:x2, y1:y2
        ]  # Replace the cutout section with the other image's pixels

        # Adjust target
        lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size()[-1] * x.size()[-2]))
        y_mix = y * lam + y[indices] * (1 - lam)
        return x, y_mix
