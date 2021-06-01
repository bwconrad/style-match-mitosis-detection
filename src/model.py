import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid, save_image
from torch.optim import lr_scheduler, Adam

# from torchmetrics import SSIM
from collections import OrderedDict

from .network import AdaInNetwork
from .loss import ContentLoss, StyleLoss


class AdaInModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_content: float = 1.0,
        weight_style: float = 1.0,
        use_bfg: bool = False,
        use_skip: bool = False,
    ):
        """Arbitrary style transfer model

        Args:
            lr: Learning rate
            weight_content: Weight for content loss
            weight_style: Weight for style loss
            use_bfg: Use bottleneck feature aggregation
            use_skip: Use unet skip connections
        """
        super().__init__()
        self.save_hyperparameters()

        # Hyperparameters
        self.lr = lr
        self.weight_content = weight_content
        self.weight_style = weight_style
        self.use_bfg = use_bfg
        self.use_skip = use_skip

        # Network
        self.net = AdaInNetwork(use_bfg=use_bfg, use_skip=use_skip)

        # Losses
        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss(use_statistics=True)

        # Metrics
        # self.val_ssim = SSIM()

    def forward(self, c, s, alpha=1.0):
        return self.net(c, s, alpha)

    def training_step(self, batch, _):
        img_c = batch["content"]
        img_s = batch["style"]

        # Pass through model
        g_t, t, f_s = self(img_c, img_s)

        # Calculate loss
        f_g_t = self.net.encoder(g_t, return_all=True)
        if self.use_bfg:
            f_g_t_cat = self.net.bfg(f_g_t)
            loss_c = self.content_loss(f_g_t_cat, t)
        else:
            loss_c = self.content_loss(f_g_t[-1], t)
        loss_s = self.style_loss(f_g_t, f_s)
        loss = self.weight_content * loss_c + self.weight_style * loss_s

        self.log("train_loss", loss)
        self.log("train_content_loss", loss_c, prog_bar=True)
        self.log("train_style_loss", loss_s, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

        return loss

    def validation_step(self, batch, _):
        img_c = batch["content"]
        img_s = batch["style"]

        # Pass through model
        g_t, t, f_s = self(img_c, img_s)

        # Calculate loss
        f_g_t = self.net.encoder(g_t, return_all=True)
        if self.use_bfg:
            f_g_t_cat = self.net.bfg(f_g_t)
            loss_c = self.content_loss(f_g_t_cat, t)
        else:
            loss_c = self.content_loss(f_g_t[-1], t)
        loss_s = self.style_loss(f_g_t, f_s)
        loss = self.weight_content * loss_c + self.weight_style * loss_s

        # ssim = self.val_ssim(img_c, g_t.float())

        self.log("val_loss", loss)
        self.log("val_content_loss", loss_c)
        self.log("val_style_loss", loss_s)
        # self.log("val_ssim", ssim)

        return OrderedDict({"val_sample": torch.stack([img_c[0], img_s[0], g_t[0]])})

    def validation_epoch_end(self, outputs):
        imgs = torch.cat([x["val_sample"] for x in outputs], 0)
        grid = make_grid(imgs, nrow=3, normalize=True)
        self.logger.experiment.add_image("val_samples", grid, self.global_step)

    def configure_optimizers(self):
        optimizer = Adam(
            self.net.parameters(),
            lr=self.lr,
        )
        scheduler = {
            "scheduler": lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_steps,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
