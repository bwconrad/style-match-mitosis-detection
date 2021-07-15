from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningArgumentParser
from src.data import MidogDataModule
from src.model import DetectionModel

# Parse arguments
parser = LightningArgumentParser()
parser.add_lightning_class_args(pl.Trainer, None)
parser.add_lightning_class_args(MidogDataModule, "data")
parser.add_lightning_class_args(DetectionModel, "model")
parser.add_argument("--test", action="store_true", help="Perform model evaluation.")
parser.add_argument(
    "--checkpoint", type=str, help="Checkpoint to test on", required=True
)
parser.link_arguments("data.crop_size", "model.crop_size")
args = parser.parse_args()
args["logger"] = False

# Setup model
dm = MidogDataModule(**args["data"])
model = DetectionModel().load_from_checkpoint(
    args["checkpoint"], style_checkpoint=args["model"]["style_checkpoint"], strict=False
)
trainer = pl.Trainer.from_argparse_args(
    Namespace(**args),
)
# Evaluate
trainer.test(model, datamodule=dm)
