from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningArgumentParser
from src.data import MidogCellDataModule
from src.model import Model

# Parse arguments
parser = LightningArgumentParser()
parser.add_lightning_class_args(pl.Trainer, None)
parser.add_lightning_class_args(MidogCellDataModule, "data")
parser.add_lightning_class_args(Model, "model")
parser.add_argument("--test", action="store_true", help="Perform model evaluation.")
parser.add_argument(
    "--checkpoint", type=str, help="Checkpoint to test on", required=True
)
args = parser.parse_args()
args["logger"] = False

# Setup model
dm = MidogCellDataModule(**args["data"])
model = Model().load_from_checkpoint(
    args["checkpoint"], style_checkpoint=args["model"]["style_checkpoint"], strict=False
)
trainer = pl.Trainer.from_argparse_args(
    Namespace(**args),
)

# Evaluate
trainer.test(model, datamodule=dm)
