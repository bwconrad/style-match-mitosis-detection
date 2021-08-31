from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningArgumentParser

from src.data import ContentStyleDataModule
from src.model import AdaInModel

# Parse arguments
parser = LightningArgumentParser()
parser.add_lightning_class_args(pl.Trainer, None)
parser.add_lightning_class_args(ContentStyleDataModule, "data")
parser.add_argument(
    "--checkpoint", type=str, help="Checkpoint to test on", required=True
)
args = parser.parse_args()
args["logger"] = False

# Setup model
dm = ContentStyleDataModule(**args["data"])
model = AdaInModel().load_from_checkpoint(args["checkpoint"])
trainer = pl.Trainer.from_argparse_args(
    Namespace(**args),
)

# Evaluate
trainer.test(model, datamodule=dm)
