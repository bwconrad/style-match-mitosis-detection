import os
from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import LightningArgumentParser
from src.model_style_transfer import AdaInModel
from src.style_transfer.data import ScannerContentStyleDataModule

# Parse arguments
parser = LightningArgumentParser()
parser.add_lightning_class_args(pl.Trainer, None)
parser.add_lightning_class_args(AdaInModel, "model")
parser.add_lightning_class_args(ScannerContentStyleDataModule, "data")
parser.add_argument(
    "--output_path",
    type=str,
    help="Directory to save outputs.",
    default="output/style_transfer/",
)
parser.add_argument(
    "--experiment_name", type=str, help="Name of experiment.", default="default"
)
args = parser.parse_args()

# Define loggers
tb_logger = TensorBoardLogger(
    save_dir=args["output_path"], name=args["experiment_name"]
)

# Setup model
dm = ScannerContentStyleDataModule(**args["data"])
model = AdaInModel(**args["model"])
trainer = pl.Trainer.from_argparse_args(Namespace(**args), logger=tb_logger)

# Save config
parser.save(
    args, os.path.join(trainer.logger.experiment.log_dir, "config.yaml"), "yaml"
)

# Run trainer
trainer.tune(model, dm)
trainer.fit(model, dm)
