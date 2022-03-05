import glob
import os
from typing import Callable, List

import pytorch_lightning as pl
import torch
import torch.utils.data as data
from PIL import Image
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from torchvision import transforms


class MidogScannerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "data/midog/",
        style_scanner: int = None,
        crop_size: int = 256,
        n_val: int = 25,
        batch_size: int = 8,
        workers: int = 4,
    ):
        """Midog classification image data module

        Args:
            data_path: Path to image directory
            style_scanner: Style images scanner (for test time)
            resize_size: Size of resize transformation (0 = no resizing)
            crop_size: Size of random crop transformation
            n_val: Number of validation samples per class
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.path = data_path
        self.style_scanner = style_scanner
        self.n_val = n_val
        self.batch_size = batch_size
        self.workers = workers

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage="fit"):
        # Split train and val images
        train_files = []
        val_files = []
        train_labels = []
        val_labels = []
        paths = sorted([f for f in glob.glob(self.path + "/*", recursive=True)])

        for i in range(4):
            for j in range(50):
                if j < 50 - self.n_val:
                    train_files.append(paths[(i * 50) + j])
                    train_labels.append(i)
                else:
                    val_files.append(paths[(i * 50) + j])
                    val_labels.append(i)

        if stage == "fit":
            # Load data
            self.train_dataset = ScannerClassificationDataset(
                train_files, train_labels, self.train_transforms
            )
            self.val_dataset = ScannerClassificationDataset(
                val_files, val_labels, self.val_transforms
            )

        elif stage == "test":
            if self.style_scanner == 1:
                style_files = val_files[: self.n_val]
            elif self.style_scanner == 2:
                style_files = val_files[self.n_val : self.n_val * 2]
            elif self.style_scanner == 3:
                style_files = val_files[self.n_val * 2 : self.n_val * 3]
            else:
                style_files = val_files[self.n_val * 3 : self.n_val * 4]

            self.test_dataset = ScannerClassificationDataset(
                val_files, val_labels, self.val_transforms
            )
            self.style_dataset = SimpleDataset(
                style_files,
                self.val_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.style_scanner:
            loaders = {
                "content": DataLoader(
                    self.test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=self.workers,
                    pin_memory=True,
                ),
                "style": DataLoader(
                    self.style_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=self.workers,
                    pin_memory=True,
                ),
            }
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
            )

        return CombinedLoader(loaders, "max_size_cycle")


class ScannerClassificationDataset(data.Dataset):
    def __init__(self, paths: List[str], labels: List[int], transforms: Callable):
        """Image dataset from directory

        Args:
            paths: List of file paths
            transforms: Image augmentations
        """
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        img = Image.open(path).convert("RGB")
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.paths)


class SimpleDataset(data.Dataset):
    def __init__(self, paths: List[str], transforms: Callable):
        """Image dataset from directory

        Args:
            paths: Paths to images
            transforms: Image augmentations
        """
        super().__init__()
        self.paths = paths
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.paths)
