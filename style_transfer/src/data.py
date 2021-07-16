import os
from typing import Callable, List, Union

import pytorch_lightning as pl
import torch.utils.data as data
from PIL import Image
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from torchvision import transforms


class ContentStyleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "data/midog/",
        content_scanners: Union[List[int], int] = 1,
        style_scanners: Union[List[int], int] = 4,
        resize_size: int = 0,
        crop_size: int = 256,
        n_val: int = 5,
        batch_size: int = 8,
        workers: int = 4,
    ):
        """Content and style image data module

        Args:
            data_path: Path to image directory
            content_scanners: Content image scanners
            style_scanners: style image scanners
            resize_size: Size of resize transformation (0 = no resizing)
            crop_size: Size of random crop transformation
            n_val: Number of validation samples per scanner
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.data_path = data_path
        self.n_val = n_val
        self.batch_size = batch_size
        self.workers = workers

        if isinstance(content_scanners, int):
            self.content_scanners = [content_scanners]
        else:
            self.content_scanners = content_scanners
        if isinstance(style_scanners, int):
            self.style_scanners = [style_scanners]
        else:
            self.style_scanners = style_scanners

        # No resize before crop
        if resize_size == 0:
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
        # Resize before crop
        else:
            self.train_transforms = transforms.Compose(
                [
                    transforms.Resize(resize_size),
                    transforms.RandomCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            self.val_transforms = transforms.Compose(
                [
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                ]
            )

    def setup(self, stage="fit"):
        all_ids = {
            1: list(range(1, 51)),
            2: list(range(51, 101)),
            3: list(range(101, 151)),
            4: list(range(151, 201)),
        }

        if stage == "fit":
            # Split content and style images into train/val splits
            train_content_ids = []
            val_content_ids = []
            train_style_ids = []
            val_style_ids = []

            for s in self.content_scanners:
                train_content_ids.extend(all_ids[s][: -self.n_val])
                val_content_ids.extend(all_ids[s][-self.n_val :])
            for s in self.style_scanners:
                train_style_ids.extend(all_ids[s][: -self.n_val])
                val_style_ids.extend(all_ids[s][-self.n_val :])

            # Load content and style images
            self.content_train = SimpleDataset(
                self.data_path, train_content_ids, self.train_transforms
            )
            self.content_val = SimpleDataset(
                self.data_path, val_content_ids, self.val_transforms
            )
            self.style_train = SimpleDataset(
                self.data_path, train_style_ids, self.train_transforms
            )
            self.style_val = SimpleDataset(
                self.data_path, val_style_ids, self.val_transforms
            )
        elif stage == "test":
            # Calculate number of images
            content_ids = all_ids[self.content_scanner][-self.n_val :]
            style_ids = all_ids[self.style_scanner][-self.n_val :]

            # Load content and style images
            self.content_test = SimpleDataset(
                self.data_path, content_ids, self.val_transforms
            )
            self.style_test = SimpleDataset(
                self.data_path, style_ids, self.val_transforms
            )

    def train_dataloader(self):
        loaders = {
            "content": DataLoader(
                self.content_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=True,
            ),
            "style": DataLoader(
                self.style_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=True,
            ),
        }

        return CombinedLoader(loaders, "max_size_cycle")

    def val_dataloader(self):
        loaders = {
            "content": DataLoader(
                self.content_val,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=True,
            ),
            "style": DataLoader(
                self.style_val,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=True,
            ),
        }

        return CombinedLoader(loaders, "max_size_cycle")

    def test_dataloader(self):
        loaders = {
            "content": DataLoader(
                self.content_test,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=True,
            ),
            "style": DataLoader(
                self.style_test,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=True,
            ),
        }

        return CombinedLoader(loaders, "max_size_cycle")


class SimpleDataset(data.Dataset):
    def __init__(self, root: str, indices: List[int], transforms: Callable):
        """Image dataset from directory

        Args:
            root: Path to directory
            indices: Start and end files indices to include
            transforms: Image augmentations
        """
        super().__init__()
        self.root = root
        self.paths = [sorted(os.listdir(root))[i - 1] for i in indices]
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.paths[index])).convert("RGB")
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.paths)
