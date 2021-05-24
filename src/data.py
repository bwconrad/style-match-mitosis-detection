import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
import torch
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from typing import Callable, List


class ContentStyleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        content_path: str = "data/midog/1/",
        style_path: str = "data/midog/4/",
        resize_size: int = 0,
        crop_size: int = 256,
        n_val: int = 5,
        batch_size: int = 8,
        workers: int = 4,
    ):
        """Content and style image data module

        Args:
            content_path: Path to content images directory
            style_path: Path to style images directory
            resize_size: Size of resize transformation (0 = no resizing)
            crop_size: Size of random crop transformation
            n_val: Number of validation samples
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.content_path = content_path
        self.style_path = style_path
        self.n_val = n_val
        self.batch_size = batch_size
        self.workers = workers

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
        if stage == "fit":
            # Calculate number of training images
            n_train_content = len(os.listdir(self.content_path)) - self.n_val
            n_train_style = len(os.listdir(self.style_path)) - self.n_val

            # Load content and style images
            self.content_train = SimpleDataset(
                self.content_path, [0, n_train_content], self.train_transforms
            )
            self.content_val = SimpleDataset(
                self.content_path,
                [n_train_content, n_train_content + self.n_val],
                self.val_transforms,
            )
            self.style_train = SimpleDataset(
                self.style_path, [0, n_train_style], self.train_transforms
            )
            self.style_val = SimpleDataset(
                self.style_path,
                [n_train_style, n_train_style + self.n_val],
                self.val_transforms,
            )

    def train_dataloader(self):
        return {
            "content": DataLoader(
                self.content_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers,
                pin_memory=True,
            ),
            "style": DataLoader(
                self.style_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers,
                pin_memory=True,
            ),
        }

    def val_dataloader(self):
        loaders = {
            "content": DataLoader(
                self.content_val,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
            ),
            "style": DataLoader(
                self.style_val,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
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
        self.paths = os.listdir(root)[indices[0] : indices[1]]
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.paths[index])).convert("RGB")
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.paths)
