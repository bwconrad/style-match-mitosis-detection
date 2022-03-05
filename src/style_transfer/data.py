import os
from typing import Callable, List, Union

import pytorch_lightning as pl
import torch.utils.data as data
from PIL import Image
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from torchvision import transforms


class BasicContentStyleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        content_path: str = "data/coco/",
        style_path: str = "data/wikiart/",
        resize_size: int = 0,
        crop_size: int = 256,
        n_val: int = 5,
        batch_size: int = 8,
        workers: int = 4,
    ):
        """Content and style image data module

        Args:
            content_path: Path to content image directory
            style_path: Path to style image directory
            resize_size: Size of resize transformation (0 = no resizing)
            crop_size: Size of random crop transformation
            n_val: Number of validation samples per scanner
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
            # Load content and style images
            content_dataset = SimpleDataset(self.content_path, self.train_transforms)
            style_dataset = SimpleDataset(self.style_path, self.train_transforms)

            self.content_train, self.content_val = data.random_split(
                content_dataset, [len(content_dataset) - self.n_val, self.n_val]
            )
            self.style_train, self.style_val = data.random_split(
                style_dataset, [len(style_dataset) - self.n_val, self.n_val]
            )

        elif stage == "test":
            raise NotImplementedError("")

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
        raise NotImplementedError("")


class ScannerContentStyleDataModule(pl.LightningDataModule):
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
            self.content_train = SimpleScannerDataset(
                self.data_path, train_content_ids, self.train_transforms
            )
            self.content_val = SimpleScannerDataset(
                self.data_path, val_content_ids, self.val_transforms
            )
            self.style_train = SimpleScannerDataset(
                self.data_path, train_style_ids, self.train_transforms
            )
            self.style_val = SimpleScannerDataset(
                self.data_path, val_style_ids, self.val_transforms
            )
        elif stage == "test":
            # Calculate number of images
            content_ids = []
            style_ids = []

            for s in self.content_scanners:
                content_ids.extend(all_ids[s][-self.n_val :])
            for s in self.style_scanners:
                style_ids.extend(all_ids[s][-self.n_val :])

            # Load content and style images
            self.content_test = SimpleScannerDataset(
                self.data_path, content_ids, self.val_transforms
            )
            self.style_test = SimpleScannerDataset(
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
                persistent_workers=True,
            ),
            "style": DataLoader(
                self.style_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
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
                persistent_workers=True,
            ),
            "style": DataLoader(
                self.style_val,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
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
                persistent_workers=True,
            ),
            "style": DataLoader(
                self.style_test,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
            ),
        }

        return CombinedLoader(loaders, "max_size_cycle")


class SimpleScannerDataset(data.Dataset):
    def __init__(
        self, root: str, ids: List[int], transforms: Callable, preload: bool = False
    ):
        """Image dataset from directory

        Args:
            root: Path to directory
            ids: Image ids
            transforms: Image augmentations
            preload: Preload images to memory
        """
        super().__init__()
        self.root = root
        self.paths = [sorted(os.listdir(root))[i - 1] for i in ids]
        self.transforms = transforms
        self.preload = preload

        # Preload images
        if preload:
            print("Loading images to memory...")
            self.imgs = []
            for i in range(len(ids)):
                file_name = os.path.join(self.root, self.paths[i])
                self.imgs.append(Image.open(file_name).convert("RGB"))
            print(f"Loaded {len(self.imgs)} images")

    def __getitem__(self, index):
        if self.preload:
            img = self.imgs[index]
        else:
            img = Image.open(os.path.join(self.root, self.paths[index])).convert("RGB")
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.paths)


class SimpleDataset(data.Dataset):
    def __init__(self, root: str, transforms: Callable):
        """Image dataset from directory
        Args:
            root: Path to directory
            transforms: Image augmentations
        """
        super().__init__()
        self.root = root
        self.paths = os.listdir(root)
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.paths[index])).convert("RGB")
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.paths)
