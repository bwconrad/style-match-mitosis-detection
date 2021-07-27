import glob
import json
import os
import random
from typing import Callable, List, Union

import albumentations as A
import cv2
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from torchvision import transforms

from .transforms import MyRandomSizedCrop


class MidogScannerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "data/midog/",
        style_path: str = "data/midog/4/",
        crop_size: int = 256,
        n_val: int = 5,
        batch_size: int = 8,
        workers: int = 4,
    ):
        """Midog classification image data module

        Args:
            data_path: Path to image directory
            style_path: Path to style image directory (for test time)
            resize_size: Size of resize transformation (0 = no resizing)
            crop_size: Size of random crop transformation
            n_val: Number of validation samples per class
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.path = data_path
        self.style_path = style_path
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
        for i in range(4):
            paths = [f for f in glob.glob(self.path + f"/{i+1}/*", recursive=True)]
            train_files += paths[: -self.n_val]
            val_files += paths[-self.n_val :]

        if stage == "fit":
            # Load data
            self.train_dataset = ScannerClassificationDataset(
                train_files, self.train_transforms
            )
            self.val_dataset = ScannerClassificationDataset(
                val_files, self.val_transforms
            )

        elif stage == "test":
            n_style = len(os.listdir(self.style_path)) - self.n_val
            self.test_dataset = ScannerClassificationDataset(
                val_files, self.val_transforms
            )
            self.style_dataset = SimpleDataset(
                self.style_path,
                [n_style, n_style + self.n_val],
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

        return CombinedLoader(loaders, "max_size_cycle")


class MidogCellDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "data/midog/",
        ann_path: str = "data/MIDOG.json",
        train_scanner: Union[List[int], int] = 1,
        val_scanner: Union[List[int], int] = 1,
        test_scanner: Union[List[int], int] = 1,
        style_scanner: Union[List[int], int] = None,
        size: int = 128,
        n_val: int = 10,
        batch_size: int = 16,
        workers: int = 4,
    ):
        """Midog classification image data module

        Args:
            data_path: Path to image directory
            ann_path: Path to annotation file
            train_scanner: Training set scanner id
            val_scanner: Validation set scanner id
            test_scanner: Test set scanner id
            style_scanner: Style set scanner id
            size: Resize size
            n_val: Number of validation samples per scanner
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.data_path = data_path
        self.ann_path = ann_path
        self.train_scanner = train_scanner
        self.val_scanner = val_scanner
        self.test_scanner = test_scanner
        self.style_scanner = style_scanner
        self.n_val = n_val
        self.batch_size = batch_size
        self.workers = workers
        self.size = size

    def setup(self, stage="fit"):
        all_ids = {
            1: list(range(1, 51)),
            2: list(range(51, 101)),
            3: list(range(101, 151)),
            4: list(range(151, 201)),
        }

        # Split train and val images
        train_ids = []
        if isinstance(self.train_scanner, list):
            for i in self.train_scanner:
                train_ids += all_ids[i][: -self.n_val]
        else:
            train_ids += all_ids[self.train_scanner][: -self.n_val]

        val_ids = []
        if isinstance(self.val_scanner, list):
            for i in self.val_scanner:
                val_ids += all_ids[i][-self.n_val :]
        else:
            val_ids += all_ids[self.val_scanner][-self.n_val :]

        if stage == "fit":
            # Load data
            self.train_dataset = CellClassificationDataset(
                train_ids, self.ann_path, self.data_path, size=self.size, train=True
            )
            self.val_dataset = CellClassificationDataset(
                val_ids, self.ann_path, self.data_path, size=self.size, train=False
            )

        elif stage == "test":
            raise NotImplementedError("")
            n_style = len(os.listdir(self.style_path)) - self.n_val
            self.test_dataset = ScannerClassificationDataset(
                val_files, self.val_transforms
            )
            self.style_dataset = SimpleDataset(
                self.style_path,
                [n_style, n_style + self.n_val],
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
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        loaders = {
            "content": DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
            ),
            "style": DataLoader(
                self.style_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
            ),
        }

        return CombinedLoader(loaders, "max_size_cycle")


class ScannerClassificationDataset(data.Dataset):
    def __init__(self, paths: List[str], transforms: Callable):
        """Image dataset from directory

        Args:
            paths: List of file paths
            transforms: Image augmentations
        """
        super().__init__()
        self.paths = paths
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        label = torch.tensor(int(path[-10]) - 1)
        img = Image.open(path).convert("RGB")
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.paths)


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


class CellClassificationDataset(data.Dataset):
    def __init__(
        self,
        ids: List[int],
        ann_path: str,
        img_path: str,
        size: int,
        train: bool = True,
    ):
        super().__init__()
        self.img_path = img_path
        self.train = train

        # Load annotations
        self.annotations, file_list = self.load_ann(ann_path)
        self.annotations = self.annotations.loc[self.annotations["id"].isin(ids)]

        # Preload images
        print("Loading images to memory...")
        self.imgs = {}
        for id in ids:
            file_name = os.path.join(self.img_path, file_list[id - 1])
            self.imgs[id] = cv2.imread(file_name)
        print(f"Loaded {len(self.imgs)} images")

        self.ids = ids

        # Train augmentations
        if train:
            self.transforms_box = A.Compose(
                [
                    A.RandomCropNearBBox(),
                    A.Resize(size, size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(),
                    ToTensorV2(),
                ],
            )
            self.transforms_rand = A.Compose(
                [
                    MyRandomSizedCrop(
                        min_max_height_width=[25, 75], height=size, width=size
                    ),
                    A.Resize(size, size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(),
                    ToTensorV2(),
                ],
            )
        else:
            self.annotations = self.generate_false_samples(self.annotations)
            self.transforms = A.Compose(
                [
                    A.RandomCropNearBBox(0),
                    A.Resize(size, size),
                    ToTensorV2(),
                ],
            )

    def generate_false_samples(self, annotations):
        rows = []
        n = len(annotations)
        for i in range(n):
            ann = annotations.iloc[i]
            file_name = ann["file_name"]
            id = ann["id"]
            width = ann["width"]
            height = ann["height"]

            if id in list(range(1, 51)):
                scanner = 1
            elif id in list(range(51, 101)):
                scanner = 2
            elif id in list(range(101, 151)):
                scanner = 3
            else:
                scanner = 4

            # Generate box
            c = [
                int((width - 100) * ((i * 13) % n / n)),
                int((height - 100) * ((i * 13) % n / n)),
            ]
            box = [
                c[0] - 25,
                c[1] - 25,
                c[0] + 25,
                c[1] + 25,
            ]
            rows.append(
                [
                    file_name,
                    id,
                    box,
                    0,
                    scanner,
                    width,
                    height,
                ]
            )

        new_anns = pd.DataFrame(
            rows,
            columns=[
                "file_name",
                "id",
                "box",
                "label",
                "scanner",
                "width",
                "height",
            ],
        )

        return pd.concat([annotations, new_anns]).reset_index()

    def load_ann(self, ann_path):
        rows = []
        with open(ann_path) as f:
            data = json.load(f)

            file_list = []
            for row in data["images"]:
                file_name = row["file_name"]
                file_list.append(file_name)
                id = row["id"]
                width = row["width"]
                height = row["height"]

                if id in list(range(1, 51)):
                    scanner = 1
                elif id in list(range(51, 101)):
                    scanner = 2
                elif id in list(range(101, 151)):
                    scanner = 3
                else:
                    scanner = 4

                # Create row for each cell annotation
                for annotation in [
                    anno for anno in data["annotations"] if anno["image_id"] == id
                ]:
                    # Clip negative coordinates to 0
                    if annotation["bbox"][0] < 0:
                        annotation["bbox"][0] = 0
                    if annotation["bbox"][1] < 0:
                        annotation["bbox"][1] = 0

                    # Clip coordinates > height or width
                    if annotation["bbox"][2] > width:
                        annotation["bbox"][2] = width
                    if annotation["bbox"][3] > height:
                        annotation["bbox"][3] = height

                    rows.append(
                        [
                            file_name,
                            id,
                            annotation["bbox"],
                            annotation["category_id"],
                            scanner,
                            width,
                            height,
                        ]
                    )

        return (
            pd.DataFrame(
                rows,
                columns=[
                    "file_name",
                    "id",
                    "box",
                    "label",
                    "scanner",
                    "width",
                    "height",
                ],
            ),
            file_list,
        )

    def __getitem__(self, index):
        # Load annotations
        ann = self.annotations.iloc[index]
        box = ann["box"]
        label = ann["label"]  # 1 = positive, 2 = hard negative
        id = ann["id"]

        # Load image
        img = self.imgs[id]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.train:
            if random.random() < 0.5:
                augmented = self.transforms_box(image=img, cropping_bbox=box)
                label = torch.tensor(label)
            else:
                augmented = self.transforms_rand(image=img, cropping_bbox=box)
                label = torch.tensor(0)
        else:
            augmented = self.transforms(image=img, cropping_bbox=box)
            label = torch.tensor(label)

        img = augmented["image"] / 255

        return img, label

    def __len__(self):
        return len(self.annotations)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # dm = MidogCellDataModule(train_scanner=1, val_scanner=2)
    # dm.setup()
    # d = dm.val_dataset
    # dm = MidogDataModule()
    # dm.setup(stage="test")
    # s = dm.style_dataset
    # t = dm.test_dataset

    d = CellClassificationDataset(
        [100, 101, 102], "data/MIDOG.json", "data/midog/", size=128, train=False
    )
    print(len(d))
    x = d[-2]
    print(x[1])
    print(x[0].size())
    plt.imshow(x[0].numpy().transpose(1, 2, 0))
    plt.show()
