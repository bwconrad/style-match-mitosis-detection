import json
import os
from typing import Callable, List

import albumentations as A
import cv2
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import transforms

from .data_utils import RandomCropIncludeBBox


class MidogDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "data/midog/",
        ann_path: str = "data/MIDOG.json",
        train_scanner: int = 1,
        val_scanner: int = 1,
        n_train_samples: int = 1500,
        n_val_samples: int = 500,
        resize_size: str = 0,
        crop_size: int = 256,
        n_val: int = 10,
        batch_size: int = 4,
        workers: int = 4,
    ):
        """Midog classification image data module

        Args:
            data_path: Path to image directory
            ann_path: Path to annotation file
            train_scanner: scanner index of training set
            val_scanner: scanner index of validation set
            n_train_samples: Number of training samples
            n_val_samples: Number of validation samples
            resize_size: Size of resize transformation (0 = no resizing)
            crop_size: Size of random crop transformation
            n_val: Number of validation samples per class
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.data_path = data_path
        self.ann_path = ann_path
        self.train_scanner = train_scanner
        self.val_scanner = val_scanner
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_val = n_val
        self.batch_size = batch_size
        self.workers = workers

        self.train_transforms = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomCrop(width=crop_size, height=crop_size),
                        RandomCropIncludeBBox(width=crop_size, height=crop_size),
                    ],
                    p=1,
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3
            ),
        )

        self.val_transforms = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomCrop(width=crop_size, height=crop_size),
                        RandomCropIncludeBBox(width=crop_size, height=crop_size),
                    ],
                    p=1,
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3
            ),
        )

    def setup(self, stage="fit"):
        all_ids = {
            1: list(range(1, 51)),
            2: list(range(51, 101)),
            3: list(range(101, 151)),
            4: list(range(151, 201)),
        }

        train_ids = all_ids[self.train_scanner][: -self.n_val]
        val_ids = all_ids[self.val_scanner][-self.n_val :]

        if stage == "fit":
            # Load data
            self.train_dataset = MigdogDataset(
                train_ids,
                self.ann_path,
                self.data_path,
                self.train_transforms,
                self.n_train_samples,
            )
            self.val_dataset = MigdogDataset(
                val_ids,
                self.ann_path,
                self.data_path,
                self.val_transforms,
                self.n_val_samples,
            )

        elif stage == "test":
            raise NotImplementedError("Test dataloading not implemented")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )


def collate_fn(batch):
    images = list()
    targets = list()

    for b in batch:
        images.append(b[0])
        targets.append(b[1])

    images = torch.stack(images, dim=0)

    return (
        images,
        targets,
    )


class MigdogDataset(data.Dataset):
    def __init__(
        self,
        ids: List[int],
        ann_path: str,
        img_path: str,
        transforms: Callable,
        n_samples: int,
    ):
        super().__init__()
        self.img_path = img_path
        self.n_samples = n_samples

        # Load annotations
        self.annotations = self.load_ann(ann_path)

        # Preload images
        print("Loading images to memory...")
        self.imgs = []
        for id in ids:
            file_name = os.path.join(self.img_path, self.annotations["file_name"][id])
            self.imgs.append(cv2.imread(file_name))
        print(f"Loaded {len(self.imgs)} images")

        self.ids = ids
        self.transforms = transforms

    def load_ann(self, ann_path):
        rows = []
        with open(ann_path) as f:
            data = json.load(f)

            for row in data["images"]:
                file_name = row["file_name"]
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

                # Get all boxes and labels for the image
                boxes = []
                labels = []
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

                    boxes.append(annotation["bbox"])
                    labels.append(annotation["category_id"])

                rows.append([file_name, id, boxes, labels, scanner, width, height])

        return pd.DataFrame(
            rows,
            columns=[
                "file_name",
                "id",
                "boxes",
                "labels",
                "scanner",
                "width",
                "height",
            ],
        ).set_index("id")

    def __getitem__(self, index):
        # Load annotations
        id = self.ids[index % len(self.ids)]
        ann = self.annotations.loc[id]
        # file_name = os.path.join(self.img_path, ann["file_name"])
        boxes = ann["boxes"]
        labels = ann["labels"]  # 0 = positive, 1 = hard negative

        # Load image
        # img = cv2.imread(file_name)
        img = self.imgs[index % len(self.ids)]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transformations
        augmented = self.transforms(image=img, bboxes=boxes, class_labels=labels)
        img = augmented["image"] / 255

        # Convert targets into correct format
        if len(augmented["bboxes"]) > 0:
            boxes = torch.as_tensor(augmented["bboxes"], dtype=torch.float32)
            labels = torch.tensor(augmented["class_labels"], dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return img, target

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    from visualize import visualize

    # dm = MidogDataModule(crop_size=2000)
    # dm.setup()
    # t = dm.train_dataset

    transforms = A.Compose(
        [
            A.OneOf(
                [
                    # A.RandomCrop(width=256, height=256),
                    RandomCropIncludeBBox(width=256, height=256),
                ],
                p=1,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3
        ),
    )

    d = MigdogDataset(
        list(range(1, 11)), "../data/MIDOG.json", "../data/midog/", transforms, 1000
    )
    # print(len(d))
    # print(d[100])

    for i in range(10):
        img, target = d[i]
        img = img.numpy().transpose(1, 2, 0)
        box = target["boxes"].numpy()
        labels = target["labels"].numpy()
        visualize(img, box, labels)
