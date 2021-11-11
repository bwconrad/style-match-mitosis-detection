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
from torch.utils.data import DataLoader
from torchvision import transforms

from .data_utils import RandomCropIncludeBBox


class MidogDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "data/midog/",
        ann_path: str = "data/MIDOG.json",
        train_scanners: Union[List[int], int] = 1,
        val_scanners: Union[List[int], int] = 1,
        test_scanners: Union[List[int], int] = 1,
        style_scanners: Union[List[int], int] = None,
        random_style_path: str = None,
        n_train_samples: int = 1500,
        n_val_samples: int = 500,
        crop_size: int = 256,
        n_val: int = 10,
        batch_size: int = 4,
        workers: int = 4,
    ):
        """Midog detection image data module

        Args:
            data_path: Path to image directory
            ann_path: Path to annotation file
            train_scanners: scanner index of training set
            val_scanners: scanner index of validation set
            test_scanners: scanner index of test set
            style_scanners: scanner index of style image set
            random_style_path: Path to style dataset if using random style trasnfer
            n_train_samples: Number of training samples
            n_val_samples: Number of validation samples
            crop_size: Size of random crop transformation
            n_val: Number of validation samples per class
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.data_path = data_path
        self.ann_path = ann_path
        self.train_scanners = train_scanners
        self.val_scanners = val_scanners
        self.test_scanners = test_scanners
        self.style_scanners = style_scanners
        self.random_style_path = random_style_path
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_val = n_val
        self.batch_size = batch_size
        self.workers = workers

        if isinstance(train_scanners, int):
            self.train_scanners = [train_scanners]
        else:
            self.train_scanners = train_scanners
        if isinstance(val_scanners, int):
            self.val_scanners = [val_scanners]
        else:
            self.val_scanners = val_scanners
        if isinstance(test_scanners, int):
            self.test_scanners = [test_scanners]
        else:
            self.test_scanners = test_scanners
        if isinstance(style_scanners, int):
            self.style_scanners = [style_scanners]
        else:
            self.style_scanners = style_scanners

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

        self.test_transforms = A.Compose(
            [
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3
            ),
        )

        self.style_train_transforms = A.Compose(
            [A.RandomCrop(crop_size, crop_size), ToTensorV2()]
        )

        self.style_val_transforms = A.Compose(
            [A.CenterCrop(crop_size, crop_size), ToTensorV2()]
        )
        self.style_rand_transforms = A.Compose(
            [A.Resize(crop_size, crop_size), ToTensorV2()]
        )


        self.style_test_transforms = A.Compose([ToTensorV2()])

    def setup(self, stage="fit"):
        all_ids = {
            1: list(range(1, 51)),
            2: list(range(51, 101)),
            3: list(range(101, 151)),
            4: list(range(151, 201)),
        }

        if stage == "fit":
            train_ids = []
            val_ids = []

            for s in self.train_scanners:
                train_ids.extend(all_ids[s][: -self.n_val])
            for s in self.val_scanners:
                val_ids.extend(all_ids[s][-self.n_val :])

            # Load data
            if self.style_scanners:
                style_ids_train = []
                style_ids_val = []
                for s in self.style_scanners:
                    style_ids_train .extend(all_ids[s][: -self.n_val])
                    style_ids_val .extend(all_ids[s][-self.n_val :])

                self.train_dataset = MigdogStyleDataset(
                    train_ids,
                    style_ids_train,
                    self.ann_path,
                    self.data_path,
                    self.train_transforms,
                    self.style_train_transforms,
                    self.n_train_samples,
                    train=True
                )
                self.val_dataset = MigdogStyleDataset(
                    val_ids,
                    style_ids_val,
                    self.ann_path,
                    self.data_path,
                    self.val_transforms,
                    self.style_val_transforms,
                    self.n_val_samples,
                    train=False
                )
            elif self.random_style_path:
                self.train_dataset = MigdogRandomStyleDataset(
                    train_ids,
                    self.ann_path,
                    self.data_path,
                    self.random_style_path,
                    self.train_transforms,
                    self.style_rand_transforms,
                    self.n_train_samples,
                    train=True
                )
                self.val_dataset = MigdogRandomStyleDataset(
                    val_ids,
                    self.ann_path,
                    self.data_path,
                    self.random_style_path,
                    self.val_transforms,
                    self.style_rand_transforms,
                    self.n_val_samples,
                    train=False
                )
            else:
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
            test_ids = []
            for s in self.test_scanners:
                test_ids.extend(all_ids[s][-self.n_val :])

            if self.style_scanners:
                style_ids = []
                for s in self.style_scanners:
                    style_ids.extend(all_ids[s][-self.n_val :])

                self.test_dataset = MigdogStyleDataset(
                    test_ids,
                    style_ids,
                    self.ann_path,
                    self.data_path,
                    self.test_transforms,
                    self.style_test_transforms,
                    len(test_ids),
                    train=False
                )
            elif self.random_style_path:
                self.test_dataset = MigdogRandomStyleDataset(
                    test_ids,
                    self.ann_path,
                    self.data_path,
                    self.random_style_path,
                    self.val_transforms,
                    self.style_rand_transforms,
                    self.n_val_samples,
                    train=False
                )
            else:
                self.test_dataset = MigdogDataset(
                    test_ids,
                    self.ann_path,
                    self.data_path,
                    self.test_transforms,
                    len(test_ids),
                )

    def train_dataloader(self):
        if self.style_scanners or self.random_style_path:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers,
                pin_memory=True,
                collate_fn=collate_fn_style,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )

    def val_dataloader(self):
        if self.style_scanners or self.random_style_path:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                collate_fn=collate_fn_style,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )

    def test_dataloader(self):
        if self.style_scanners or self.random_style_path:
            return DataLoader(
                self.test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                collate_fn=collate_fn_style,
            )
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=1,
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


def collate_fn_style(batch):
    images = list()
    targets = list()
    style_images = list()

    for b in batch:
        images.append(b[0])
        targets.append(b[1])
        style_images.append(b[2])

    images = torch.stack(images, dim=0)
    style_images = torch.stack(style_images, dim=0)

    return (images, targets, style_images)


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


class MigdogStyleDataset(data.Dataset):
    def __init__(
        self,
        ids: List[int],
        style_ids: List[int],
        ann_path: str,
        img_path: str,
        transforms: Callable,
        style_transforms: Callable,
        n_samples: int,
        train: bool=True,
    ):
        super().__init__()
        self.img_path = img_path
        self.n_samples = n_samples
        self.train = train

        # Load annotations
        self.annotations = self.load_ann(ann_path)

        # Preload images
        print("Loading images to memory...")
        self.imgs = {}
        all_ids = list(set(ids + style_ids))
        for id in all_ids:
            file_name = os.path.join(self.img_path, self.annotations["file_name"][id])
            self.imgs[id] = cv2.imread(file_name)
        print(f"Loaded {len(self.imgs)} images")

        self.ids = ids
        self.style_ids = style_ids

        self.transforms = transforms
        self.style_transforms = style_transforms

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
        if not self.train:
            id_style = self.style_ids[index % len(self.style_ids)]
        else:
            id_style = random.choice(self.style_ids)
        ann = self.annotations.loc[id]

        boxes = ann["boxes"]
        labels = ann["labels"]  # 0 = positive, 1 = hard negative

        # Load image
        img = self.imgs[id]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        style_img = self.imgs[id_style]
        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)

        # Apply transformations
        augmented = self.transforms(image=img, bboxes=boxes, class_labels=labels)
        img = augmented["image"] / 255

        style_img = self.style_transforms(image=style_img)["image"]
        style_img = style_img / 255

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

        return img, target, style_img

    def __len__(self):
        return self.n_samples


class MigdogRandomStyleDataset(data.Dataset):
    def __init__(
        self,
        ids: List[int],
        ann_path: str,
        img_path: str,
        style_path: str,
        transforms: Callable,
        style_transforms: Callable,
        n_samples: int,
        train: bool=True,
    ):
        super().__init__()
        self.img_path = img_path
        self.style_path = style_path
        self.n_samples = n_samples
        self.train = train

        # Load annotations
        self.annotations = self.load_ann(ann_path)

        # Preload images
        print("Loading images to memory...")
        self.imgs = {}
        for id in ids:
            file_name = os.path.join(self.img_path, self.annotations["file_name"][id])
            self.imgs[id] = cv2.imread(file_name)
        print(f"Loaded {len(self.imgs)} images")

        self.ids = ids
        self.style_files = os.listdir(style_path)

        self.transforms = transforms
        self.style_transforms = style_transforms

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
        if not self.train:
            idx_style = index % len(self.style_files)
        else:
            idx_style = random.randint(0, len(self.style_files)-1)
        ann = self.annotations.loc[id]

        boxes = ann["boxes"]
        labels = ann["labels"]  # 0 = positive, 1 = hard negative

        # Load image
        img = self.imgs[id]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        style_img = cv2.imread(os.path.join(self.style_path, self.style_files[idx_style]))
        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)

        # Apply transformations
        augmented = self.transforms(image=img, bboxes=boxes, class_labels=labels)
        img = augmented["image"] / 255

        style_img = self.style_transforms(image=style_img)["image"]
        style_img = style_img / 255

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

        return img, target, style_img

    def __len__(self):
        return self.n_samples

class SimpleDataset(data.Dataset):
    def __init__(self, root: str, ids: List[int], transforms: Callable):
        """Image dataset from directory

        Args:
            root: Path to directory
            indices: Start and end files indices to include
            transforms: Image augmentations
        """
        super().__init__()
        self.root = root
        self.paths = [sorted(os.listdir(root))[i - 1] for i in ids]
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.paths[index])).convert("RGB")
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.paths)


if __name__ == "__main__":
    from torchvision.utils import save_image
    from utils import visualize

    # dm = MidogDataModule(style_scanners=2)
    # dm.setup()
    # t = dm.train_dataset
    # img, tar, style_img = t[10]
    # save_image(img, "1.png")
    # save_image(style_img, "2.png")

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
    style_transforms = A.Compose(
            [A.RandomCrop(256, 256), ToTensorV2()]
        )

    d = MigdogRandomStyleDataset(
        list(range(1, 11)), "data/MIDOG.json", "data/midog/", "data/wikiart/" , transforms, style_transforms, 1000
    )
    img, _, img_s = d[0]
    save_image(img, "0.png")
    save_image(img_s, "1.png")


    # for i in range(10):
    #     img, target = d[i]
    #     img = img.numpy().transpose(1, 2, 0)
    #     box = target["boxes"].numpy()
    #     labels = target["labels"].numpy()
    #     visualize(img, box, labels)
