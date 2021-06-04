import torch
from argparse import ArgumentParser
from pathlib import Path
import os
import sys
import torchvision.transforms as t
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd()))

from src.data import SimpleDataset
from src.network import VGGEncoder


def load_data(path, resize_size, crop_size, batch_size):
    if resize_size == 0:
        transforms = t.Compose(
            [
                t.RandomCrop(crop_size),
                t.RandomHorizontalFlip(),
                t.ToTensor(),
            ]
        )
    else:
        transforms = t.Compose(
            [
                t.Resize(resize_size),
                t.RandomCrop(crop_size),
                t.RandomHorizontalFlip(),
                t.ToTensor(),
            ]
        )

    dataset = SimpleDataset(path, [0, len(os.listdir(path))], transforms)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    print(f"Loading data from {path}\nLoaded {len(dataset)} images")

    return dataloader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=Path, help="Path to data directory.", required=True
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Path to save to", required=True
    )
    parser.add_argument(
        "--n_batches", type=int, default=8, help="Number of batches to average from"
    )
    parser.add_argument(
        "--resize_size",
        type=int,
        default=0,
        help="Size of resize transformation (0 = no resizing)",
    )
    parser.add_argument(
        "--crop_size", type=int, default=256, help="Size of crop transformation"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_n = args.n_batches * args.batch_size

    # Load data
    dataloader = load_data(
        args.input, args.resize_size, args.crop_size, args.batch_size
    )

    # Load model
    encoder = VGGEncoder()
    encoder = encoder.to(device)

    features = {}

    # Sample max_n times
    print(f"Getting average from {max_n} samples...")
    with torch.no_grad():
        n = 0
        while n < max_n:
            for img in dataloader:
                if n >= max_n:
                    break

                img = img.to(device)

                f1, f2, f3, f4 = encoder(img, return_all=True)

                if features == {}:
                    features["f1"] = f1.sum(0).detach().cpu()
                    features["f2"] = f2.sum(0).detach().cpu()
                    features["f3"] = f3.sum(0).detach().cpu()
                    features["f4"] = f4.sum(0).detach().cpu()
                else:
                    features["f1"] += f1.sum(0).detach().cpu()
                    features["f2"] += f2.sum(0).detach().cpu()
                    features["f3"] += f3.sum(0).detach().cpu()
                    features["f4"] += f4.sum(0).detach().cpu()

                n += args.batch_size

    # Average samples
    features["f1"] /= max_n
    features["f2"] /= max_n
    features["f3"] /= max_n
    features["f4"] /= max_n

    torch.save(
        [features["f1"], features["f2"], features["f3"], features["f4"]], args.output
    )
    print(f"Saved to {args.output}")
