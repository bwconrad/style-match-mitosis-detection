import torch
from argparse import ArgumentParser
from pathlib import Path
import os
import sys
import torchvision.transforms as t
from torch.utils.data import DataLoader, dataloader

sys.path.append(os.path.join(os.getcwd()))

from src.data import SimpleDataset
from src.network import VGGEncoder


def load_data(path, resize_size, crop_size):
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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

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
        "-n", type=int, default=100, help="Number of samples to average from"
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

    # Load data
    dataloader = load_data(args.input, args.resize_size, args.crop_size)

    # Load model
    encoder = VGGEncoder()
    encoder = encoder.to(device)

    features = {
        "f1": [],
        "f2": [],
        "f3": [],
        "f4": [],
    }

    # Sample n times
    print(f"Getting average from {args.n} samples...")
    with torch.no_grad():
        n = 0
        while n < args.n:
            for img in dataloader:
                n += args.batch_size
                if n > args.n:
                    break

                img = img.to(device)

                f1, f2, f3, f4 = encoder(img, return_all=True)

                features["f1"].append(f1.detach().cpu())
                features["f2"].append(f2.detach().cpu())
                features["f3"].append(f3.detach().cpu())
                features["f4"].append(f4.detach().cpu())

    # Average samples
    features["f1"] = torch.mean(torch.cat(features["f1"][: args.n], dim=0), dim=0)
    features["f2"] = torch.mean(torch.cat(features["f2"][: args.n], dim=0), dim=0)
    features["f3"] = torch.mean(torch.cat(features["f3"][: args.n], dim=0), dim=0)
    features["f4"] = torch.mean(torch.cat(features["f4"][: args.n], dim=0), dim=0)

    torch.save(features, args.output)
    print(f"Saved to {args.output}")
