from typing import *
from torch import Tensor

import os
import argparse
import time
import random
import shutil

from PIL import Image
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models.alexnet import AlexNet_Weights

from src.utils.util import *
from src.data.splits_builder import CSVSplitsBuilder
from src.data.threed_front import CachedThreedFront


class ImageFolderDataset(Dataset):
    def __init__(self, directory: str, is_train=True):
        images = sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith("png")
        ])
        N = len(images) // 2

        start = 0 if is_train else N
        self.images = images[start:start+N]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx]


class SyntheticVsRealDataset(Dataset):
    def __init__(self, real: Dataset, synthetic: Dataset):
        self.N = min(len(real), len(synthetic))
        self.real = real
        self.synthetic = synthetic

    def __len__(self):
        return 2*self.N

    def __getitem__(self, idx: int):
        if idx < self.N:
            image_path = self.real[idx]
            label = 1
        else:
            image_path = self.synthetic[idx - self.N]
            label = 0

        img = Image.open(image_path)
        img = np.asarray(img).astype(np.float32) / np.float32(255)
        img = np.transpose(img[:, :, :3], (2, 0, 1))

        return torch.from_numpy(img), torch.tensor([label], dtype=torch.float)


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.fc = nn.Linear(9216, 1)

    def forward(self, x: Tensor):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = self.fc(x.view(len(x), -1))
        x = torch.sigmoid(x)

        return x


class AverageMeter:
    def __init__(self):
        self._value = 0
        self._cnt = 0

    def __iadd__(self, x: Union[Tensor, float]):
        if torch.is_tensor(x):
            self._value += x.sum().item()
            self._cnt += x.numel()
        else:
            self._value += x
            self._cnt += 1
        return self

    @property
    def value(self):
        return self._value / self._cnt


def main():
    parser = argparse.ArgumentParser(
        description=("Train a classifier to discriminate between real and synthetic scenes")
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--checkpoint_epoch",
        type=int,
        default=None,
        help="The epoch to load the checkpoint from"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Set the batch size for training and evaluating"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Set the PyTorch data loader workers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG (default=0)"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=30,
        help="Train for that many epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use for training"
    )

    args = parser.parse_args()

    # Set the random seed
    if args.seed is not None and args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"You have chosen to seed([{args.seed}]) the experiment")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    print(f"Run code on device [{device}]\n")

    # Create an experiment directory using the `tag`
    if args.tag is None:
        tag = time.strftime("%Y-%m-%d_%H:%M") + "_" + \
            os.path.split(args.config_file)[-1].split()[0]  # config file name
    else:
        tag = args.tag

    # Check if `save_dir` exists
    exp_dir = os.path.join(args.output_dir, tag)
    save_dir = os.path.join(exp_dir, "generated_scenes", f"epoch_{args.checkpoint_epoch:05d}")
    assert os.path.exists(save_dir), f"Path [{save_dir}] does not exist"

    ################################################################

    # Parse the config file
    config: Dict[str, Dict[str, Any]] = load_config(args.config_file)

    # Filter the dataset by split
    splits_builder = CSVSplitsBuilder(config["data"]["annotation_file"])
    train_3dfront_dataset = CachedThreedFront(
        base_dir=config["data"]["dataset_directory"],
        config=config["data"],
        scene_ids=splits_builder.get_splits(["train", "val"])
    )
    test_3dfront_dataset = CachedThreedFront(
        base_dir=config["data"]["dataset_directory"],
        config=config["data"],
        scene_ids=splits_builder.get_splits(["test"])
    )

    # Collect real images for the train split
    print(f"Collect real images from [{train_3dfront_dataset._base_dir}] train split")
    real_train_dir = os.path.join(train_3dfront_dataset._base_dir, "_train_blender_rendered_scene_256_topdown")
    if not os.path.exists(real_train_dir):
        os.makedirs(real_train_dir, exist_ok=True)
        real_images = [
            os.path.join(train_3dfront_dataset._base_dir, pi, "blender_rendered_scene_256", "topdown.png")
            for pi in train_3dfront_dataset._tags
        ]
        for path in real_images:
            name = path.split("/")[-3] + "_topdown.png"
            shutil.copyfile(path, os.path.join(real_train_dir, name))
    # Collect real images for the test split
    print(f"Collect real images from [{test_3dfront_dataset._base_dir}] test split")
    real_test_dir = os.path.join(test_3dfront_dataset._base_dir, "_test_blender_rendered_scene_256_topdown")
    if not os.path.exists(real_test_dir):
        os.makedirs(real_test_dir, exist_ok=True)
        real_images = [
            os.path.join(test_3dfront_dataset._base_dir, pi, "blender_rendered_scene_256", "topdown.png")
            for pi in test_3dfront_dataset._tags
        ]
        for path in real_images:
            name = path.split("/")[-3] + "_topdown.png"
            shutil.copyfile(path, os.path.join(real_test_dir, name))

    ################################################################

    # Collect synthesized images
    print(f"Collect synthesized images from [{save_dir}]")
    syn_dir = os.path.join(save_dir, "all_syns")
    if not os.path.exists(syn_dir):
        os.makedirs(syn_dir, exist_ok=True)
        syn_images = [
            os.path.join(save_dir, scene_id, "topdown.png")
            for scene_id in os.listdir(save_dir)
            if os.path.exists(os.path.join(save_dir, scene_id, "topdown.png"))
        ]
        for path in syn_images:
            name = os.path.split(path)[0].split("@")[-1] + "_topdown.png"
            shutil.copyfile(path, os.path.join(syn_dir, name))

    ################################################################

    # Prepare the datasets for training a classifier
    train_real = ImageFolderDataset(real_train_dir, is_train=True)
    test_real = ImageFolderDataset(real_test_dir, is_train=False)
    train_syn = ImageFolderDataset(syn_dir, is_train=True)
    test_syn = ImageFolderDataset(syn_dir, is_train=False)

    # Join them in useable datasets
    train_dataset = SyntheticVsRealDataset(train_real, train_syn)
    test_dataset = SyntheticVsRealDataset(test_real, test_syn)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers
    )

    # Create the classifier model
    print("Train a classifier model\n")
    model = AlexNet()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    max_val_acc, best_epoch = 0., None
    for e in range(args.n_epochs):
        loss_meter, acc_meter = AverageMeter(), AverageMeter()
        for i, (x, y) in enumerate(train_dataloader):
            model.train()
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = F.binary_cross_entropy(y_hat, y)
            acc = (torch.abs(y-y_hat) < 0.5).float().mean()
            loss.backward()
            optimizer.step()

            loss_meter += loss
            acc_meter += acc

            msg = "iter [{: 5d}] loss: {:.4f} | acc: {:.4f}".format(
                e * args.batch_size + i, loss_meter.value, acc_meter.value
            )
            print(msg + "\b"*len(msg), end="", flush=True)
        print()

        with torch.no_grad():
            model.eval()
            val_loss_meter, val_acc_meter = AverageMeter(), AverageMeter()
            for i, (x, y) in enumerate(test_dataloader):
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss = F.binary_cross_entropy(y_hat, y)
                acc = (torch.abs(y-y_hat) < 0.5).float().mean()

                if acc > max_val_acc:
                    max_val_acc = acc
                    best_epoch = e

                val_loss_meter += loss
                val_acc_meter += acc

                msg = "epoch [{: 2d}] val_loss: {:.4f} | val_acc: {:.4f}".format(
                    e, val_loss_meter.value, val_acc_meter.value
                )
                print(msg + "\b"*len(msg), end="", flush=True)
            print()

    print(f"\nClassification Accuracy: [{max_val_acc * 100:.4f}%] at epoch [{best_epoch}]\n")
    with open(os.path.join(save_dir, "classification_acc.txt"), "w") as f:
        f.write(f"Classification Accuracy: {max_val_acc * 100:.4f}%\n")


if __name__ == "__main__":
    main()
