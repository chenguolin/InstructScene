from typing import *
from argparse import Namespace
from torch.nn import Module
from torch.optim import Optimizer
from diffusers.training_utils import EMAModel

import os
import json
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import torch


# Copied from https://github.com/huggingface/pytorch-image-models/timm/data/loader.py
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def yield_forever(iterator: Iterator[Any]):
    while True:
        for x in iterator:
            yield x


def load_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def save_experiment_params(args: Namespace, experiment_tag: str, directory: str) -> None:
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    params["experiment_tag"] = experiment_tag
    for k, v in list(params.items()):
        if v == "":
            params[k] = None
    if hasattr(args, "config_file"):
        config = load_config(args.config_file)
        params.update(config)
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


def save_model_architecture(model: Module, directory: str) -> None:
    """Save the model architecture to a `.txt` file."""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    message = f'Number of trainable / all parameters: {num_trainable_params} / {num_params}\n\n' + str(model)

    with open(os.path.join(directory, 'model.txt'), 'w') as f:
        f.write(message)


def load_checkpoints(
    model: Module,
    ckpt_dir: str,
    ema_states: Optional[EMAModel]=None,
    optimizer: Optional[Optimizer]=None,
    epoch: Optional[int]=None,
    device=torch.device("cpu")
) -> int:
    """Load checkpoint from the given experiment directory and return the epoch of this checkpoint."""
    if epoch is not None and epoch < 0:
        epoch = None

    model_files = [f.split(".")[0] for f in os.listdir(ckpt_dir)
        if f.startswith("epoch_") and f.endswith(".pth")]

    if len(model_files) == 0:  # no checkpoints found
        print(f"No checkpoint found in {ckpt_dir}, starting from scratch\n")
        return -1

    epoch = epoch or max([int(f[6:]) for f in model_files])  # load the latest checkpoint by default
    checkpoint_path = os.path.join(ckpt_dir, f"epoch_{epoch:05d}.pth")
    if not os.path.exists(checkpoint_path):  # checkpoint file not found
        print(f"Checkpoint file {checkpoint_path} not found, starting from scratch\n")
        return -1

    print(f"Load checkpoint from {checkpoint_path}\n")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    if ema_states is not None:
        ema_states.load_state_dict(checkpoint["ema_states"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return epoch


def save_checkpoints(model: Module, optimizer: Optimizer, ckpt_dir: str, epoch: int, ema_states: Optional[EMAModel]=None) -> None:
    """Save checkpoint to the given experiment directory."""
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    if ema_states is not None:
        save_dict["ema_states"] = ema_states.state_dict()

    save_path = os.path.join(ckpt_dir, f"epoch_{epoch:05d}.pth")
    torch.save(save_dict, save_path)
