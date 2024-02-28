import os
import argparse
import time
import random
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffusers.training_utils import EMAModel

from src.utils import *
from src.data import get_encoded_dataset, filter_function
from src.models import model_from_config, optimizer_from_config, ObjectFeatureVQVAE


def main():
    parser = argparse.ArgumentParser(
        description="Train a generative model on scene bounding boxes, conditioned on scene graphs"
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--fvqvae_tag",
        type=str,
        required=True,
        help="Tag that refers to the fVQ-VAE experiment"
    )
    parser.add_argument(
        "--fvqvae_epoch",
        type=int,
        default=1999,
        help="Epoch of the pretrained fVQ-VAE"
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
        "--n_workers",
        type=int,
        default=4,
        help="The number of processed spawned by the batch provider (default=0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG (default=0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device to use for training"
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

    # Check if `output_dir` exists and if it doesn't create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create an experiment directory using the `tag`
    if args.tag is None:
        tag = time.strftime("%Y-%m-%d_%H:%M") + "_" + \
            os.path.split(args.config_file)[-1].split()[0]  # config file name
    else:
        tag = args.tag

    exp_dir = os.path.join(args.output_dir, tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save the parameters of this run to a file
    save_experiment_params(args, tag, exp_dir)

    # Parse the config file
    config: Dict[str, Dict[str, Any]] = load_config(args.config_file)

    # Load the training and validation datasets
    train_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        path_to_bounds=None,
        augmentations=config["data"].get("augmentations", None),
        split=config["training"].get("splits", ["train", "val"])
    )
    # Compute the bounds for this experiment, save them to a file in the
    # experiment directory and pass them to the validation dataset
    np.savez(
        os.path.join(exp_dir, "bounds.npz"),
        translations=train_dataset.bounds["translations"],
        sizes=train_dataset.bounds["sizes"],
        angles=train_dataset.bounds["angles"]
    )
    print(f"Training set has bounds: {train_dataset.bounds}")
    print(f"Load [{len(train_dataset)}] training scenes with [{train_dataset.n_object_types}] object types\n")

    config["data"]["encoding_type"] += "_eval"  # use the evaluation encoding for the validation dataset
    val_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        path_to_bounds=os.path.join(exp_dir, "bounds.npz"),
        augmentations=None,
        split=config["validation"].get("splits", ["test"])
    )
    print(f"Load [{len(val_dataset)}] validation scenes with [{val_dataset.n_object_types}] object types\n")

    # Make sure that the `train_dataset` and the `val_dataset` have the same number of object categories
    assert train_dataset.object_types == val_dataset.object_types

    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=args.n_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=args.n_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
        shuffle=False
    )

    # Load pretrained VQ-VAE weights; TODO: make it configurable
    print("Load pretrained VQ-VAE\n")
    with open(f"{args.output_dir}/{args.fvqvae_tag}/objfeat_bounds.pkl", "rb") as f:
        kwargs = pickle.load(f)
    vqvae_model = ObjectFeatureVQVAE("openshape_vitg14", "gumbel", **kwargs)
    ckpt_path = f"{args.output_dir}/{args.fvqvae_tag}/checkpoints/epoch_{args.fvqvae_epoch:05d}.pth"
    vqvae_model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    vqvae_model = vqvae_model.to(device)
    vqvae_model.eval()

    # Initialize the model and optimizer
    model = model_from_config(
        config["model"],
        train_dataset.n_object_types,
        train_dataset.n_predicate_types,
        **kwargs
    ).to(device)
    optimizer = optimizer_from_config(
        config["training"]["optimizer"],
        filter(lambda p: p.requires_grad, model.parameters())
    )

    # Save the model architecture to a file
    save_model_architecture(model, exp_dir)

    # Create EMA for the model
    ema_config = config["training"]["ema"]
    if ema_config["use_ema"]:
        print(f"Use exponential moving average (EMA) for model parameters\n")
        ema_states = EMAModel(
            model.parameters(),
            decay=ema_config["max_decay"],
            min_decay=ema_config["min_decay"],
            update_after_step=ema_config["update_after_step"],
            use_ema_warmup=ema_config["use_warmup"],
            inv_gamma=ema_config["inv_gamma"],
            power=ema_config["power"]
        )
        ema_states.to(device)
    else:
        ema_states: EMAModel = None

    # Load the weights from a previous run if specified
    start_epoch = 0
    start_epoch = load_checkpoints(model, ckpt_dir, ema_states, optimizer, args.checkpoint_epoch, device) + 1

    # Initialize the logger
    writer = SummaryWriter(os.path.join(exp_dir, "tensorboard"))

    # Log the stats to a log file
    StatsLogger.instance().add_output_file(open(
        os.path.join(exp_dir, "logs.txt"), "w"
    ))

    epochs = config["training"]["epochs"]
    steps_per_epoch = config["training"]["steps_per_epoch"]
    loss_weights = config["training"]["loss_weights"]
    save_freq = config["training"]["save_frequency"]  # in epochs
    log_freq = config["training"]["log_frequency"]    # in iterations
    val_freq = config["validation"]["frequency"]      # in epochs

    # Start training
    for i in range(start_epoch, epochs):
        model.train()

        for b, batch in zip(range(steps_per_epoch), yield_forever(train_loader)):
            # Move everything to the device
            for k, v in batch.items():
                if not isinstance(v, list):
                    batch[k] = v.to(device)
            # Zero previous gradients
            optimizer.zero_grad()
            # Compute the loss
            losses = model.compute_losses(batch, vqvae_model=vqvae_model)
            total_loss = torch.zeros(1, device=device)
            for k, v in losses.items():
                if k in loss_weights:
                    total_loss += loss_weights[k] * v
                else:  # weight is not specified
                    total_loss += v
            # Backpropagate
            total_loss.backward()
            # Update parameters
            optimizer.step()
            # Update EMA states
            if ema_states is not None:
                ema_states.step(model.parameters())

            StatsLogger.instance().update_loss(total_loss.item() * batch["objs"].shape[0], batch["objs"].shape[0])
            if (i * steps_per_epoch + b) % log_freq == 0:
                StatsLogger.instance().print_progress(i, b)
                writer.add_scalar("training/loss", total_loss.item(), i * steps_per_epoch + b)
                if len(losses) > 1:
                    for k, v in losses.items():
                        writer.add_scalar(f"training/{k}", v.item(), i * steps_per_epoch + b)
                if ema_states is not None:
                    writer.add_scalar("training/ema_decay", ema_states.cur_decay_value, i * steps_per_epoch + b)

        if (i+1) % save_freq == 0:
            save_checkpoints(model, optimizer, ckpt_dir, i, ema_states)
        StatsLogger.instance().clear()

        if (i+1) % val_freq == 0:
            print("\n================ Validation ================")
            # Evaluate with the EMA parameters if specified
            if ema_states is not None:
                ema_states.store(model.parameters())
                ema_states.copy_to(model.parameters())
            model.eval()

            with torch.no_grad():
                for val_b, val_batch in enumerate(val_loader):
                    # Move everything to the device
                    for k, v in val_batch.items():
                        if not isinstance(v, list):
                            val_batch[k] = v.to(device)
                    # Compute the loss
                    val_losses = model.compute_losses(batch, vqvae_model=vqvae_model)
                    val_total_loss = torch.zeros(1, device=device)
                    for k, v in val_losses.items():
                        if k in loss_weights:
                            val_total_loss += loss_weights[k] * v
                        else:  # weight is not specified
                            val_total_loss += v

                    StatsLogger.instance().update_loss(val_total_loss.item() * batch["objs"].shape[0], batch["objs"].shape[0])
                    StatsLogger.instance().print_progress(i, val_b)

            writer.add_scalar("validation/loss", StatsLogger.instance().loss, i * steps_per_epoch + b)
            if len(losses) > 1:
                for k, v in losses.items():
                    writer.add_scalar(f"validation/{k}", StatsLogger.instance()[k].value, i * steps_per_epoch + b)
            StatsLogger.instance().clear()

            # Restore the model parameters from EMA states
            if ema_states is not None:
                ema_states.restore(model.parameters())
            print("================ Validation ================\n")

if __name__ == "__main__":
    main()
