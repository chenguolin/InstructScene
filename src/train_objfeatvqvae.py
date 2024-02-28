import os
import argparse
import time
import random
import pickle
from copy import deepcopy

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from diffusers.training_utils import EMAModel

from src.utils import *
from src.data import filter_function
from src.data.threed_front import ThreedFront, parse_threed_front_scenes
from src.data.threed_future_dataset import ThreedFutureFeatureDataset
from src.models import model_from_config, optimizer_from_config


def main():
    parser = argparse.ArgumentParser(
        description="Train a VQ-VAE on object features"
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

    if os.getenv("PATH_TO_OBJFEATS_TRAIN") and os.getenv("PATH_TO_OBJFEATS_VAL"):
        assert config["model"]["objfeat_type"] in os.getenv("PATH_TO_OBJFEATS_TRAIN") and \
            config["model"]["objfeat_type"] in os.getenv("PATH_TO_OBJFEATS_VAL")
        print(f"Load pickled training 3D-FRONT object features from {os.getenv('PATH_TO_OBJFEATS_TRAIN')}")
        train_objects = pickle.load(open(os.getenv("PATH_TO_OBJFEATS_TRAIN"), "rb"))
        print(f"Load pickled validation 3D-FRONT object features from {os.getenv('PATH_TO_OBJFEATS_VAL')}")
        val_objects = pickle.load(open(os.getenv("PATH_TO_OBJFEATS_VAL"), "rb"))
    else:
        # Load all scenes in 3D-FRONT
        scenes = parse_threed_front_scenes(
            dataset_directory=config["data"]["path_to_3d_front_dataset_directory"],
            path_to_model_info=config["data"]["path_to_model_info"],
            path_to_models=config["data"]["path_to_3d_future_dataset_directory"]
        )

        # Collect objects used in three types of rooms
        ROOM_TYPES = ["bedroom", "bedroom", "diningroom", "livingroom"]  # two "bedroom" for implementation convenience
        scene_train_datasets, scene_val_datasets = [], []
        for i in range(1, len(ROOM_TYPES)):
            # Replace the room type in the config file
            for k, v in config["data"].items():
                config["data"][k] = v.replace(ROOM_TYPES[i-1], ROOM_TYPES[i])
            # Train
            filter_fn = filter_function(
                config["data"],
                config["training"].get("splits", ["train", "val"])
            )
            scene_train_datasets.append(
                ThreedFront([s for s in map(filter_fn, deepcopy(scenes)) if s])
            )
            print(f"Load [{ROOM_TYPES[i]}] train dataset with {len(scene_train_datasets[i-1])} rooms")
            # Validation
            filter_fn = filter_function(
                config["data"],
                config["validation"].get("splits", ["test"])
            )
            scene_val_datasets.append(
                ThreedFront([s for s in map(filter_fn, deepcopy(scenes)) if s])
            )
            print(f"Load [{ROOM_TYPES[i]}] validation dataset with {len(scene_val_datasets[i-1])} rooms")

        # Collect the set of objects in the scenes
        # Train
        train_objects = {}
        for scene_train_dataset in scene_train_datasets:
            for scene in scene_train_dataset:
                for obj in scene.bboxes:
                    train_objects[obj.model_jid] = obj
        train_objects = [vi for vi in train_objects.values()]
        with open(
            f"dataset/InstructScene/threed_front_objfeat_{config['model']['objfeat_type']}_train.pkl",
        "wb") as f:
            pickle.dump(train_objects, f)
        # Validation
        val_objects = {}
        for scene_val_dataset in scene_val_datasets:
            for scene in scene_val_dataset:
                for obj in scene.bboxes:
                    val_objects[obj.model_jid] = obj
        val_objects = [vi for vi in val_objects.values()]
        with open(
            f"dataset/InstructScene/threed_front_objfeat_{config['model']['objfeat_type']}_val.pkl",
        "wb") as f:
            pickle.dump(val_objects, f)

    train_jids = [obj.model_jid for obj in train_objects]
    val_jids = [obj.model_jid for obj in val_objects]
    only_val_jids = list(set(val_jids) - set(train_jids))
    only_val_objects = [obj for obj in val_objects if obj.model_jid in only_val_jids]
    all_jids = train_jids + only_val_jids
    all_objects = train_objects + only_val_objects

    train_dataset = ThreedFutureFeatureDataset(
        all_objects,  # not split into train and val
        objfeat_type=config["model"]["objfeat_type"]
    )
    print(f"\nLoad [{len(train_dataset)}] training objects")

    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=args.n_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )

    # Compute the value bounds of training object features, saved as model parameters
    train_objfeats = [
        eval(f"obj.{config['model']['objfeat_type']}_features")
        for obj in train_objects
    ]
    train_objfeats = np.stack(train_objfeats, axis=0)
    kwargs = {
        "objfeat_min": train_objfeats.min(),
        "objfeat_max": train_objfeats.max()
    }
    with open(os.path.join(exp_dir, "objfeat_bounds.pkl"), "wb") as f:
        pickle.dump(kwargs, f)

    # Initialize the model and optimizer
    model = model_from_config(config["model"], **kwargs).to(device)
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
            losses = model.compute_losses(batch)
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

            StatsLogger.instance().update_loss(total_loss.item() * batch["objfeats"].shape[0], batch["objfeats"].shape[0])
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

if __name__ == "__main__":
    PATH_TO_SCENES = "dataset/InstructScene/threed_front.pkl"
    PATH_TO_OBJFEATS_TRAIN = "dataset/InstructScene/threed_front_objfeat_openshape_vitg14_train.pkl"
    PATH_TO_OBJFEATS_VAL = "dataset/InstructScene/threed_front_objfeat_openshape_vitg14_val.pkl"

    if os.path.exists(PATH_TO_SCENES):
        os.environ["PATH_TO_SCENES"] = PATH_TO_SCENES
    if os.path.exists(PATH_TO_OBJFEATS_TRAIN) and os.path.exists(PATH_TO_OBJFEATS_VAL):
        os.environ["PATH_TO_OBJFEATS_TRAIN"] = PATH_TO_OBJFEATS_TRAIN
        os.environ["PATH_TO_OBJFEATS_VAL"] = PATH_TO_OBJFEATS_VAL

    main()
