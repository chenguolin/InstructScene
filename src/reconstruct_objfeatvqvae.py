import os
import argparse
import time
import random
import pickle
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image
from diffusers.training_utils import EMAModel

from src.utils import *
from src.data import filter_function
from src.data.threed_front import ThreedFront, parse_threed_front_scenes
from src.data.threed_future_dataset import ThreedFutureFeatureDataset
from src.models import model_from_config


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
        "--n_saved",
        type=int,
        default=5,
        help="The number of batches to save"
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

    # Collect all training and validation object features for retrieval
    all_objfeats = {
        obj.model_jid: eval(f"obj.{config['model']['objfeat_type']}_features")
        for obj in train_objects + val_objects
    }
    all_objfeats = [[k ,v] for k, v in all_objfeats.items()]  # unique object ids and features
    jids = np.array([v[0] for v in all_objfeats])
    objfeats = torch.from_numpy(
        np.stack([v[1] for v in all_objfeats], axis=0)
    ).float().to(device)  # (M, D)

    train_jids = [obj.model_jid for obj in train_objects]
    val_jids = [obj.model_jid for obj in val_objects]
    only_val_jids = list(set(val_jids) - set(train_jids))
    only_val_objects = [obj for obj in val_objects if obj.model_jid in only_val_jids]
    all_jids = train_jids + only_val_jids
    all_objects = train_objects + only_val_objects

    dataset = ThreedFutureFeatureDataset(
        all_objects,  # not split into train and val
        objfeat_type=config["model"]["objfeat_type"]
    )
    print(f"Load [{len(dataset)}] validation objects\n")

    dataloader = DataLoader(
        dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=args.n_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        shuffle=False
    )

    # Load the object feature bounds
    with open(os.path.join(exp_dir, "objfeat_bounds.pkl"), "rb") as f:
        kwargs = pickle.load(f)

    # Initialize the model
    model = model_from_config(config["model"], **kwargs).to(device)

    # Create EMA for the model
    ema_config = config["training"]["ema"]
    if ema_config["use_ema"]:
        ema_states = EMAModel(model.parameters())
        ema_states.to(device)
    else:
        ema_states: EMAModel = None

    # Load the weights from a checkpoint
    load_epoch = load_checkpoints(model, ckpt_dir, ema_states, epoch=args.checkpoint_epoch, device=device)

    # Evaluate with the EMA parameters if specified
    if ema_states is not None:
        print(f"Copy EMA parameters to the model\n")
        ema_states.copy_to(model.parameters())
    model.eval()

    # Check if `save_dir` exists and if it doesn't create it
    save_dir = os.path.join(exp_dir, "reconstruct_objfeats", f"epoch_{load_epoch:05d}")
    os.makedirs(save_dir, exist_ok=True)

    # Reconstruct the object features
    rev_count, rev_correct_count = 0, 0
    for batch_idx, batch in tqdm(
        enumerate(dataloader),
        desc=f"Reconstruct object features",
        total=len(dataloader), ncols=125
    ):
        with torch.no_grad():
            # Move everything to the device
            for k, v in batch.items():
                if not isinstance(v, list):
                    batch[k] = v.to(device)
            true_jids = batch["jids"]  # (B,)
            # Reconstruction
            rec_features = model.reconstruct(batch["objfeats"])  # (B, D)
            rec_features = F.normalize(rec_features, dim=-1)
            # Retrieve the cosine-similarity closest object features
            sim = torch.matmul(rec_features, objfeats.T)  # (B, M)
            rev_jids = jids[torch.argmax(sim, dim=-1).cpu()]  # (B,)
            # Evaluate the retrieval performance
            for true_jid, rev_jid in zip(true_jids, rev_jids):
                rev_count += 1
                if true_jid == rev_jid:
                    rev_correct_count += 1
            # Save the retrieved object image pairs
            if batch_idx < args.n_saved:
                image_path = "dataset/3D-FRONT/3D-FUTURE-model/{}/image.jpg"
                true_images = torch.stack([
                    T.ToTensor()(
                        Image.open(image_path.format(true_jid)).resize((256, 256))
                    ) for true_jid in true_jids
                ], dim=0)  # (B, 3, 256, 256)
                rec_images = torch.stack([
                    T.ToTensor()(
                        Image.open(image_path.format(rev_jid)).resize((256, 256))
                    ) for rev_jid in rev_jids
                ], dim=0)  # (B, 3, 256, 256)
                save_image(
                    torch.cat([true_images, rec_images], dim=0),
                    os.path.join(save_dir, f"batch_{batch_idx:03d}.jpg"), nrow=len(batch["jids"])
                )            

    eval_info = f"Retrieval accuracy: {rev_correct_count / rev_count * 100:.2f}% ({rev_correct_count}/{rev_count})"
    with open(os.path.join(save_dir, "eval.txt"), "w") as f:
        f.write(eval_info)
    print(eval_info)


if __name__ == "__main__":
    # Pickle files have been prepared by the training script
    os.environ["PATH_TO_SCENES"] = "dataset/InstructScene/threed_front.pkl"
    os.environ["PATH_TO_OBJFEATS_TRAIN"] = "dataset/InstructScene/threed_front_objfeat_openshape_vitg14_train.pkl"
    os.environ["PATH_TO_OBJFEATS_VAL"] = "dataset/InstructScene/threed_front_objfeat_openshape_vitg14_val.pkl"

    main()
