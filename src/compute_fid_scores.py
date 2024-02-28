import os
import argparse
import time
import random
import shutil

import torch
from cleanfid import fid

from src.utils import *
from src.data.splits_builder import CSVSplitsBuilder
from src.data.threed_front import CachedThreedFront


def main():
    parser = argparse.ArgumentParser(
        description="Compute the FID/KID scores between the real and the synthetic images"
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
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG (default=0)"
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Whether to use CPU or not"
    )
    parser.add_argument(
        "--use_ground_truth",
        action="store_true",
        help="Whether to use ground truth images to compute FID/KID scores"
    )

    args = parser.parse_args()

    # A warning message for use ground truth
    if args.use_ground_truth:
        print(
            "You have chosen to use ground truth images from the train split to compute FID/KID scores, " +
            "which are expected to be the lower bound of the scores. " +
            "However, it is not true for the evaluation of synthesized images with prompts, " +
            "since their distributions could be more similar to the test images with the help of text conditions " +
            "than the ground truth images in the training split.\n\n"
        )

    # Set the random seed
    if args.seed is not None and args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"You have chosen to seed([{args.seed}]) the experiment")

    if not args.use_cpu:
        device = torch.device("cuda")  # `fid` only supports cuda:0
    else:
        device = torch.device("cpu")
    print(f"Run code on device [{'cpu' if args.use_cpu else 'cuda'}]\n")

    if not args.use_ground_truth:
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
    dataset = CachedThreedFront(
        base_dir=config["data"]["dataset_directory"],
        config=config["data"],
        scene_ids=splits_builder.get_splits(["test"])
    )

    # Collect real images from the test split
    print(f"Collect real images from [{dataset._base_dir}]")
    real_dir = os.path.join(dataset._base_dir, "_test_blender_rendered_scene_256_topdown")
    if not os.path.exists(real_dir) or len(os.listdir(real_dir)) == 0:
        os.makedirs(real_dir, exist_ok=True)
        real_images = [
            os.path.join(dataset._base_dir, pi, "blender_rendered_scene_256", "topdown.png")  # TODO: support other views
            for pi in dataset._tags
        ]
        for path in real_images:
            name = path.split("/")[-3] + "_topdown.png"
            shutil.copyfile(path, os.path.join(real_dir, name))
        num_real_images = len(real_images)
    else:
        num_real_images = len(os.listdir(real_dir))
    print(f"Found [{num_real_images}] real images\n")

    ################################################################

    # Collect synthesized images
    if not args.use_ground_truth:
        print(f"Collect synthesized images from [{save_dir}]")
        syn_dir = os.path.join(save_dir, "all_syns")
        if not os.path.exists(syn_dir) or len(os.listdir(syn_dir)) == 0:
            os.makedirs(syn_dir, exist_ok=True)
            syn_images = [
                os.path.join(save_dir, scene_id, "topdown.png")  # TODO: support other views
                for scene_id in os.listdir(save_dir)
                if os.path.exists(os.path.join(save_dir, scene_id, "topdown.png"))
            ]
            for path in syn_images:
                name = os.path.split(path)[0].split("@")[-1] + "_topdown.png"
                shutil.copyfile(path, os.path.join(syn_dir, name))
            num_syn_images = len(syn_images)
        else:
            num_syn_images = len(os.listdir(syn_dir))
        print(f"Found [{num_syn_images}] synthesized images\n")
    # Collect real images from the train split as synthesized images
    else:
        print(f"Collect real images as synthesized images from [{dataset._base_dir}]")
        train_dataset = CachedThreedFront(
            base_dir=config["data"]["dataset_directory"],
            config=config["data"],
            scene_ids=splits_builder.get_splits(["train", "val"])
        )
        syn_dir = os.path.join(dataset._base_dir, "_train_blender_rendered_scene_256_topdown")
        if not os.path.exists(syn_dir) or len(os.listdir(syn_dir)) == 0:
            os.makedirs(syn_dir, exist_ok=True)
            syn_images = [
                os.path.join(dataset._base_dir, pi, "blender_rendered_scene_256", "topdown.png")  # TODO: support other views
                for pi in train_dataset._tags
            ]
            print(f"Found [{len(syn_images)}] ground truth images, randomly sample [{num_real_images}] images as synthesized images\n")
            syn_images = random.sample(syn_images, num_real_images)
            for path in syn_images:
                name = path.split("/")[-3] + "_topdown.png"
                shutil.copyfile(path, os.path.join(syn_dir, name))
            num_syn_images = len(syn_images)
        else:
            print(f"Found [{len(os.listdir(syn_dir))}] ground truth images synthesized images\n")
            num_syn_images = len(os.listdir(syn_dir))

    assert num_real_images == len(dataset._tags), \
        f"Number of real images ({num_real_images}) does not match the number of scene ids ({len(dataset._tags)})"
    # assert num_syn_images == num_real_images, \
    #     f"Number of synthesized images ({num_syn_images}) does not match the number of real images ({num_real_images})"

    ################################################################

    # Compute FID/KID scores
    print("Compute FID/KID scores\n")
    eval_info = ""
    configs = {
        "fdir1": real_dir,
        "fdir2": syn_dir,
        "device": device
    }

    fid_score = fid.compute_fid(**configs)
    print(f"FID score: {fid_score:.6f}\n")
    eval_info += f"FID score: {fid_score:.6f}\n"
    clip_fid_score = fid.compute_fid(model_name="clip_vit_b_32", **configs)
    print(f"CLIP-FID score: {clip_fid_score:.6f}\n")
    eval_info += f"CLIP-FID score: {clip_fid_score:.6f}\n"
    kid_score = fid.compute_kid(**configs)
    print(f"KID score: {kid_score:.6f}\n")
    eval_info += f"KID score: {kid_score:.6f}\n"

    print("\n" + eval_info)
    if not args.use_ground_truth:
        with open(os.path.join(save_dir, "fid_scores.txt"), "w") as f:
            f.write(eval_info)


if __name__ == "__main__":
    main()
