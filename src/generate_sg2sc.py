import os
import argparse
import random
import pickle

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from diffusers.training_utils import EMAModel

from src.utils.util import *
from src.utils.visualize import *
from src.data import filter_function, get_encoded_dataset, get_dataset_raw_and_encoded
from src.data.threed_future_dataset import ThreedFutureDataset
from src.data.threed_front_dataset_base import trs_to_corners
from src.data.utils_text import compute_loc_rel, reverse_rel
from src.models import model_from_config, ObjectFeatureVQVAE


def main():
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
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
        "--n_scenes",
        type=int,
        default=5,
        help="The number of scenes to be generated"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the generated scenes"
    )
    parser.add_argument(
        "--eight_views",
        action="store_true",
        help="Render 8 views of the scene"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Resolution of the rendered image"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more information"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use for training"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.,
        help="scale for the classifier-free guidance"
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

    # Check if `ckpt_dir` exists
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    assert os.path.exists(ckpt_dir), f"Checkpoint directory {ckpt_dir} does not exist."

    config = load_config(args.config_file)

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        config["data"]["path_to_pickled_3d_futute_models"]
    )
    print(f"Load [{len(objects_dataset)}] 3D-FUTURE models")

    if "eval" not in config["data"]["encoding_type"]: config["data"]["encoding_type"] += "_eval"
    # Load training dataset to compute prior statistics for VAE
    train_dataset = get_encoded_dataset(
        config["data"],  # same encoding type as validation dataset
        filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        path_to_bounds=None,
        augmentations=None,  # no need for prior statistics computation
        split=config["training"].get("splits", ["train", "val"])
    )
    # Compute the bounds for this experiment, save them to a file in the
    # experiment directory and pass them to the validation dataset
    if not os.path.exists(os.path.join(exp_dir, "bounds.npz")):
        np.savez(
            os.path.join(exp_dir, "bounds.npz"),
            translations=train_dataset.bounds["translations"],
            sizes=train_dataset.bounds["sizes"],
            angles=train_dataset.bounds["angles"]
        )
    print(f"Training set has bounds: {train_dataset.bounds}")
    print(f"Load [{len(train_dataset)}] training scenes with [{train_dataset.n_object_types}] object types\n")

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        path_to_bounds=os.path.join(exp_dir, "bounds.npz"),
        augmentations=None,
        split=config["validation"].get("splits", ["test"])
    )
    print(f"Load [{len(dataset)}] validation scenes with [{dataset.n_object_types}] object types\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=args.n_workers,
        pin_memory=False,
        collate_fn=train_dataset.collate_fn,
        shuffle=False  # no need for prior statistics computation
    )
    if args.n_scenes == 0:  # use all scenes
        B = config["validation"]["batch_size"]
    else:
        B = args.n_scenes
    dataloader = DataLoader(
        dataset,
        batch_size=B,
        num_workers=args.n_workers,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
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

    # Initialize the model
    model = model_from_config(
        config["model"],
        raw_dataset.n_object_types,
        raw_dataset.n_predicate_types
    ).to(device)

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
    save_dir = os.path.join(exp_dir, "generated_scenes", f"epoch_{load_epoch:05d}")
    os.makedirs(save_dir, exist_ok=True)

    # Generate the boxes from scene graphs
    classes = np.array(dataset.object_types)
    rel_counts, rel_count_errors = 0, 0
    print("Sample boxes from scene graphs with the [diffusion model]")
    for batch_idx, batch in tqdm(
        enumerate(dataloader),
        desc=f"Process each batch",
        total=len(dataloader), ncols=125,
        disable=args.verbose
    ):
        # Move everything to the device
        for k, v in batch.items():
            if not isinstance(v, list):
                batch[k] = v.to(device)
        # Unpack the batch parameters
        objs = batch["objs"]
        edges = batch["edges"]
        objfeat_vq_indices = batch["objfeat_vq_indices"]
        obj_masks = batch["obj_masks"]
        # Generate the box parameters
        with torch.no_grad():
            boxes_pred = model.generate_samples(
                objs, edges, objfeat_vq_indices, obj_masks,
                vqvae_model,
                cfg_scale=args.cfg_scale
            )

        # Decode objfeat indices to objfeat embeddings
        B, N = objfeat_vq_indices.shape[:2]
        objfeats = vqvae_model.reconstruct_from_indices(
            objfeat_vq_indices.reshape(B*N, -1)
        ).reshape(B, N, -1)
        objfeats = (objfeats * obj_masks[..., None].float()).cpu().numpy()

        objs = objs.cpu()
        obj_masks = obj_masks.cpu()
        boxes_pred = boxes_pred.cpu()  # (B, N, 8)

        bbox_params = {
            "class_labels": F.one_hot(objs, num_classes=raw_dataset.n_object_types+1).float(),  # +1 for empty node (not really used)
            "translations": boxes_pred[..., :3],
            "sizes": boxes_pred[..., 3:6],
            "angles": boxes_pred[..., 6:]
        }
        boxes = dataset.post_process(bbox_params)
        bbox_params_t = torch.cat([
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ], dim=-1).numpy()
        assert bbox_params_t.shape[-1] == 7 + raw_dataset.n_object_types+1

        # Evaluate (and visualize) each scene in the batch
        progress_bar = tqdm(
            total=len(bbox_params_t),
            desc="Visualize each scene",
            ncols=125,
            disable=args.verbose
        )
        for i in range(len(bbox_params_t)):
            # Get the textured objects by retrieving the 3D models
            trimesh_meshes, bbox_meshes, obj_classes, obj_sizes, _ = get_textured_objects(
                bbox_params_t[i],
                objects_dataset, classes,
                objfeats[i] if model.use_objfeat else None,
                "openshape_vitg14",  # TODO: make it configurable
                verbose=args.verbose
            )

            # Evaluation for scene graph condition
            # 1. Get the decoded scene graphs from generated boxes
            obj_class_ids = [
                dataset.object_types.index(c) if c is not None
                else dataset.n_object_types
                for c in obj_classes
            ]
            relations = []  # [[cls_id1, pred_id, cls_id2], ...]
            cls_dim = dataset.n_object_types+1
            for idx in range(len(obj_class_ids)):
                if obj_class_ids[idx] == dataset.n_object_types:  # empty object
                    continue
                c1_id = obj_class_ids[idx]
                t1 = bbox_params_t[i, idx, cls_dim:cls_dim+3]
                r1 = bbox_params_t[i, idx, cls_dim+6]
                s1 = obj_sizes[idx]  # bbox_params_t[i, idx, cls_dim+3:cls_dim+6]; use the retrieved size
                corners1 = trs_to_corners(t1, r1, s1)
                name1 = dataset.object_types[c1_id]
                for other_idx in range(idx+1, len(obj_class_ids)):
                    if obj_class_ids[other_idx] == dataset.n_object_types:  # empty object
                        continue 
                    c2_id = obj_class_ids[other_idx]
                    t2 = bbox_params_t[i, other_idx, cls_dim:cls_dim+3]
                    r2 = bbox_params_t[i, other_idx, cls_dim+6]
                    s2 = obj_sizes[other_idx]  # bbox_params_t[i, other_idx, cls_dim+3:cls_dim+6]; use the retrieved size
                    corners2 = trs_to_corners(t2, r2, s2)
                    name2 = dataset.object_types[c2_id]

                    loc_rel_str = compute_loc_rel(corners1, corners2, name1, name2)
                    if loc_rel_str is not None:
                        # print(classes[obj_class_ids[idx]], loc_rel_str, classes[obj_class_ids[other_idx]])
                        relation_id = dataset.predicate_types.index(loc_rel_str)
                        relations.append([idx, relation_id, other_idx])
                        # Add the reverse relation
                        # print(classes[obj_class_ids[other_idx]], reverse_rel(loc_rel_str), classes[obj_class_ids[idx]])
                        rev_relation_id = dataset.predicate_types.index(reverse_rel(loc_rel_str))
                        relations.append([other_idx, rev_relation_id, idx])

            # 2. Compare the decoded scene graph with the ground truth
            true_edges = batch["edges"][i].cpu().numpy()
            edges = dataset.n_predicate_types * np.ones((len(obj_class_ids), len(obj_class_ids)), dtype=np.int64)  # (n, n)
            for s, p, o in relations:
                edges[s, o] = p
            obj_masks = batch["obj_masks"][i].cpu().numpy()  # (n,)
            edge_masks = obj_masks[:, None] * obj_masks[None, :] * \
                (~np.eye(len(obj_class_ids), dtype=bool)).astype(np.int64)  # (n, n)
            true_edges = true_edges * edge_masks
            edges = edges * edge_masks

            num_objs = batch["obj_masks"][i].cpu().numpy().sum()
            assert edge_masks.sum() == num_objs * (num_objs - 1)
            assert (edges != true_edges).sum() % 2 == 0  # because of the reverse relations

            rel_counts += edge_masks.sum()
            rel_count_errors += (edges != true_edges).sum()

            # print("rel_count_errors", (edges != true_edges).sum(), " |  rel_counts", edge_masks.sum(), "\n")
            # predicate_types = dataset.predicate_types + ["empty"]
            # for II in range(len(obj_class_ids)):
            #     for JJ in range(II+1, len(obj_class_ids)):
            #         if edges[II, JJ] != true_edges[II, JJ]:
            #             print("[   edges  ]", classes[obj_class_ids[II]], predicate_types[edges[II, JJ]], classes[obj_class_ids[JJ]])
            #             print("[true_edges]", classes[obj_class_ids[II]], predicate_types[true_edges[II, JJ]], classes[obj_class_ids[JJ]])
            #             print()

            progress_bar.update(1)
            progress_bar.set_postfix({
                "rel_error": "{:.4f}".format(rel_count_errors/rel_counts)
            })

            # Whether to visualize the scene by blender rendering
            if not args.visualize:
                continue

            # To get the manually created floor plan, which includes vertices of all meshes in the scene
            all_vertices = np.concatenate([
                tr_mesh.vertices for tr_mesh in trimesh_meshes
            ], axis=0)
            x_max, x_min = all_vertices[:, 0].max(), all_vertices[:, 0].min()
            z_max, z_min = all_vertices[:, 2].max(), all_vertices[:, 2].min()

            tr_floor, _ = floor_plan_from_scene(
                raw_dataset[0], config["data"]["path_to_floor_plan_textures"],  # `raw_dataset[0]` is not really used
                without_room_mask=True,
                rectangle_floor=True, room_size=[x_min, z_min, x_max, z_max]
            )
            trimesh_meshes.append(tr_floor)

            # Create a trimesh scene and export it to a temporary directory
            ii = batch_idx * B + i
            export_dir = os.path.join(save_dir, f"{ii:04d}@{batch['scene_uids'][i]}_cfg{args.cfg_scale:.1f}")
            tmp_dir = os.path.join(export_dir, "tmp")
            os.makedirs(export_dir, exist_ok=True)
            os.makedirs(tmp_dir, exist_ok=True)
            export_scene(tmp_dir, trimesh_meshes, bbox_meshes)

            # Render the exported scene by calling blender
            blender_render_scene(
                tmp_dir,
                export_dir,
                top_down_view=(not args.eight_views),
                resolution_x=args.resolution,
                resolution_y=args.resolution
            )
            # Delete the temporary directory
            os.system(f"rm -rf {tmp_dir}")
            if args.verbose:
                print(f"Save the scene to {export_dir}\n")

        # Not generate all scenes
        if args.n_scenes != 0:  # only generate the first `n_scenes` scenes
            break

    # Save the evaluation results
    eval_info = f"Relation count error: [{rel_count_errors:4d}/{rel_counts:4d}] = {rel_count_errors/rel_counts:.4f}\n"
    with open(os.path.join(save_dir, f"eval_cfg{args.cfg_scale:.1f}.txt"), "w") as f:
        f.write(eval_info)

if __name__ == "__main__":
    main()
