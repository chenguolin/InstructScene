from .utils_text import reverse_rel
from .threed_front_dataset_base import *


class SG2SC(DatasetDecoratorBase):
    def __init__(self, dataset, objfeat_type=None):
        super().__init__(dataset)
        self.objfeat_type = objfeat_type

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        sample_params_new = {}
        for k, v in sample_params.items():
            if k == "class_labels":
                class_labels = np.copy(v)
                class_ids = np.argmax(class_labels, axis=-1).astype(np.int64)
                sample_params_new["objs"] = class_ids

        # Load information file for every object
        with open(sample_params["models_info_path"], "rb") as f:
            models_info = pickle.load(f)
        objfeat_vq_indices = [
            np.array(model_info["objfeat_vq_indices"])
            for model_info in models_info
        ]
        # Permutation augmentation
        if "permutation" in sample_params:
            objfeat_vq_indices = [objfeat_vq_indices[i] for i in sample_params["permutation"]]

        sample_params_new["objfeat_vq_indices"] = np.vstack(objfeat_vq_indices)  # (n, k)

        sample_params.update(sample_params_new)

        # Add the number of bounding boxes in the scene
        sample_params["length"] = sample_params["class_labels"].shape[0]

        return sample_params

    def collate_fn(self, samples):
        # Pad the batch to the local maximum number of objects
        sample_params_pad = {
            "scene_uids": [],  # str; (bs,)
            "boxes": [],       # Tensor; (bs, n, 8)
            "objs": [],        # Tensor; (bs, n)
            "edges": [],       # Tensor; (bs, n, n)
            "obj_masks": [],   # LongTensor; (bs, n)
            # "room_masks": [],  # Tensor; (bs, 1, 64, 64)
            "objfeat_vq_indices": []  # LongTensor; (bs, n, k)
        }

        # Compute the max length of the sequences in the batch
        max_length = max(sample["length"] for sample in samples)

        for sample_params in samples:
            scene_uid = str(sample_params["scene_uid"])
            objs = sample_params["objs"]
            triples = sample_params["relations"]
            boxes = np.concatenate([
                sample_params["translations"],
                sample_params["sizes"],
                sample_params["angles"]
            ], axis=-1)
            # room_mask = sample_params["room_layout"]
            if self.objfeat_type is not None:
                objfeats = sample_params[f"{self.objfeat_type}_features"]

            sample_params_pad["scene_uids"].append(scene_uid)
            # sample_params_pad["room_masks"].append(room_mask)

            sample_params_pad["objs"].append(np.pad(
                objs, (0, max_length - objs.shape[0]),
                mode="constant", constant_values=self.n_object_types
            ))  # (n,)
            sample_params_pad["boxes"].append(np.pad(
                boxes, ((0, max_length - boxes.shape[0]), (0, 0)),
                mode="constant", constant_values=0.
            ))  # (n, 8)
            if self.objfeat_type is not None:
                try:
                    sample_params_pad[f"{self.objfeat_type}_features"].append(np.pad(
                        objfeats, ((0, max_length - objfeats.shape[0]), (0, 0)),
                        mode="constant", constant_values=0.
                    ))  # (n, 512/768)
                except KeyError:
                    sample_params_pad[f"{self.objfeat_type}_features"] = [np.pad(
                        objfeats, ((0, max_length - objfeats.shape[0]), (0, 0)),
                        mode="constant", constant_values=0.
                    )]  # (n, 512/768)

            edges = self.n_predicate_types * np.ones((max_length, max_length), dtype=np.int64)  # (n, n)
            for s, p, o in triples:
                rev_p = self.predicate_types.index(
                    reverse_rel(self.predicate_types[p])
                )
                edges[s, o] = p
                edges[o, s] = rev_p
            sample_params_pad["edges"].append(edges)

            objfeat_vq_indices = sample_params["objfeat_vq_indices"]
            objfeat_vq_indices_pad = np.random.randint(0, 64, size=(max_length, objfeat_vq_indices.shape[1]))  # TODO: make `64` configurable
            objfeat_vq_indices_pad[:objfeat_vq_indices.shape[0]] = objfeat_vq_indices  # pad with random indices (not really used)
            sample_params_pad["objfeat_vq_indices"].append(objfeat_vq_indices_pad)  # (n, k)

            obj_mask = np.zeros(max_length, dtype=np.int64)  # (n,)
            obj_mask[:sample_params["length"]] = 1
            sample_params_pad["obj_masks"].append(obj_mask)

        # Make torch tensors from the numpy tensors
        for k, v in sample_params_pad.items():
            if k == "scene_uids":
                sample_params_pad[k] = v
            elif k in ["boxes", "room_masks"]:
                sample_params_pad[k] = torch.from_numpy(np.stack(v, axis=0)).float()
            else:
                sample_params_pad[k] = torch.from_numpy(np.stack(v, axis=0)).long()

        return sample_params_pad

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims


class SGDiffusion(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        max_length = self.max_length

        sample_params_new = {}
        for k, v in sample_params.items():
            if k in ["translations", "sizes", "angles"]:
                p = np.copy(v)
                # Set the attributes to for the end symbol
                L, C = p.shape
                sample_params_new[k] = np.vstack([p, np.tile(np.zeros(C)[None, :], [max_length - L, 1])]).astype(np.float32)

            elif k == "class_labels":
                class_labels = np.copy(v)
                # Delete the start label
                # Represent objectness as the last channel of class label
                new_class_labels = np.concatenate([class_labels[:, :-2], class_labels[:, -1:]], axis=-1)
                L, C = new_class_labels.shape
                # Pad the end label in the end of each sequence
                end_label = np.eye(C)[-1]
                sample_params_new["objs"] = np.vstack([
                    new_class_labels, np.tile(end_label[None, :], [max_length - L, 1])
                ]).argmax(axis=-1)  # (n,)
                # Add the number of bounding boxes in the scene
                sample_params_new["length"] = L

            elif k == "relations":
                triples = np.copy(v)
                edges = self.n_predicate_types * np.ones((max_length, max_length), dtype=np.int64)  # (n, n)
                for s, p, o in triples:
                    rev_p = self.predicate_types.index(
                        reverse_rel(self.predicate_types[p])
                    )
                    edges[s, o] = p
                    edges[o, s] = rev_p
                uppertri_edges = edges[np.triu_indices(max_length, k=1)]  # (n*(n-1)/2,)
                assert uppertri_edges.shape[0] == max_length * (max_length - 1) // 2
                sample_params_new["edges"] = uppertri_edges

        sample_params_new["scene_uid"] = sample_params["scene_uid"]
        # sample_params_new["room_mask"] = sample_params["room_layout"]

        if "descriptions" in sample_params:
            sample_params_new["descriptions"] = sample_params["descriptions"]

        # Load information file for every object
        with open(sample_params["models_info_path"], "rb") as f:
            models_info = pickle.load(f)
        objfeat_vq_indices = [
            np.array(model_info["objfeat_vq_indices"])
            for model_info in models_info
        ]
        object_descs = [
            model_info["chatgpt_caption"]
            for model_info in models_info
        ]
        # Permutation augmentation
        if "permutation" in sample_params:
            objfeat_vq_indices = [objfeat_vq_indices[i] for i in sample_params["permutation"]]
            object_descs = [object_descs[i] for i in sample_params["permutation"]]

        objfeat_vq_indices = np.vstack(objfeat_vq_indices)  # (n', k)
        objfeat_vq_indices_pad = 64 * np.ones([max_length, objfeat_vq_indices.shape[1]])  # TODO: make `64` configurable
        objfeat_vq_indices_pad[:objfeat_vq_indices.shape[0]] = objfeat_vq_indices  # pad with new empty indices
        sample_params_new["objfeat_vq_indices"] = objfeat_vq_indices_pad  # (n, k)
        objfeats_vq = np.eye(64)[objfeat_vq_indices]  # (n', k, m); TODO: make `64` configurable
        objfeats_vq_pad = np.zeros([max_length, objfeats_vq.shape[1], objfeats_vq.shape[2]])  # (n, k, m)
        objfeats_vq_pad[:objfeats_vq.shape[0]] = objfeats_vq
        sample_params_new["objfeats_vq"] = objfeats_vq_pad * 2. - 1.  # {0, 1} -> {-1, 1}; (n, k, m)
        sample_params_new["object_descs"] = object_descs  # ["a corner side table with a round top", ...]

        return sample_params_new

    def collate_fn(self, samples):
        sample_params_batch = {
            "scene_uids": [],    # str; (bs,)
            "lengths": [],       # LongTensor; (bs,)
            "objs": [],          # LongTensor; (bs, n）
            "edges": [],         # LongTensor; (bs, n*(n-1)//2)
            "boxes": [],         # Tensor; (bs, n, 8)
            "descriptions": [],  # dict; (bs,)
            # "room_masks": [],    # Tensor; (bs, 1, 64, 64)
            "objfeat_vq_indices": [],  # LongTensor; (bs, n*k)
            "objfeats_vq": [],         # Tensor; (bs, n, k*m)
            "object_descs": []         # list of strings; (bs,)
        }

        for sample_params in samples:
            scene_uid = str(sample_params["scene_uid"])
            length = sample_params["length"]
            objs = sample_params["objs"]
            edges = sample_params["edges"]
            boxes = np.concatenate([
                sample_params["translations"],
                sample_params["sizes"],
                sample_params["angles"]
            ], axis=-1)

            sample_params_batch["scene_uids"].append(scene_uid)
            sample_params_batch["lengths"].append(length)
            sample_params_batch["objs"].append(objs)
            sample_params_batch["edges"].append(edges)
            sample_params_batch["boxes"].append(boxes)

            if "descriptions" in sample_params:
                descriptions = sample_params["descriptions"]
                sample_params_batch["descriptions"].append(descriptions)

            # room_mask = sample_params["room_mask"]
            # sample_params_batch["room_masks"].append(room_mask)

            objfeat_vq_indices = sample_params["objfeat_vq_indices"]  # (n, k)
            sample_params_batch["objfeat_vq_indices"].append(objfeat_vq_indices.reshape(-1))  # (n*k,)
            objfeats_vq = sample_params["objfeats_vq"]  # (n, k, m)
            sample_params_batch["objfeats_vq"].append(objfeats_vq.reshape(objfeats_vq.shape[0], -1))  # (n, k*m)

            if "object_descs" in sample_params:
                object_descs = sample_params["object_descs"]  # ["a corner side table with a round top", ...]
                sample_params_batch["object_descs"].append(object_descs)

        # Make torch tensors from the numpy tensors
        for k, v in sample_params_batch.items():
            if k in ["scene_uids", "descriptions", "object_descs"]:
                sample_params_batch[k] = v
            elif k in ["boxes", "room_masks"]:
                sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).float()
            else:
                sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).long()

        return sample_params_batch


################################################################


## Dataset encoding API
def dataset_encoding_factory(
    name,
    dataset,
    augmentations=None,
    box_ordering=None
) -> DatasetDecoratorBase:
    # NOTE: The ordering might change after augmentations so really it should
    #       be done after the augmentations. For class frequencies it is fine
    #       though.
    if "cached" in name:
        dataset_collection = OrderedDataset(
            CachedDatasetCollection(dataset),
            ["class_labels", "translations", "sizes", "angles"],
            box_ordering=box_ordering
        )
    else:
        box_ordered_dataset = BoxOrderedDataset(
            dataset,
            box_ordering
        )
        # room_layout = RoomLayoutEncoder(box_ordered_dataset)
        class_labels = ClassLabelsEncoder(box_ordered_dataset)
        translations = TranslationEncoder(box_ordered_dataset)
        sizes = SizeEncoder(box_ordered_dataset)
        angles = AngleEncoder(box_ordered_dataset)

        dataset_collection = DatasetCollection(
            # room_layout,
            class_labels,
            translations,
            sizes,
            angles
        )

    if name == "basic":
        return DatasetCollection(
            class_labels,
            translations,
            sizes,
            angles
        )

    if isinstance(augmentations, list):
        for aug_type in augmentations:
            if aug_type == "rotation":
                print("Apply [rotation] augmentation")
                dataset_collection = Rotation(dataset_collection)
            elif aug_type == "fixed_rotation":
                print("Applying [fixed rotation] augmentation")
                dataset_collection = Rotation(dataset_collection, fixed=True)
            elif aug_type == "jitter":
                print("Apply [jittering] augmentation")
                dataset_collection = Jitter(dataset_collection)

    # Add scene graphs
    if "graph" in name or "desc" in name:
        print("Add [scene graphs] to the dataset")
        dataset_collection = Add_SceneGraph(dataset_collection)

    # Add scene descriptions
    if "desc" in name:
        if "seed" in name:
            seed = int(name.split("_")[-1])
        else:
            seed = None
        print("Add [scene descriptions] to the dataset")
        dataset_collection = Add_Description(dataset_collection, seed=seed)

    # Add object features
    objfeat_type = None
    if "objfeat" in name:
        print("Add [object features] to the dataset")
        if "openshape_vitg14" in name:
            objfeat_type = "openshape_vitg14"
        else:
            raise ValueError(f"Not found valid object feature type in [{name}]")
        dataset_collection = Add_Objfeature(dataset_collection, objfeat_type)

    # Scale the input
    print(f"Scale {list(dataset_collection.bounds.keys())}")
    if "sincos_angle" in name:
        print("Use [cos, sin] for angle encoding")
        dataset_collection = Scale_CosinAngle(dataset_collection)
    else:
        dataset_collection = Scale(dataset_collection)

    permute_keys = [
        "class_labels", "translations", "sizes", "angles",
        "relations", "descriptions", "openshape_vitg14_features"
    ]

    ################################################################

    if "sg2sc" in name:
        assert "graph" in name or "desc" in name, \
            "Add scene graphs to the dataset first (as conditions)."
        print("Use [Sg2Sc diffusion] model")
        if "no_prm" in name or "eval" in name:
            return SG2SC(dataset_collection, objfeat_type)
        else:
            print(f"Apply [permutation] augmentations on {permute_keys}")
            dataset_collection = Permutation(
                dataset_collection,
                permute_keys,
            )
            return SG2SC(dataset_collection, objfeat_type)

    elif "sgdiffusion" in name:
        assert "graph" in name or "desc" in name, \
            "Add scene graphs to the dataset first (as ground-truth)."
        print("Use [SG diffusion] model")
        if "no_prm" in name or "eval" in name:
            return SGDiffusion(dataset_collection)
        else:
            print(f"Apply [permutation] augmentations on {permute_keys}")
            dataset_collection = Permutation(
                dataset_collection,
                permute_keys,
            )
            return SGDiffusion(dataset_collection)

    else:
        raise NotImplementedError()
