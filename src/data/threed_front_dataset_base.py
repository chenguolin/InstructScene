# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import numpy as np

from functools import lru_cache
from scipy.ndimage import rotate

import torch
from torch.utils.data import Dataset

import os
import pickle
from .utils_text import compute_loc_rel, reverse_rel, rotate_rel


class DatasetDecoratorBase(Dataset):
    """A base class that helps us implement decorators for ThreeDFront-like
    datasets."""
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def bounds(self):
        return self._dataset.bounds

    @property
    def n_classes(self):
        return self._dataset.n_classes

    @property
    def class_labels(self):
        return self._dataset.class_labels

    @property
    def class_frequencies(self):
        return self._dataset.class_frequencies

    @property
    def n_object_types(self):
        return self._dataset.n_object_types

    @property
    def object_types(self):
        return self._dataset.object_types

    @property
    def feature_size(self):
        return self.bbox_dims + self.n_classes

    @property
    def bbox_dims(self):
        raise NotImplementedError()

    ################################ For InstructScene BEGIN ################################

    # Get the number of predicate types in scene graphs
    @property
    def n_predicate_types(self):
        return self._dataset.n_predicate_types

    # Get the predicate types in scene graphs
    @property
    def predicate_types(self):
        return self._dataset.predicate_types

    # Get the max input length for diffusion models
    @property
    def max_length(self):
        return self._dataset.max_length 

    ################################ For InstructScene END ################################

    def post_process(self, s):
        return self._dataset.post_process(s)


class BoxOrderedDataset(DatasetDecoratorBase):
    def __init__(self, dataset, box_ordering=None):
        super().__init__(dataset)
        self.box_ordering = box_ordering

    @lru_cache(maxsize=16)
    def _get_boxes(self, scene_idx):
        scene = self._dataset[scene_idx]
        if self.box_ordering is None:
            return scene.bboxes
        elif self.box_ordering == "class_frequencies":
            return scene.ordered_bboxes_with_class_frequencies(
                self.class_frequencies
            )
        else:
            raise NotImplementedError()


class DataEncoder(BoxOrderedDataset):
    """DataEncoder is a wrapper for all datasets we have
    """
    @property
    def property_type(self):
        raise NotImplementedError()


class RoomLayoutEncoder(DataEncoder):
    @property
    def property_type(self):
        return "room_layout"

    def __getitem__(self, idx):
        """Implement the encoding for the room layout as images."""
        img = self._dataset[idx].room_mask[:, :, 0:1]
        return np.transpose(img, (2, 0, 1))

    @property
    def bbox_dims(self):
        return 0


class ClassLabelsEncoder(DataEncoder):
    """Implement the encoding for the class labels."""
    @property
    def property_type(self):
        return "class_labels"

    def __getitem__(self, idx):
        # Make a local copy of the class labels
        classes = self.class_labels

        # Get the scene
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        C = len(classes)  # number of classes
        class_labels = np.zeros((L, C), dtype=np.float32)
        for i, bs in enumerate(boxes):
            class_labels[i] = bs.one_hot_label(classes)
        return class_labels

    @property
    def bbox_dims(self):
        return 0


class TranslationEncoder(DataEncoder):
    @property
    def property_type(self):
        return "translations"

    def __getitem__(self, idx):
        # Get the scene
        scene = self._dataset[idx]
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        translations = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            translations[i] = bs.centroid(-scene.centroid)
        return translations

    @property
    def bbox_dims(self):
        return 3


class SizeEncoder(DataEncoder):
    @property
    def property_type(self):
        return "sizes"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        sizes = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            sizes[i] = bs.size
        return sizes

    @property
    def bbox_dims(self):
        return 3


class AngleEncoder(DataEncoder):
    @property
    def property_type(self):
        return "angles"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        # Get the rotation matrix for the current scene
        L = len(boxes)  # sequence length
        angles = np.zeros((L, 1), dtype=np.float32)
        for i, bs in enumerate(boxes):
            angles[i] = bs.z_angle
        return angles

    @property
    def bbox_dims(self):
        return 1


class DatasetCollection(DatasetDecoratorBase):
    def __init__(self, *datasets):
        super().__init__(datasets[0])
        self._datasets = datasets

    @property
    def bbox_dims(self):
        return sum(d.bbox_dims for d in self._datasets)

    def __getitem__(self, idx):
        sample_params = {}
        for di in self._datasets:
            sample_params[di.property_type] = di[idx]
        return sample_params

    @staticmethod
    def collate_fn(samples):
        # We assume that all samples have the same set of keys
        key_set = set(samples[0].keys()) - set(["length"])

        # Compute the max length of the sequences in the batch
        max_length = max(sample["length"] for sample in samples)

        # Assume that all inputs that are 3D or 1D do not need padding.
        # Otherwise, pad the first dimension.
        padding_keys = set(k for k in key_set if len(samples[0][k].shape) == 2)
        sample_params = {}
        sample_params.update({
            k: np.stack([sample[k] for sample in samples], axis=0)
            for k in (key_set-padding_keys)
        })

        sample_params.update({
            k: np.stack([
                np.vstack([
                    sample[k],
                    np.zeros((max_length-len(sample[k]), sample[k].shape[1]))
                ]) for sample in samples
            ], axis=0)
            for k in padding_keys
        })
        sample_params["lengths"] = np.array([
            sample["length"] for sample in samples
        ])

        # Make torch tensors from the numpy tensors
        torch_sample = {
            k: torch.from_numpy(sample_params[k]).float()
            for k in sample_params
        }

        torch_sample.update({
            k: torch_sample[k][:, None]
            for k in torch_sample.keys()
            if "_tr" in k
        })

        return torch_sample


class CachedDatasetCollection(DatasetCollection):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._dataset = dataset

    def __getitem__(self, idx):
        return self._dataset.get_room_params(idx)

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims


class Rotation(DatasetDecoratorBase):
    def __init__(self, dataset, min_rad=0.174533, max_rad=5.06145, fixed=False):
        super().__init__(dataset)
        self._min_rad = min_rad
        self._max_rad = max_rad
        self._fixed   = fixed

    @staticmethod
    def rotation_matrix_around_y(theta):
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        return R

    @property
    def rot_angle(self):
        if np.random.rand() < 0.5:
            return np.random.uniform(self._min_rad, self._max_rad)
        else:
            return 0.0

    @property
    def fixed_rot_angle(self):
        if np.random.rand() < 0.25:
            return np.pi * 1.5
        elif np.random.rand() < 0.50:
            return np.pi
        elif np.random.rand() < 0.75:
            return np.pi * 0.5
        else:
            return 0.0

    def __getitem__(self, idx):
        # Get the rotation matrix for the current scene
        if self._fixed:
            rot_angle = self.fixed_rot_angle
        else:
            rot_angle = self.rot_angle
        R = Rotation.rotation_matrix_around_y(rot_angle)

        sample_params = self._dataset[idx]
        sample_params["aug_angle"] = rot_angle  # for check in `Add_SceneGraph`
        for k, v in sample_params.items():
            if k == "translations":
                sample_params[k] = v.dot(R)

            elif k == "angles":
                angle_min, _ = self.bounds["angles"]
                sample_params[k] = \
                    (v + rot_angle - angle_min) % (2 * np.pi) + angle_min

            elif k == "room_layout":
                # Fix the ordering of the channels because it was previously
                # changed
                img = np.transpose(v, (1, 2, 0))
                sample_params[k] = np.transpose(rotate(
                    img, rot_angle * 180 / np.pi, reshape=False
                ), (2, 0, 1))

        return sample_params


class Scale(DatasetDecoratorBase):
    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k in bounds:
                sample_params[k] = Scale.scale(
                    v, bounds[k][0], bounds[k][1]
                )
        return sample_params

    def post_process(self, sample_params):
        bounds = self.bounds
        for k, v in sample_params.items():
            if k in bounds:
                print(f"Postprocess [{k}] by bounds {bounds[k][0]}~{bounds[k][1]}")
                sample_params[k] = Scale.descale(
                    v, bounds[k][0], bounds[k][1]
                )
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 1


class Scale_CosinAngle(DatasetDecoratorBase):
    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "angles":
                sample_params[k] = np.concatenate([np.cos(v), np.sin(v)], axis=-1)

            elif k in bounds:
                sample_params[k] = Scale.scale(
                    v, bounds[k][0], bounds[k][1]
                )
        return sample_params

    def post_process(self, sample_params):
        bounds = self.bounds
        for k, v in sample_params.items():
            if k == "angles":
                # theta = arctan sin/cos y/x
                print(f"Postprocess [{k}] by [arctan2]")
                sample_params[k] = np.arctan2(v[..., 1:2], v[..., 0:1])

            elif k in bounds:
                print(f"Postprocess [{k}] by bounds {bounds[k][0]}~{bounds[k][1]}")
                sample_params[k] = Scale.descale(
                    v, bounds[k][0], bounds[k][1]
                )
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 2


class Jitter(DatasetDecoratorBase):
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k in ["translations", "sizes", "angles"]:
                sample_params[k] = v + np.random.normal(0, 0.01)
        return sample_params


class Permutation(DatasetDecoratorBase):
    def __init__(self, dataset, permutation_keys, permutation_axis=0):
        super().__init__(dataset)
        self._permutation_keys = permutation_keys
        self._permutation_axis = permutation_axis

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        shapes = sample_params["class_labels"].shape
        ordering = np.random.permutation(shapes[self._permutation_axis])
        sample_params["permutation"] = ordering

        for k in self._permutation_keys:
            if k not in sample_params:
                continue

            ################################ For InstructScene BEGIN ################################

            if k == "relations":
                if sample_params[k].shape[0] > 0:
                    idx_mapping = {ordering[i]: i for i in range(len(ordering))}
                    sample_params[k][:, 0] = np.vectorize(idx_mapping.get)(sample_params[k][:, 0])
                    sample_params[k][:, 2] = np.vectorize(idx_mapping.get)(sample_params[k][:, 2])
            elif k == "descriptions":
                sample_params[k]["obj_class_ids"] = [sample_params[k]["obj_class_ids"][i] for i in ordering]
                idx_mapping = {ordering[i]: i for i in range(len(ordering))}
                for i in range(len(sample_params[k]["obj_relations"])):
                    s, p, o = sample_params[k]["obj_relations"][i]
                    s_new, o_new = idx_mapping[s], idx_mapping[o]
                    sample_params[k]["obj_relations"][i] = (s_new, p, o_new)

            ################################ For InstructScene END ################################

            else:
                sample_params[k] = sample_params[k][ordering]
        return sample_params


class OrderedDataset(DatasetDecoratorBase):
    def __init__(self, dataset, ordered_keys, box_ordering=None):
        super().__init__(dataset)
        self._ordered_keys = ordered_keys
        self._box_ordering = box_ordering

    def __getitem__(self, idx):
        if self._box_ordering is None:
            return self._dataset[idx]

        if self._box_ordering != "class_frequencies":
            raise NotImplementedError()

        sample = self._dataset[idx]
        order = self._get_class_frequency_order(sample)
        for k in self._ordered_keys:
            sample[k] = sample[k][order]
        return sample

    def _get_class_frequency_order(self, sample):
        t = sample["translations"]
        c = sample["class_labels"].argmax(-1)
        class_frequencies = self.class_frequencies
        class_labels = self.class_labels
        f = np.array([
            [class_frequencies[class_labels[ci]]]
            for ci in c
        ])

        return np.lexsort(np.hstack([t, f]).T)[::-1]


################################ For InstructScene BEGIN ################################

class Add_SceneGraph(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        if not os.path.exists(sample_params["relation_path"]):
            relations = []
            num_objs = len(sample_params["translations"])

            for idx in range(num_objs):
                c1_id = sample_params["class_labels"][idx, :].argmax()
                t1 = sample_params["translations"][idx, :]
                r1 = sample_params["angles"][idx, 0]
                s1 = sample_params["sizes"][idx, :]
                corners1 = trs_to_corners(t1, r1, s1)  # (8, 3)
                name1 = self.object_types[c1_id]

                # Full scene graph
                for other_idx in range(idx+1, num_objs):
                    c2_id = sample_params["class_labels"][other_idx, :].argmax()
                    t2 = sample_params["translations"][other_idx, :]
                    r2 = sample_params["angles"][other_idx, 0]
                    s2 = sample_params["sizes"][other_idx, :]
                    corners2 = trs_to_corners(t2, r2, s2)  # (8, 3)
                    name2 = self.object_types[c2_id]

                    # Compute location relation
                    loc_rel_str = compute_loc_rel(corners1, corners2, name1, name2)
                    if loc_rel_str is not None:
                        relation_id = self.predicate_types.index(loc_rel_str)
                        relations.append([idx, relation_id, other_idx])

                # (Optional) Add a virtual relation to root node
                # relations.append([idx, self.n_predicate_types, num_objs])

            sample_params["relations"] = np.array(relations)  # (num_triples, 3)
            # Only cache scene graphs when no augmentations
            if "aug_angle" not in sample_params or sample_params["aug_angle"] == 0.:
                np.save(sample_params["relation_path"], sample_params["relations"])

        else:
            sample_params["relations"] = np.load(sample_params["relation_path"], allow_pickle=True)

            if "aug_angle" in sample_params:
                for i in range(len(sample_params["relations"])):
                    p = sample_params["relations"][i, 1]
                    sample_params["relations"][i, 1] = self.predicate_types.index(
                        rotate_rel(self.predicate_types[p], sample_params["aug_angle"])
                    )

        # (Optional) Add virtual root node
        # onehot_root_label = np.eye(sample_params["class_labels"].shape[1])[self.n_object_types]
        # sample_params["class_labels"] = np.vstack([sample_params["class_labels"], onehot_root_label])
        # sample_params["translations"] = np.vstack([sample_params["translations"], np.zeros(3)])
        # sample_params["angles"] = np.vstack([sample_params["angles"], np.zeros(1)])
        # sample_params["sizes"] = np.vstack([sample_params["sizes"], np.ones(3)])

        return sample_params


class Add_Description(DatasetDecoratorBase):
    def __init__(self, dataset, only_describe_main_objects=True, seed=None):
        super().__init__(dataset)
        self.only_describe_main_objects = only_describe_main_objects
        self.seed = seed

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        assert "relations" in sample_params, "Relations must be computed before adding descriptions."

        if not os.path.exists(sample_params["description_path"]):
            descriptions = {
                "obj_class_ids": [],  # [4("chair"), 18("table"), ...]; 4, 18 are class ids of the objects at 0 and 1 indices
                "obj_relations": [],  # [(0, 2("in front of"), 1), ...]; 0, 1 are indices of objects; 2 is relation id
            }

            class_ids = sample_params["class_labels"].argmax(axis=1)
            descriptions["obj_class_ids"] = class_ids.tolist()

            if self.only_describe_main_objects:
                # Get the mapping from a object to its relations
                relations = {}
                for i in range(len(sample_params["relations"])):
                    s, p, o = sample_params["relations"][i]
                    if p != self.n_predicate_types:  # not a virtual relation
                        rev_p = self.predicate_types.index(
                            reverse_rel(self.predicate_types[p])
                        )
                        try:
                            relations[s].append((p, o))
                        except KeyError:
                            relations[s] = [(p, o)]
                        try:
                            relations[o].append((rev_p, s))
                        except KeyError:
                            relations[o] = [(rev_p, s)]

                # Get the main objects by the area of their ground bounding boxes
                obj_volumes = np.array(list(map(lambda x: x[0]*x[2], sample_params["sizes"])))
                obj_volumn_sorted_indices = np.argsort(obj_volumes)[::-1]
                main_obj_indices = obj_volumn_sorted_indices[:min(3, len(obj_volumn_sorted_indices))]  # top 3 main objects

                # Get the relations between the main objects and others
                relation_ids = []
                for s in main_obj_indices:
                    if relations.get(s) is not None:
                        for p, o in relations[s]:
                            relation_ids.append((
                                int(s), int(p), int(o)  # e.g., (0, 2, 1); 0, 1 are indices of objects; 2 is relation id
                            ))
                descriptions["obj_relations"] = relation_ids

            else:
                # Get the relations between objects
                relation_ids = []
                for triples in sample_params["relations"]:
                    s, p, o = triples
                    relation_ids.append((
                        int(s), int(p), int(o)  # e.g., (0, 2, 1); 0, 1 are indices of objects; 2 is relation id
                    ))
                descriptions["obj_relations"] = relation_ids

            # Only cache descriptions when no augmentations
            if "aug_angle" not in sample_params or sample_params["aug_angle"] == 0.:
                with open(sample_params["description_path"], 'wb') as f:
                    pickle.dump(descriptions, f)
            sample_params["descriptions"] = descriptions

        else:
            with open(sample_params["description_path"], 'rb') as f:
                descriptions = pickle.load(f)

            if "aug_angle" in sample_params:
                for i in range(len(descriptions["obj_relations"])):
                    s_class_id, p, o_class_id = descriptions["obj_relations"][i]
                    descriptions["obj_relations"][i] = (
                        int(s_class_id),
                        int(self.predicate_types.index(
                            rotate_rel(self.predicate_types[p], sample_params["aug_angle"])
                        )),
                        int(o_class_id)
                    )
            sample_params["descriptions"] = descriptions

        return sample_params


class Add_Objfeature(DatasetDecoratorBase):
    def __init__(self, dataset, objfeat_type="openshape_vitg14"):
        super().__init__(dataset)
        self.objfeat_type = objfeat_type

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        sample_params[f"{self.objfeat_type}_features"] = \
            np.load(sample_params[f"{self.objfeat_type}_path"])

        return sample_params


################################################################


## Helper functions
def trs_to_corners(t: np.ndarray, r: float, s: np.ndarray) -> np.ndarray:
    """Get the corners of the bounding box from the translation, rotation and size."""
    # Points in `template` are in the same order as `trimesh`,
    # which is used in `ThreedFutureModel` for loading corners
    template = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [ 1, -1, -1], [ 1, -1, 1], [ 1, 1, -1], [ 1, 1, 1]
    ])
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(r)
    R[0, 2] = -np.sin(r)
    R[2, 0] = np.sin(r)
    R[2, 2] = np.cos(r)
    R[1, 1] = 1.

    return (template * s).dot(R) + t

################################ For InstructScene END ################################
