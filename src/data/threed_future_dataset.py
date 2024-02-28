# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import numpy as np
import pickle

import torch
from .utils import parse_threed_future_models


class ThreedFutureDataset(object):
    def __init__(self, objects):
        assert len(objects) > 0
        self.objects = objects

    def __len__(self):
        return len(self.objects)

    def __str__(self):
        return "Dataset contains {} objects with {} discrete types".format(
            len(self)
        )

    def __getitem__(self, idx):
        return self.objects[idx]

    def _filter_objects_by_label(self, label):
        if label is not None:
            return [oi for oi in self.objects if oi.label == label]
        else:  # return all objects if `label` is not specified
            return [oi for oi in self.objects]

    def get_closest_furniture_to_box(self, query_label, query_size):
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            mses[oi] = np.sum((oi.size - query_size)**2, axis=-1)
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x:x[1])]
        return sorted_mses[0]

    def get_closest_furniture_to_2dbox(self, query_label, query_size):
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            mses[oi] = (
                (oi.size[0] - query_size[0])**2 +
                (oi.size[2] - query_size[1])**2
            )
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x: x[1])]
        return sorted_mses[0]

    ################################ For InstructScene BEGIN ################################

    def get_closest_furniture_to_objfeat_and_size(self, query_label, query_size, query_objfeat, objfeat_type):
        # 1. Filter objects by label
        # 2. Sort objects by feature cosine similarity
        # 3. Pick top N objects (N=1 by default), i.e., select objects by feature cossim only
        # 4. Sort remaining objects by size MSE
        objects = self._filter_objects_by_label(query_label)

        cos_sims = {}
        for i, oi in enumerate(objects):
            query_objfeat = query_objfeat / np.linalg.norm(query_objfeat, axis=-1, keepdims=True)  # L2 normalize
            assert np.allclose(np.linalg.norm(eval(f"oi.{objfeat_type}_features"), axis=-1), 1.0)  # sanity check: already L2 normalized
            cos_sims[oi] = np.dot(eval(f"oi.{objfeat_type}_features"), query_objfeat)
        sorted_cos_sims = [k for k, v in sorted(cos_sims.items(), key=lambda x:x[1], reverse=True)]

        N = 1  # TODO: make it configurable
        filted_objects = sorted_cos_sims[:min(N, len(sorted_cos_sims))]
        mses = {}
        for i, oi in enumerate(filted_objects):
            mses[oi] = np.sum((oi.size - query_size)**2, axis=-1)
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x:x[1])]
        return sorted_mses[0], cos_sims[sorted_mses[0]]  # return values of cossim for debugging

    ################################ For InstructScene END ################################

    @classmethod
    def from_dataset_directory(
        cls, dataset_directory, path_to_model_info, path_to_models
    ):
        objects = parse_threed_future_models(
            dataset_directory, path_to_models, path_to_model_info
        )
        return cls(objects)

    @classmethod
    def from_pickled_dataset(cls, path_to_pickled_dataset):
        with open(path_to_pickled_dataset, "rb") as f:
            dataset = pickle.load(f)
        return dataset


################################ For InstructScene BEGIN ################################

class ThreedFutureFeatureDataset(ThreedFutureDataset):
    def __init__(self, objects, objfeat_type: str):
        super().__init__(objects)

        self.objfeat_type = objfeat_type
        self.objfeat_dim = {
            "openshape_vitg14": 1280
        }[objfeat_type]

    def __getitem__(self, idx):
        obj = self.objects[idx]
        return {
            "jid": obj.model_jid,
            "objfeat": eval(f"obj.{self.objfeat_type}_features")
        }

    def collate_fn(self, samples):
        sample_batch = {
            "jids": [],     # str; (bs,)
            "objfeats": []  # Tensor; (bs, objfeat_dim)
        }

        for sample in samples:
            sample_batch["jids"].append(str(sample["jid"]))
            sample_batch["objfeats"].append(sample["objfeat"])
        sample_batch["objfeats"] = torch.from_numpy(np.stack(sample_batch["objfeats"], axis=0)).float()

        return sample_batch

################################ For InstructScene END ################################
