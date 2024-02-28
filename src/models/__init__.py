from typing import *
from torch import Tensor
from torch.optim import Optimizer

from functools import partial

from torch import optim

from .objfeat_vqvae import ObjectFeatureVQVAE
from .sg_diffusion_vq_objfeat import SgObjfeatVQDiffusion
from .sg2sc_diffusion import Sg2ScDiffusion


def model_from_config(
    config: Dict[str, Any],
    num_objs=-1, num_preds=-1,
    text_emb_dim=512,
    **kwargs
):
    # Scene graph to scene layout
    if "sg2sc" in config["name"]:
        if "diffusion" in config["name"]:
            return Sg2ScDiffusion(
                num_objs, num_preds,
                use_objfeat="objfeat" in config["name"]
            )
        else:
            raise NotImplementedError(f"Unknown model name: {config['name']}")

    # Instruction to scene graph
    elif "sg" in config["name"]:
        if "vq_objfeat" in config["name"]:
            return SgObjfeatVQDiffusion(
                num_objs, num_preds,
                text_emb_dim=text_emb_dim
            )
        else:
            raise NotImplementedError(f"Unknown model name: {config['name']}")

    # Object feature VQ VAE
    elif "objfeatvqvae" in config["name"]:
        return ObjectFeatureVQVAE(
            config["objfeat_type"],
            config["vq_type"],
            **kwargs
        )

    else:
        raise NotImplementedError(f"Unknown model name: {config['name']}")


def optimizer_from_config(config: Dict[str, Any], params: Iterable[Tensor]) -> Optimizer:
    name = config["name"]
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.)

    kwargs = dict(
        lr=lr,
        weight_decay=weight_decay
    )

    # SGD config
    momentum = config.get("momentum", 0.)
    nesterov = config.get("nesterov", False)

    # Adam/AdamW/Radam config
    betas = config.get("betas", (0.9, 0.999))

    optimizer = {
        "sgd": partial(optim.SGD, momentum=momentum, nesterov=nesterov, **kwargs),
        "adam": partial(optim.Adam, betas=betas, **kwargs),
        "adamw": partial(optim.AdamW, betas=betas, **kwargs),
        "radam": partial(optim.RAdam, betas=betas, **kwargs),
    }[name]

    return optimizer(params)
