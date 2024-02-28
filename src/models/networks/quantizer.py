from typing import *
from torch import Tensor, LongTensor

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GumbelQuantize(nn.Module):
    def __init__(self,
        num_hiddens: int, embedding_dim: int, n_embed: int,
        straight_through=True,
        kl_weight=5e-4, temperature=1.,
        remap: Optional[str]=None,
        unknown_index: Union[str, int]="random",
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temperature
        self.kl_weight = kl_weight

        self.proj = nn.Linear(num_hiddens, n_embed)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"\nRemapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices\n")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds: LongTensor):
        assert inds.ndim > 1

        ishape = inds.shape
        inds = inds.reshape(ishape[0], -1)  # (B, N)
        used = self.used.to(inds)  # (M,)
        match = (inds[:, :, None] == used[None, None, ...]).long()  # (B, N, M); one-hot on M

        new = match.argmax(dim=-1)
        unknown = match.sum(dim=2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds: LongTensor):
        assert inds.ndim > 1

        ishape = inds.shape
        inds = inds.reshape(ishape[0], -1)  # (B, N)
        used = self.used.to(inds)  # (M,)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds>=self.used.shape[0]] = 0  # simply set to zero

        back = torch.gather(used[None, :][[0]*inds.shape[0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z: Tensor, temperature: Optional[float]=None, kl_weight: Optional[float]=None):
        hard = self.straight_through if self.training else True  # force be true during eval as we must quantize; actually, always true seems to work
        temperature = self.temperature if temperature is None else temperature  # anneal from 1 during training
        kl_weight = self.kl_weight if kl_weight is None else kl_weight  # increase from 0 during training

        logits = self.proj(z)  # (B, N, M)
        if self.remap is not None:
            # Continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:, :, self.used]

        soft_one_hot = F.gumbel_softmax(logits, tau=temperature, dim=-1, hard=hard)
        if self.remap is not None:
            # Go back to all entries but unused set to zero
            full_zeros[:, :, self.used] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = soft_one_hot @ self.embed.weight  # (B, N, M) @ (M, D) -> (B, N, D)

        # + KL divergence to the prior loss
        qy = F.softmax(logits, dim=-1)
        loss = kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=-1).mean()

        encoding_indices = soft_one_hot.argmax(dim=-1)  # (B, N)
        if self.remap is not None:
            encoding_indices = self.remap_to_used(encoding_indices)

        return z_q, loss, encoding_indices

    def get_codebook_entry(self, indices: LongTensor):
        shape = indices.shape
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # (B, N)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # (B*N,)

        # Get quantized latent vectors
        one_hot: Tensor = F.one_hot(indices, num_classes=self.n_embed).float()   # (B*N, M)
        z_q = one_hot @ self.embed.weight  # (B*N, M) @ (M, D) -> (B*N, D)
        z_q = z_q.view(*shape, -1)  # (B, N, D)
        return z_q
