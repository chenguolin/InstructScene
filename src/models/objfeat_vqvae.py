from typing import *
from torch import Tensor

import torch
from torch import nn

from .networks import *

from src.utils.logger import StatsLogger


class ObjectFeatureVQVAE(nn.Module):
    def __init__(self, objfeat_type: str, vq_type="gumbel", **kwargs):
        super().__init__()

        objfeat_dim = {
            "openshape_vitg14": 1280,
        }[objfeat_type]

        # TODO: make these parameters configurable
        self.network = ObjectFeatureTransformerVQVAEWrapper(
            n_tokens=4,
            n_embeds=64, embed_dim=512,
            kv_dim=objfeat_dim,
            n_heads=8, n_layers=4,
            gated_ff=True, dropout=0.1,
            vq_type=vq_type,
            use_pe=True
        )

        self.objfeat_type = objfeat_type
        self.objfeat_min = kwargs.get("objfeat_min", None)
        self.objfeat_max = kwargs.get("objfeat_max", None)

    def compute_losses(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        # Unpack batch
        features = batch["objfeats"]

        assert self.objfeat_min is not None and self.objfeat_max is not None, \
            "Object feature min/max must be set before computing losses"
        assert torch.all(features >= self.objfeat_min) and torch.all(features <= self.objfeat_max)

        # Pre-processing
        features = (features - self.objfeat_min) / (self.objfeat_max - self.objfeat_min)  # [0, 1]
        features = features * 2. - 1.  # [-1, 1]

        rec_features, qloss = self.network(features)

        losses = {}
        losses["qloss"] = qloss
        losses["rec_mse"] = F.mse_loss(rec_features, features)

        try:
            for k, v in losses.items():
                StatsLogger.instance()[k].update(v.item() * features.shape[0], features.shape[0])
        except:  # `StatsLogger` is not initialized
            pass
        return losses

    @torch.no_grad()
    def quantize_to_indices(self, features: Tensor) -> LongTensor:
        assert self.objfeat_min is not None and self.objfeat_max is not None, \
            "Object feature min/max must be set before computing losses"
        assert torch.all(features >= self.objfeat_min) and torch.all(features <= self.objfeat_max)

        # Pre-processing
        features = (features - self.objfeat_min) / (self.objfeat_max - self.objfeat_min)  # [0, 1]
        features = features * 2. - 1.  # [-1, 1]

        return self.network.encode(features, not_quantize=False)[-1]  # (B, N)

    @torch.no_grad()
    def reconstruct_from_indices(self, indices: LongTensor) -> Tensor:
        assert self.objfeat_min is not None and self.objfeat_max is not None, \
            "Object feature min/max must be set before computing losses"

        rec_features = self.network.decode_indices(indices)  # (B, D)
        assert torch.all(rec_features >= -1.) and torch.all(rec_features <= 1.)

        # Post-processing
        rec_features = (rec_features + 1.) / 2.  # [0, 1]
        rec_features = rec_features * (self.objfeat_max - self.objfeat_min) + self.objfeat_min  # [min, max]
        return rec_features

    @torch.no_grad()
    def reconstruct(self, features: Tensor):
        quant_indices = self.quantize_to_indices(features)  # (B, N)
        rec_features = self.reconstruct_from_indices(quant_indices)  # (B, D)
        return rec_features


class ObjectFeatureTransformerVQVAEWrapper(nn.Module):
    def __init__(self,
        n_tokens: int,
        n_embeds: int, embed_dim: int,
        kv_dim=1280,
        n_heads=8, n_layers=4,
        gated_ff=True, dropout=0.1,
        vq_type="gumbel",
        use_pe=True,
        # For Gumbel VQ-VAE
        straight_through=True, kl_weight=5e-4, temperature=1.,
        # For remapping
        remap: Optional[str]=None,
        unknown_index: Union[str, int]="random",
    ):
        super().__init__()
        assert n_layers % 2 == 0, f"Number of VQ-VAE layers must be even, but got {n_layers}"

        #### Encoder ####

        self.tokens = nn.Parameter(torch.empty(n_tokens, embed_dim))
        self.tokens.data.uniform_(-1./embed_dim, 1./embed_dim)

        self.transformer_en = nn.ModuleList([
            BasicTransformerBlock(  # self-attn + cross-attn + ff
                embed_dim, embed_dim, kv_dim, None,
                n_heads, gated_ff, dropout
            ) for _ in range(n_layers//2)
        ])

        #### Vector Quantizer ####

        if vq_type == "gumbel":
            self.quantizer = GumbelQuantize(
                embed_dim, embed_dim, n_embeds,
                straight_through,
                kl_weight, temperature,
                remap, unknown_index
            )
        else:
            raise NotImplementedError

        #### Decoder ####

        self.transformer_de = nn.ModuleList([
            BasicTransformerBlock(  # self-attn + ff
                embed_dim, embed_dim, None, None,
                n_heads, gated_ff, dropout
            ) for _ in range(n_layers - n_layers//2)
        ])

        self.out_norm = nn.LayerNorm(embed_dim)
        self.out = nn.Sequential(
            nn.Linear(embed_dim, kv_dim),
            nn.GELU(),
            nn.Linear(kv_dim, kv_dim)
        )

        self.use_pe = use_pe

    def encode(self, x: Tensor, not_quantize=False):
        assert torch.all(x >= -1.) and torch.all(x <= 1.)

        h = self.tokens.unsqueeze(0).repeat(x.shape[0], 1, 1)  # (B, N, D)
        if self.use_pe:
            h = h + get_1d_sincos_encode(
                torch.arange(h.shape[1], device=h.device),
                h.shape[-1], h.shape[1]
            ).unsqueeze(0)

        for layer in self.transformer_en:
            h = layer(h, context=x.unsqueeze(1))  # self-attn + cross-attn + ff

        if not_quantize:  # not go through quantization layer
            return h, None, None

        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info

    def decode(self, h: Tensor, not_quantize=False):
        if not_quantize:  # not go through quantization layer
            quant = h
        else:
            quant, _, _ = self.quantizer(h)

        h = quant  # (B, N, D)
        if self.use_pe:
            h = h + get_1d_sincos_encode(
                torch.arange(h.shape[1], device=h.device),
                h.shape[-1], h.shape[1]
            ).unsqueeze(0)

        for layer in self.transformer_de:
            h = layer(h)  # self-attn + ff

        h = self.out_norm(h)  # (B, N, D)
        dec = torch.mean(h, dim=1)  # (B, D); average over tokens
        return self.out(dec).clamp(-1., 1.)

    def decode_indices(self, indices: LongTensor):
        quant = self.quantizer.get_codebook_entry(indices)
        return self.decode(quant, not_quantize=True)

    def forward(self, inputs: Tensor):
        quant, qloss, _ = self.encode(inputs, not_quantize=False)
        rec_input = self.decode(quant, not_quantize=True)
        return rec_input, qloss
