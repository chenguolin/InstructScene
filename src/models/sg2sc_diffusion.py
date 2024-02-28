from typing import *
from torch import Tensor, LongTensor

import torch
from torch import nn

from tqdm import tqdm
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from .networks import *

from src.utils.logger import StatsLogger


class Sg2ScDiffusion(nn.Module):
    def __init__(self,
        num_objs: int, num_preds: int,
        diffusion_type="ddpm",
        cfg_drop_ratio=0.2,
        use_objfeat=True
    ):
        super().__init__()

        # TODO: make these parameters configurable
        if diffusion_type == "ddpm":
            self.scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001, beta_end=0.02,
                beta_schedule="linear",
                variance_type="fixed_small",
                prediction_type="epsilon",
                clip_sample=True,
                clip_sample_range=1.
            )
        elif diffusion_type == "ddim":
            self.scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001, beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon",
                clip_sample=True,
                clip_sample_range=1.
            )
        else:
            raise NotImplementedError

        # TODO: make these parameters configurable
        self.network = Sg2ScTransformerDiffusionWrapper(
            node_dim=num_objs+1,   # +1 for empty node (not really used)
            edge_dim=num_preds+1,  # +1 for empty edge
            t_dim=128, attn_dim=512,
            global_condition_dim=None,
            context_dim=None,  # not use text condition in Sg2Sc
            n_heads=8, n_layers=5,
            gated_ff=True, dropout=0.1, ada_norm=True,
            cfg_drop_ratio=cfg_drop_ratio,
            use_objfeat=use_objfeat,
        )

        self.num_objs = num_objs
        self.num_preds = num_preds
        self.use_objfeat = use_objfeat

        self.cfg_scale = 1.  # for information logging

    def compute_losses(self, sample_params: Dict[str, Tensor], vqvae_model: nn.Module) -> Dict[str, Tensor]:
        # Unpack sample params
        x = sample_params["objs"]          # (B, N)
        e = sample_params["edges"]         # (B, N, N)
        o = sample_params["objfeat_vq_indices"]  # (B, N, K)
        mask = sample_params["obj_masks"]  # (B, N)
        boxes = sample_params["boxes"]     # (B, N, 8)

        noise = torch.randn_like(boxes)

        B, device = x.shape[0], x.device
        timesteps = torch.randint(1, self.scheduler.config.num_train_timesteps, (B,)).to(device)

        # Mask out the padding boxes
        box_mask = mask.unsqueeze(-1)  # (B, N, 1)
        noise = noise * box_mask
        boxes = boxes * box_mask

        target = noise
        noisy_boxes = self.scheduler.add_noise(boxes, noise, timesteps) * box_mask

        if self.use_objfeat:
            with torch.no_grad():
                # Decode objfeat indices to objfeat embeddings
                B, N = o.shape[:2]
                o = vqvae_model.reconstruct_from_indices(
                    o.reshape(B*N, -1)
                ).reshape(B, N, -1)
        else:
            o = None

        pred = self.network(
            noisy_boxes, x, e, o,  # `x`, `e` and `o` as conditions
            timesteps, mask=mask
        ) * box_mask

        losses = {}
        losses["pos_mse"] = F.mse_loss(pred[..., :3], target[..., :3], reduction="none").sum() / box_mask.sum()
        losses["size_mse"] = F.mse_loss(pred[..., 3:6], target[..., 3:6], reduction="none").sum() / box_mask.sum()
        losses["angle_mse"] = F.mse_loss(pred[..., 6:8], target[..., 6:8], reduction="none").sum() / box_mask.sum()

        try:
            for k, v in losses.items():
                StatsLogger.instance()[k].update(v.item() * B, B)
        except:  # `StatsLogger` is not initialized
            pass
        return losses

    @torch.no_grad()
    def generate_samples(self,
        x: LongTensor, e: LongTensor, o: Optional[LongTensor], mask: LongTensor,
        vqvae_model: nn.Module,
        num_timesteps: Optional[int]=100,
        cfg_scale=1.
    ):
        self.cfg_scale = cfg_scale
        B, N, device = x.shape[0], x.shape[1], x.device

        if self.use_objfeat:
            with torch.no_grad():
                # Decode objfeat indices to objfeat embeddings
                B, N = o.shape[:2]
                o = vqvae_model.reconstruct_from_indices(
                    o.reshape(B*N, -1)
                ).reshape(B, N, -1)
        else:
            o = None

        boxes = torch.randn(B, N, 8).to(device)

        # Mask out the padding boxes
        box_mask = mask.unsqueeze(-1)  # (B, N, 1)
        boxes = boxes * box_mask

        if num_timesteps is None:
            num_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(num_timesteps)
        for t in tqdm(self.scheduler.timesteps, desc="Generating scenes", ncols=125):
            pred = self.network(boxes, x, e, o, t, mask=mask, cfg_scale=cfg_scale) * box_mask
            boxes = self.scheduler.step(pred, t, boxes).prev_sample * box_mask

        return boxes


    @torch.no_grad()
    def complete(self,
        sample_params: Dict[str, Tensor],
        mask_object_indices: List[List[Optional[int]]],
        x: LongTensor, e: LongTensor, o: Optional[LongTensor], mask: LongTensor,
        vqvae_model: nn.Module,
        num_timesteps: Optional[int]=None,
        cfg_scale=1.
    ):
        # Unpack sample params
        boxes = sample_params["boxes"]     # (B, N, 8)

        if self.use_objfeat:
            with torch.no_grad():
                # Decode objfeat indices to objfeat embeddings
                B, N = o.shape[:2]
                o = vqvae_model.reconstruct_from_indices(
                    o.reshape(B*N, -1)
                ).reshape(B, N, -1)
        else:
            o = None

        self.cfg_scale = cfg_scale
        B, N, device = x.shape[0], x.shape[1], x.device
        device = next(self.network.parameters()).device

        # Mask the boxes to be completed
        assert len(mask_object_indices) == B
        complete_mask = torch.zeros((B, N), dtype=torch.long, device=device)
        for i in range(B):
            for obj_idx in mask_object_indices[i]:
                if obj_idx is not None:
                    complete_mask[i, obj_idx] = 1

        # Clear boxes
        boxes_start = boxes
        # All-noised boxes
        boxes = torch.randn_like(boxes)

        # Mask out the padding boxes
        box_mask = mask.unsqueeze(-1)  # (B, N, 1)
        boxes_start = boxes_start * box_mask
        boxes = boxes * box_mask

        if num_timesteps is None:
            num_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(num_timesteps)
        for t in tqdm(self.scheduler.timesteps, desc="Generating scenes", ncols=125):
            noise = torch.randn_like(boxes_start)
            noisy_boxes = self.scheduler.add_noise(boxes_start, noise, t) * box_mask  # noise the original `x` depending on `t`
            boxes = boxes * complete_mask.unsqueeze(-1) + noisy_boxes * (1 - complete_mask.unsqueeze(-1))  # complete the masked part with `noisy_x`

            pred = self.network(boxes, x, e, o, t, mask=mask, cfg_scale=cfg_scale) * box_mask
            boxes = self.scheduler.step(pred, t, boxes).prev_sample * box_mask

        return boxes


class Sg2ScTransformerDiffusionWrapper(nn.Module):
    def __init__(self,
        node_dim: int, edge_dim: int,
        attn_dim=512, t_dim=128,
        global_dim: Optional[int]=None,
        global_condition_dim: Optional[int]=None,
        context_dim: Optional[int]=None,
        n_heads=8, n_layers=5,
        gated_ff=True, dropout=0.1, ada_norm=True,
        cfg_drop_ratio=0.2,
        use_objfeat=True
    ):
        super().__init__()

        if not ada_norm:
            global_dim = t_dim  # not use AdaLN, use global information in graph-attn instead

        self.node_embed = nn.Sequential(
            nn.Embedding(node_dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, attn_dim),
        )
        if use_objfeat:
            self.node_proj_in = nn.Linear(attn_dim+1280+8, attn_dim)  # TODO: make `1280` configurable
        else:
            self.node_proj_in = nn.Linear(attn_dim+8, attn_dim)

        self.edge_embed = nn.Sequential(
            nn.Embedding(edge_dim, attn_dim//4),  # TODO: make `//4` configurable
            nn.GELU(),
            nn.Linear(attn_dim//4, attn_dim//4),
        )

        self.time_embed = nn.Sequential(
            Timestep(t_dim),
            TimestepEmbed(t_dim, t_dim)
        )

        if global_condition_dim is not None:
            self.global_condition_embed = nn.Sequential(
                nn.Linear(global_condition_dim, t_dim),
                nn.GELU(),
                nn.Linear(t_dim, t_dim)
            )

        self.transformer_blocks = nn.ModuleList([
            GraphTransformerBlock(
                attn_dim, attn_dim//4, attn_dim, global_dim,
                context_dim, t_dim,
                n_heads, gated_ff, dropout, ada_norm
            ) for _ in range(n_layers)
        ])

        self.proj_out = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, 8)
        )

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.use_global_info = global_dim is not None
        self.cfg_drop_ratio = cfg_drop_ratio

    def forward(self,
        box: Tensor, x: LongTensor, e: LongTensor, o: Optional[Tensor],
        t: LongTensor, global_condition: Optional[Tensor]=None,
        condition: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, condition_mask: Optional[LongTensor]=None,
        cfg_scale=1.
    ):
        if not torch.is_tensor(t):
            if isinstance(t, (int, float)):  # single timestep
                t = torch.tensor([t], device=x.device)
            else:  # list of timesteps
                assert len(t) == x.shape[0]
                t = torch.tensor(t, device=x.device)
        else:  # is tensor
            if t.dim() == 0:
                t = t.unsqueeze(-1).to(x.device)
        # Broadcast to batch dimension, in a way that's campatible with ONNX/Core ML
        t = t * torch.ones(x.shape[0], dtype=t.dtype, device=t.device)

        x_emb = self.node_embed(x)
        if o is not None:
            x_emb = self.node_proj_in(torch.cat([x_emb, o, box], dim=-1))
        else:
            x_emb = self.node_proj_in(torch.cat([x_emb, box], dim=-1))
        e_emb = self.edge_embed(e)
        t_emb = self.time_embed(t)
        if self.use_global_info:
            y_emb = t_emb
        else:
            y_emb =None
        if global_condition is not None:
            t_emb += self.global_condition_embed(global_condition)

        # Mask out the diagonal
        eye_mask = torch.eye(x.shape[1], device=x.device).bool().unsqueeze(0)  # (1, N, N)
        e_emb = e_emb * (~eye_mask).float().unsqueeze(-1)

        # Instance embeddings (TODO: do we need this for Sg2Sc model?)
        inst_emb = get_1d_sincos_encode(
            torch.arange(x_emb.shape[1], device=x_emb.device),
            x_emb.shape[-1], x_emb.shape[1]
        ).unsqueeze(0)  # (1, n, dx)
        x_emb = x_emb + inst_emb

        # Classifier-free guidance in training
        if self.training and self.cfg_drop_ratio > 0.:
            assert cfg_scale == 1., "Do not use `cfg_scale` during training"
            empty_e_emb = torch.zeros_like(e_emb[0]).unsqueeze(0)
            empty_prob = torch.rand(e_emb.shape[0], device=e_emb.device) < self.cfg_drop_ratio
            e_emb[empty_prob, ...] = empty_e_emb

        # Prepare for classifier-free guidance in inference
        if not self.training and cfg_scale != 1.:
            empty_e_emb = torch.zeros_like(e_emb)
            e_emb = torch.cat([empty_e_emb, e_emb], dim=0)
            x_emb = torch.cat([x_emb, x_emb], dim=0)
            y_emb = torch.cat([y_emb, y_emb], dim=0) if y_emb is not None else None
            t_emb = torch.cat([t_emb, t_emb], dim=0)
            if condition is not None:
                condition = torch.cat([condition, condition], dim=0)
            if mask is not None:
                mask = torch.cat([mask, mask], dim=0)
            if condition_mask is not None:
                condition_mask = torch.cat([condition_mask, condition_mask], dim=0)

        for block in self.transformer_blocks:
            x_emb, e_emb, y_emb = block(x_emb, e_emb, y_emb, t_emb, condition, mask, condition_mask)

        out_box = self.proj_out(x_emb)
        if mask is not None:
            out_box = out_box * mask.unsqueeze(-1)

        # Do classifier-free guidance in inference
        if not self.training and cfg_scale != 1.:
            out_box_uncond, out_box_cond = out_box.chunk(2, dim=0)
            out_box = out_box_uncond + cfg_scale * (out_box_cond - out_box_uncond)

        return out_box
