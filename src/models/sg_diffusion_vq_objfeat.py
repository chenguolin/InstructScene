from typing import *
from torch import Tensor, LongTensor

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .networks import *

from src.utils.logger import StatsLogger


class SgObjfeatVQDiffusion(nn.Module):
    def __init__(self,
        num_objs: int,
        num_preds: int,
        num_timesteps=100,
        parameterization="x0",
        sample_method="importance",
        mask_weight=[1., 1.],
        auxiliary_loss_weight=5e-4,
        adaptive_auxiliary_loss=True,
        cfg_drop_ratio=0.2,
        text_emb_dim=512
    ):
        super().__init__()

        # TODO: make these parameters configurable
        self.network = SgObjfeatTransformerVQDiffusionWrapper(
            node_dim=num_objs+2,   # +2 for empty node and [mask] token
            edge_dim=num_preds+2,  # +2 for empty edge and [mask] token
            num_objfeat_embeds=64, objfeat_dim=512,
            attn_dim=512, t_dim=128,
            context_dim=text_emb_dim,
            global_condition_dim=text_emb_dim,
            n_heads=8, n_layers=5,
            gated_ff=True, dropout=0.1, ada_norm=True,
            use_e_cross_attn=False,
            concat_global_condition=True,
            use_positional_encoding=True
        )

        at, bt, ct, att, btt, ctt = alpha_schedule(num_timesteps, N=num_objs+1)  # +1 for empty node

        at = torch.tensor(at.astype("float64"))
        bt = torch.tensor(bt.astype("float64"))
        ct = torch.tensor(ct.astype("float64"))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype("float64"))
        btt = torch.tensor(btt.astype("float64"))
        ctt = torch.tensor(ctt.astype("float64"))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1e-5

        # Convert to float32 and register buffers.
        self.register_buffer("log_at", log_at.float())
        self.register_buffer("log_bt", log_bt.float())
        self.register_buffer("log_ct", log_ct.float())
        self.register_buffer("log_cumprod_at", log_cumprod_at.float())
        self.register_buffer("log_cumprod_bt", log_cumprod_bt.float())
        self.register_buffer("log_cumprod_ct", log_cumprod_ct.float())
        self.register_buffer("log_1_min_ct", log_1_min_ct.float())
        self.register_buffer("log_1_min_cumprod_ct", log_1_min_cumprod_ct.float())

        self.register_buffer("Lt_history", torch.zeros(num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(num_timesteps))

        self.zero_vector_x, self.zero_vector_e, self.zero_vector_o = None, None, None

        self.num_timesteps = num_timesteps
        self.parametrization = parameterization
        self.sample_method = sample_method
        self.mask_weight = mask_weight
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.cfg_drop_ratio = cfg_drop_ratio

        self.num_node_classes = num_objs+2  # +2 for empty node and [mask] token
        self.num_edge_classes = num_preds+2  # +2 for empty edge and [mask] token
        self.num_objfeat_classes = 64+2  # +2 for empty and [mask] token; TODO: make `64` configurable

        self.cfg_scale = 1.  # for information logging

    def compute_losses(self,
        sample_params: Dict[str, Tensor],
        text_last_hidden_state: Optional[Tensor]=None,
        text_embeds: Optional[Tensor]=None,
        edge_weight=10.
    ):
        # Unpack sample parameters
        x = sample_params["objs"]
        e = sample_params["edges"]
        o = sample_params["objfeat_vq_indices"]

        B, N, E, NK = x.shape[0], x.shape[-1], e.shape[-1], o.shape[-1]
        assert E == N * (N-1) // 2
        assert NK == N * 4  # TODO: make `4` configurable

        # Classifier-free guidance
        if self.training and self.cfg_drop_ratio > 0.:
            empty_last_hidden_state = torch.zeros_like(text_last_hidden_state[0]).unsqueeze(0)  # e.g., for CLIP ViT-B/32: (1, 77, 512)
            empty_prob = torch.rand(B, device=x.device) < self.cfg_drop_ratio
            text_last_hidden_state[empty_prob, ...] = empty_last_hidden_state
            if text_embeds is not None:
                empty_embeds = torch.zeros_like(text_embeds[0]).unsqueeze(0)  # e.g., for CLIP ViT-B/32: (1, 512)
                empty_prob = torch.rand(B, device=x.device) < self.cfg_drop_ratio
                text_embeds[empty_prob, ...] = empty_embeds

        _, _, _, vb_loss_x, vb_loss_e, vb_loss_o = self._train_loss(x, e, o, text_last_hidden_state, text_embeds, edge_weight)
        assert vb_loss_x.shape == vb_loss_e.shape == vb_loss_o.shape == (B,)

        losses = {}
        losses["vb_x"] = vb_loss_x.mean() / x.shape[-1]
        losses["vb_e"] = edge_weight * vb_loss_e.mean() / e.shape[-1]
        losses["vb_o"] = vb_loss_o.mean() / o.shape[-1]  # TODO: add `objfeat_weight`

        try:
            for k, v in losses.items():
                StatsLogger.instance()[k].update(v.item() * x.shape[0], x.shape[0])
        except:  # `StatsLogger` is not initialized
            pass
        return losses

    @torch.no_grad()
    def generate_samples(self,
        batch_size: int, num_nodes: int,
        text_last_hidden_state: Optional[Tensor]=None,
        text_embeds: Optional[Tensor]=None,
        filter_ratio=0.,
        cfg_scale=1.,
        truncation_rate=1.,
        skip_step=0
    ):
        self.cfg_scale = cfg_scale
        device = next(self.parameters()).device

        start_step = int(self.num_timesteps * filter_ratio)
        if start_step == 0:
            # Use full mask sample
            zero_logits_x = torch.zeros((batch_size, self.num_node_classes-1, num_nodes), device=device)
            one_logits_x = torch.ones((batch_size, 1, num_nodes),device=device)
            mask_logits_x = torch.cat([zero_logits_x, one_logits_x], dim=1)
            log_z_x = torch.log(mask_logits_x)  # (B, Cx, N)
            zero_logits_e = torch.zeros((batch_size, self.num_edge_classes-1, num_nodes*(num_nodes-1)//2), device=device)
            one_logits_e = torch.ones((batch_size, 1, num_nodes*(num_nodes-1)//2),device=device)
            mask_logits_e = torch.cat([zero_logits_e, one_logits_e], dim=1)
            log_z_e = torch.log(mask_logits_e)  # (B, Ce, N*(N-1)/2)
            zero_logits_o = torch.zeros((batch_size, self.num_objfeat_classes-1, num_nodes*4), device=device)  # TODO: make `4` configurable
            one_logits_o = torch.ones((batch_size, 1, num_nodes*4),device=device)  # TODO: make `4` configurable
            mask_logits_o = torch.cat([zero_logits_o, one_logits_o], dim=1)
            log_z_o = torch.log(mask_logits_o)  # (B, Co, N*4)

            start_step = self.num_timesteps
            if skip_step == 0:
                for diffusion_index in tqdm(range(start_step-1, -1, -1), desc="Generating graphs", ncols=125):
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                    log_z_x, log_z_e, log_z_o = self.p_sample(
                        log_z_x, log_z_e, log_z_o, t,
                        text_last_hidden_state, text_embeds,
                        cfg_scale, truncation_rate
                    )
            # Fast sampling by skipping some steps
            else:
                diffusion_list = list(range(start_step-1, -1, -1-skip_step))
                if diffusion_list[-1] != 0:
                    diffusion_list.append(0)
                for diffusion_index in tqdm(diffusion_list, desc="Generating graphs", ncols=125):
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                    log_x_recon, log_e_recon, log_o_recon = self.predict_start(
                        log_z_x, log_z_e, log_z_o, t,
                        text_last_hidden_state, text_embeds,
                        cfg_scale, truncation_rate
                    )
                    log_z_x = self.q_posterior(
                        log_x_start=log_x_recon, log_x_t=log_z_x,
                        t=t-skip_step if diffusion_index > skip_step else t,  # skip some steps
                        num_classes=self.num_node_classes
                    )
                    log_z_e = self.q_posterior(
                        log_x_start=log_e_recon, log_x_t=log_z_e,
                        t=t-skip_step if diffusion_index > skip_step else t,  # skip some steps
                        num_classes=self.num_edge_classes
                    )
                    log_z_o = self.q_posterior(
                        log_x_start=log_o_recon, log_x_t=log_z_o,
                        t=t-skip_step if diffusion_index > skip_step else t,  # skip some steps
                        num_classes=self.num_objfeat_classes
                    )
                    log_z_x = self.log_sample_categorical(log_z_x, self.num_node_classes)
                    log_z_e = self.log_sample_categorical(log_z_e, self.num_edge_classes)
                    log_z_o = self.log_sample_categorical(log_z_o, self.num_objfeat_classes)

        else:
            raise NotImplementedError

        content_token_x = log_onehot_to_index(log_z_x)
        content_token_e = log_onehot_to_index(log_z_e)
        content_token_o = log_onehot_to_index(log_z_o).reshape(batch_size, num_nodes, -1)

        onehot_x = F.one_hot(content_token_x, self.num_node_classes-1).float()  # -1 for [mask] token
        onehot_e = F.one_hot(content_token_e, self.num_edge_classes-1).float()  # -1 for [mask] token

        return onehot_x, onehot_e, content_token_o


    @torch.no_grad()
    def complete(self,  # Condition on partial graphs, generate the rest
        sample_params: Dict[str, Tensor],
        mask_object_indices: List[List[Optional[int]]],
        text_last_hidden_state: Optional[Tensor]=None,
        text_embeds: Optional[Tensor]=None,
        filter_ratio=0.5,
        cfg_scale=1.,
        truncation_rate=1.
    ):
        # Unpack sample parameters
        x = sample_params["objs"]
        e = sample_params["edges"]
        o = sample_params["objfeat_vq_indices"]

        self.cfg_scale = cfg_scale
        batch_size, num_nodes = x.shape[0], x.shape[1]
        device = next(self.parameters()).device

        # Mask the scene to be completed
        assert len(mask_object_indices) == batch_size
        x_mask = torch.zeros_like(x).long()  # (B, N)
        e_mask = torch.zeros((batch_size, num_nodes, num_nodes), device=device).long()
        o_mask = torch.zeros((batch_size, num_nodes, 4), device=device).long()  # TODO: make `4` configurable
        for i in range(batch_size):
            for obj_idx in mask_object_indices[i]:
                if obj_idx is not None:
                    x_mask[i, obj_idx] = 1
                    e_mask[i, obj_idx, :] = 1
                    e_mask[i, :, obj_idx] = 1
                    o_mask[i, obj_idx, :] = 1
        uppertri_ind = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
        e_mask = e_mask[:, uppertri_ind[0], uppertri_ind[1]]  # (B, N*(N-1)/2)
        o_mask = o_mask.reshape(batch_size, -1)  # (B, N*4)

        # Clear graphs
        log_z_x_start = index_to_log_onehot(x, self.num_node_classes)  # (B, Cx, N)
        log_z_e_start = index_to_log_onehot(e, self.num_edge_classes)  # (B, Ce, N*(N-1)/2)
        log_z_o_start = index_to_log_onehot(o, self.num_objfeat_classes)  # (B, Co, NK)
        # All-masked graphs
        zero_logits_x = torch.zeros((batch_size, self.num_node_classes-1, num_nodes), device=device)
        one_logits_x = torch.ones((batch_size, 1, num_nodes),device=device)
        mask_logits_x = torch.cat([zero_logits_x, one_logits_x], dim=1)
        log_z_x_mask = torch.log(mask_logits_x)  # (B, Cx, N)
        zero_logits_e = torch.zeros((batch_size, self.num_edge_classes-1, num_nodes*(num_nodes-1)//2), device=device)
        one_logits_e = torch.ones((batch_size, 1, num_nodes*(num_nodes-1)//2),device=device)
        mask_logits_e = torch.cat([zero_logits_e, one_logits_e], dim=1)
        log_z_e_mask = torch.log(mask_logits_e)  # (B, Ce, N*(N-1)/2)
        zero_logits_o = torch.zeros((batch_size, self.num_objfeat_classes-1, num_nodes*4), device=device)  # TODO: make `4` configurable
        one_logits_o = torch.ones((batch_size, 1, num_nodes*4),device=device)  # TODO: make `4` configurable
        mask_logits_o = torch.cat([zero_logits_o, one_logits_o], dim=1)
        log_z_o_mask = torch.log(mask_logits_o)  # (B, Co, N*4)

        start_step = int(self.num_timesteps * filter_ratio)
        if start_step == 0:
            start_step = self.num_timesteps
            for diffusion_index in tqdm(range(start_step-1, -1, -1), desc="Completing graphs", ncols=125):
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                log_z_x_t = self.q_sample(log_z_x_start, t, self.num_node_classes)  # mask the original `x` depending on `t`
                log_z_x = torch.where(x_mask.unsqueeze(1).bool(), log_z_x_mask, log_z_x_t)
                log_z_e_t = self.q_sample(log_z_e_start, t, self.num_edge_classes)  # mask the original `e` depending on `t`
                log_z_e = torch.where(e_mask.unsqueeze(1).bool(), log_z_e_mask, log_z_e_t)
                log_z_o_t = self.q_sample(log_z_o_start, t, self.num_objfeat_classes)  # mask the original `o` depending on `t`
                log_z_o = torch.where(o_mask.unsqueeze(1).bool(), log_z_o_mask, log_z_o_t)
                log_z_x, log_z_e, log_z_o = self.p_sample(
                    log_z_x, log_z_e, log_z_o, t,
                    text_last_hidden_state, text_embeds,
                    cfg_scale, truncation_rate
                )
        else:
            log_z_x = log_z_x_mask
            log_z_e = log_z_e_mask
            log_z_o = log_z_o_mask
            for diffusion_index in tqdm(range(start_step-1, -1, -1), desc="Completing graphs", ncols=125):
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                log_z_x = torch.where(x_mask.unsqueeze(1).bool(), log_z_x, log_z_x_start)  # keep the original `x`
                log_z_e = torch.where(e_mask.unsqueeze(1).bool(), log_z_e, log_z_e_start)  # keep the original `e`
                log_z_o = torch.where(o_mask.unsqueeze(1).bool(), log_z_o, log_z_o_start)  # keep the original `o`
                log_z_x, log_z_e, log_z_o = self.p_sample(
                    log_z_x, log_z_e, log_z_o, t,
                    text_last_hidden_state, text_embeds,
                    cfg_scale, truncation_rate
                )

        log_z_x = torch.where(x_mask.unsqueeze(1).bool(), log_z_x, log_z_x_start)  # keep the original `x`
        log_z_e = torch.where(e_mask.unsqueeze(1).bool(), log_z_e, log_z_e_start)  # keep the original `e`
        log_z_o = torch.where(o_mask.unsqueeze(1).bool(), log_z_o, log_z_o_start)  # keep the original `o`

        content_token_x = log_onehot_to_index(log_z_x)
        content_token_e = log_onehot_to_index(log_z_e)
        content_token_o = log_onehot_to_index(log_z_o).reshape(batch_size, num_nodes, -1)

        onehot_x = F.one_hot(content_token_x, self.num_node_classes-1).float()  # -1 for [mask] token
        onehot_e = F.one_hot(content_token_e, self.num_edge_classes-1).float()  # -1 for [mask] token

        return onehot_x, onehot_e, content_token_o


    @torch.no_grad()
    def rearrange(self,  # Condition on `x` and `o`, generate `e`
        sample_params: Dict[str, Tensor],
        text_last_hidden_state: Optional[Tensor]=None,
        text_embeds: Optional[Tensor]=None,
        filter_ratio=0.5,
        cfg_scale=1.,
        truncation_rate=1.
    ):
        # Unpack sample parameters
        x = sample_params["objs"]
        o = sample_params["objfeat_vq_indices"]

        self.cfg_scale = cfg_scale
        batch_size, num_nodes = x.shape[0], x.shape[1]
        device = next(self.parameters()).device

        log_z_x_start = index_to_log_onehot(x, self.num_node_classes)  # (B, Cx, N)
        log_z_o_start = index_to_log_onehot(o, self.num_objfeat_classes)  # (B, Co, NK)

        zero_logits_e = torch.zeros((batch_size, self.num_edge_classes-1, num_nodes*(num_nodes-1)//2), device=device)
        one_logits_e = torch.ones((batch_size, 1, num_nodes*(num_nodes-1)//2),device=device)
        mask_logits_e = torch.cat([zero_logits_e, one_logits_e], dim=1)
        log_z_e = torch.log(mask_logits_e)  # (B, Ce, N*(N-1)/2)

        start_step = int(self.num_timesteps * filter_ratio)
        if start_step == 0:
            start_step = self.num_timesteps
            for diffusion_index in tqdm(range(start_step-1, -1, -1), desc="Rearranging graphs", ncols=125):
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                log_z_x = self.q_sample(log_z_x_start, t, self.num_node_classes)  # mask the original `x` depending on `t`
                log_z_o = self.q_sample(log_z_o_start, t, self.num_objfeat_classes)  # mask the original `o` depending on `t`
                log_z_x, log_z_e, log_z_o = self.p_sample(
                    log_z_x, log_z_e, log_z_o, t,
                    text_last_hidden_state, text_embeds,
                    cfg_scale, truncation_rate
                )
        else:
            for diffusion_index in tqdm(range(start_step-1, -1, -1), desc="Rearranging graphs", ncols=125):
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                log_z_x, log_z_o = log_z_x_start, log_z_o_start  # keep the original `x` and `o`
                log_z_x, log_z_e, log_z_o = self.p_sample(
                    log_z_x, log_z_e, log_z_o, t,
                    text_last_hidden_state, text_embeds,
                    cfg_scale, truncation_rate
                )

        log_z_x, log_z_o = log_z_x_start, log_z_o_start

        content_token_x = log_onehot_to_index(log_z_x)
        content_token_e = log_onehot_to_index(log_z_e)
        content_token_o = log_onehot_to_index(log_z_o).reshape(batch_size, num_nodes, -1)

        onehot_x = F.one_hot(content_token_x, self.num_node_classes-1).float()  # -1 for [mask] token
        onehot_e = F.one_hot(content_token_e, self.num_edge_classes-1).float()  # -1 for [mask] token

        return onehot_x, onehot_e, content_token_o


    @torch.no_grad()
    def stylize(self,  # Condition on `x` and `e`, generate `o`
        sample_params: Dict[str, Tensor],
        text_last_hidden_state: Optional[Tensor]=None,
        text_embeds: Optional[Tensor]=None,
        filter_ratio=0.5,
        cfg_scale=1.,
        truncation_rate=1.
    ):
        # Unpack sample parameters
        x = sample_params["objs"]
        e = sample_params["edges"]

        self.cfg_scale = cfg_scale
        batch_size, num_nodes = x.shape[0], x.shape[1]
        device = next(self.parameters()).device

        log_z_x_start = index_to_log_onehot(x, self.num_node_classes)  # (B, Cx, N)
        log_z_e_start = index_to_log_onehot(e, self.num_edge_classes)  # (B, Ce, N*(N-1)/2)

        zero_logits_o = torch.zeros((batch_size, self.num_objfeat_classes-1, num_nodes*4), device=device)  # TODO: make `4` configurable
        one_logits_o = torch.ones((batch_size, 1, num_nodes*4),device=device)  # TODO: make `4` configurable
        mask_logits_o = torch.cat([zero_logits_o, one_logits_o], dim=1)
        log_z_o = torch.log(mask_logits_o)  # (B, Co, N*4)

        start_step = int(self.num_timesteps * filter_ratio)
        if start_step == 0:
            start_step = self.num_timesteps
            for diffusion_index in tqdm(range(start_step-1, -1, -1), desc="Stylizing graphs", ncols=125):
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                log_z_x = self.q_sample(log_z_x_start, t, self.num_node_classes)  # mask the original `x` depending on `t`
                log_z_e = self.q_sample(log_z_e_start, t, self.num_edge_classes)  # mask the original `e` depending on `t`
                log_z_x, log_z_e, log_z_o = self.p_sample(
                    log_z_x, log_z_e, log_z_o, t,
                    text_last_hidden_state, text_embeds,
                    cfg_scale, truncation_rate
                )
        else:
            for diffusion_index in tqdm(range(start_step-1, -1, -1), desc="Stylizing graphs", ncols=125):
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                log_z_x, log_z_e = log_z_x_start, log_z_e_start  # keep the original `x` and `e`
                log_z_x, log_z_e, log_z_o = self.p_sample(
                    log_z_x, log_z_e, log_z_o, t,
                    text_last_hidden_state, text_embeds,
                    cfg_scale, truncation_rate
                )

        log_z_x, log_z_e = log_z_x_start, log_z_e_start

        content_token_x = log_onehot_to_index(log_z_x)
        content_token_e = log_onehot_to_index(log_z_e)
        content_token_o = log_onehot_to_index(log_z_o).reshape(batch_size, num_nodes, -1)

        onehot_x = F.one_hot(content_token_x, self.num_node_classes-1).float()  # -1 for [mask] token
        onehot_e = F.one_hot(content_token_e, self.num_edge_classes-1).float()  # -1 for [mask] token

        return onehot_x, onehot_e, content_token_o


    ################################################################


    def multinomial_kl(self, log_prob1: Tensor, log_prob2: Tensor):  # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t: Tensor, t: LongTensor, num_classes: int):  # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)  # at
        log_bt = extract(
            (self.log_bt.exp() * (self.num_node_classes-1) / (num_classes-1)).log(),
            t, log_x_t.shape
        )  # bt; -1 for [mask] token
        log_ct = extract(self.log_ct, t, log_x_t.shape)  # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)  # 1-ct

        _prob_sum = log_at.exp() + log_bt.exp() * (num_classes-1) + log_ct.exp()
        assert torch.allclose(_prob_sum, torch.ones_like(_prob_sum))

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :-1, ...] + log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, ...] + log_1_min_ct, log_ct)
            ],
            dim=1
        )

        return log_probs

    def q_pred(self, log_x_start: Tensor, t: LongTensor, num_classes: int):  # q(xt|x0)
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)  # at~
        log_cumprod_bt = extract(
            (self.log_cumprod_bt.exp() * (self.num_node_classes-1) / (num_classes-1)).log(),
            t, log_x_start.shape
        )  # bt~; -1 for [mask] token
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)  # 1-ct~

        _prob_sum = log_cumprod_at.exp() + log_cumprod_bt.exp() * (num_classes-1) + log_cumprod_ct.exp()
        assert torch.allclose(_prob_sum, torch.ones_like(_prob_sum))

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :-1, ...] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(log_x_start[:, -1:, ...] + log_1_min_cumprod_ct, log_cumprod_ct)
            ],
            dim=1
        )

        return log_probs

    def predict_start(self,
        log_x_t: Tensor, log_e_t: Tensor, log_o_t: Tensor, t: LongTensor,
        cond_emb: Tensor, global_cond_emb: Optional[Tensor]=None,
        cfg_scale=1., truncation_rate=1.
    ):  # p(x0|xt)
        x_t: LongTensor = log_onehot_to_index(log_x_t)
        e_t: LongTensor = log_onehot_to_index(log_e_t)
        o_t: LongTensor = log_onehot_to_index(log_o_t)

        # Prepare for classifier-free guidance
        if cfg_scale != 1.:
            empty_cond_emb = torch.zeros_like(cond_emb)  # e.g., for CLIP ViT-B/32: (B, 77, 512)
            cond_emb = torch.cat([empty_cond_emb, cond_emb], dim=0)
            x_t = torch.cat([x_t, x_t], dim=0)
            e_t = torch.cat([e_t, e_t], dim=0)
            o_t = torch.cat([o_t, o_t], dim=0)
            t = torch.cat([t, t], dim=0)
            if global_cond_emb is not None:
                empty_global_cond_emb = torch.zeros_like(global_cond_emb)  # e.g., for CLIP ViT-B/32: (B, 512)
                global_cond_emb = torch.cat([empty_global_cond_emb, global_cond_emb], dim=0)

        out_x, out_e, out_o = self.network(x_t, e_t, o_t, t, cond_emb, global_cond_emb)
        out_x: Tensor; out_e: Tensor; out_o: Tensor

        assert out_x.shape[0]  == x_t.shape[0]
        assert out_x.shape[1]  == self.num_node_classes-1  # -1 for [mask] token
        assert out_x.shape[2:] == x_t.shape[1:]
        assert out_e.shape[0]  == e_t.shape[0]
        assert out_e.shape[1]  == self.num_edge_classes-1  # -1 for [mask] token
        assert out_e.shape[2:] == e_t.shape[1:]
        assert out_o.shape[0] == o_t.shape[0]
        assert out_o.shape[1] == self.num_objfeat_classes-1  # -1 for [mask] token
        assert out_o.shape[2:] == o_t.shape[1:]

        log_pred_x = F.log_softmax(out_x.double(), dim=1).float()  # (B, Cx, N)
        log_pred_e = F.log_softmax(out_e.double(), dim=1).float()  # (B, Ce, N*(N-1)/2)
        log_pred_o = F.log_softmax(out_o.double(), dim=1).float()  # (B, Co, NK)
        batch_size = log_x_t.shape[0]

        if cfg_scale != 1.:
            log_pred_x_uncond, log_pred_x_cond = log_pred_x.chunk(2, dim=0)
            log_pred_x = log_pred_x_uncond + cfg_scale * (log_pred_x_cond - log_pred_x_uncond)
            log_pred_x -= torch.logsumexp(log_pred_x, dim=1, keepdim=True)  # normalize
            log_pred_e_uncond, log_pred_e_cond = log_pred_e.chunk(2, dim=0)
            log_pred_e = log_pred_e_uncond + cfg_scale * (log_pred_e_cond - log_pred_e_uncond)
            log_pred_e -= torch.logsumexp(log_pred_e, dim=1, keepdim=True)  # normalize
            log_pred_o_uncond, log_pred_o_cond = log_pred_o.chunk(2, dim=0)
            log_pred_o = log_pred_o_uncond + cfg_scale * (log_pred_o_cond - log_pred_o_uncond)
            log_pred_o -= torch.logsumexp(log_pred_o, dim=1, keepdim=True)  # normalize
            t, cond_emb = t[batch_size:, ...], cond_emb[batch_size:, ...]
            if global_cond_emb is not None:
                global_cond_emb = global_cond_emb[batch_size:, ...]

        log_pred_x = self.truncate(log_pred_x, truncation_rate)
        log_pred_e = self.truncate(log_pred_e, truncation_rate)
        log_pred_o = self.truncate(log_pred_o, truncation_rate)

        if self.zero_vector_x is None or self.zero_vector_x.shape[0] != batch_size:
            self.zero_vector_x = torch.zeros(batch_size, 1, *out_x.shape[2:]).type_as(log_x_t) - 70.
        if self.zero_vector_e is None or self.zero_vector_e.shape[0] != batch_size:
            self.zero_vector_e = torch.zeros(batch_size, 1, *out_e.shape[2:]).type_as(log_e_t) - 70.
        if self.zero_vector_o is None or self.zero_vector_o.shape[0] != batch_size:
            self.zero_vector_o = torch.zeros(batch_size, 1, *out_o.shape[2:]).type_as(log_o_t) - 70.
        log_pred_x = torch.cat([log_pred_x, self.zero_vector_x], dim=1)  # (B, Cx+1, N)
        log_pred_e = torch.cat([log_pred_e, self.zero_vector_e], dim=1)  # (B, Ce+1, N*(N-1)/2)
        log_pred_o = torch.cat([log_pred_o, self.zero_vector_o], dim=1)  # (B, Co+1, NK)
        log_pred_x = torch.clamp(log_pred_x, -70., 0.)
        log_pred_e = torch.clamp(log_pred_e, -70., 0.)
        log_pred_o = torch.clamp(log_pred_o, -70., 0.)

        return log_pred_x, log_pred_e, log_pred_o

    def q_posterior(self,
        log_x_start: Tensor, log_x_t: Tensor, t: LongTensor,
        num_classes: int
    ):  # p_theta(xt_1|xt) = sum( q(xt-1|xt,x0') * p(x0') )
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps

        batch_size = log_x_start.shape[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == num_classes-1).unsqueeze(1)  # -1 for [mask] token
        log_one_vector = torch.zeros(batch_size, 1, *([1] * (len(log_x_start.shape)-2))).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector+1e-30).expand(-1, -1, *log_x_start.shape[2:])

        log_qt = self.q_pred(log_x_t, t, num_classes)  # q(xt|x0)
        log_qt = log_qt[:, :-1, ...]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, num_classes-1, *([-1]*(len(log_x_start.shape)-2)))
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t, num_classes)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat([log_qt_one_timestep[:, :-1, ...], log_zero_vector], dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)  # ct
        ct_vector = log_ct.expand(-1, num_classes-1, *([-1]*(len(log_x_start.shape)-2)))
        ct_vector = torch.cat([ct_vector, log_one_vector], dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector

        q = log_x_start[:, :-1, ...] - log_qt
        q = torch.cat([q, log_zero_vector], dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1, num_classes) + log_qt_one_timestep + q_log_sum_exp

        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70., 0.)

    def p_pred(self,
        log_x: Tensor, log_e: Tensor, log_o: Tensor, t: LongTensor,
        cond_emb: Tensor, global_cond_emb: Optional[Tensor],
        cfg_scale=1., truncation_rate=1.
    ):  # if x0, first p(x0|xt), than sum( q(xt-1|xt,x0) * p(x0|xt) )
        if self.parametrization == "x0":
            log_x_recon, log_e_recon, log_o_recon = self.predict_start(
                log_x, log_e, log_o, t,
                cond_emb, global_cond_emb,
                cfg_scale, truncation_rate
            )
            log_model_pred_x = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t,
                num_classes=self.num_node_classes
            )
            log_model_pred_e = self.q_posterior(
                log_x_start=log_e_recon, log_x_t=log_e, t=t,
                num_classes=self.num_edge_classes
            )
            log_model_pred_o = self.q_posterior(
                log_x_start=log_o_recon, log_x_t=log_o, t=t,
                num_classes=self.num_objfeat_classes
            )
        elif self.parametrization == "direct":
            log_x_recon, log_e_recon, log_o_recon = None, None, None
            log_model_pred_x, log_model_pred_e, log_model_pred_o = self.predict_start(
                log_x, log_e, log_o, t,
                cond_emb, global_cond_emb,
                cfg_scale, truncation_rate
            )
        else:
            raise ValueError

        return log_model_pred_x, log_model_pred_e, log_model_pred_o, log_x_recon, log_e_recon, log_o_recon

    @torch.no_grad()
    def p_sample(self,
        log_x: Tensor, log_e: Tensor, log_o: Tensor, t: LongTensor,
        cond_emb: Tensor, global_cond: Optional[Tensor]=None,
        cfg_scale=1., truncation_rate=1.
    ):  # sample q(xt-1) for next step from xt, actually is p(xt-1|xt)
        log_model_pred_x, log_model_pred_e, log_model_pred_o, _, _, _ = self.p_pred(
            log_x, log_e, log_o, t,
            cond_emb, global_cond,
            cfg_scale, truncation_rate
        )

        # Gumbel sample
        out_x = self.log_sample_categorical(log_model_pred_x, self.num_node_classes)
        out_e = self.log_sample_categorical(log_model_pred_e, self.num_edge_classes)
        out_o = self.log_sample_categorical(log_model_pred_o, self.num_objfeat_classes)
        return out_x, out_e, out_o

    def log_sample_categorical(self, logits: Tensor, num_classes: int):  # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)

        log_sample = index_to_log_onehot(sample, num_classes)
        return log_sample

    def q_sample(self, log_x_start: Tensor, t: LongTensor, num_classes: int):  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t, num_classes)

        # Gumbel sample
        log_sample = self.log_sample_categorical(log_EV_qxt_x0, num_classes)
        return log_sample

    def sample_time(self, b: int, device: torch.device, method="uniform"):
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method="uniform")

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # overwrite L0 (i.e., the decoder nll) term with L1
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)
            return t, pt
        elif method == "uniform":
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self,
        x: Tensor, e: Tensor, o: Tensor,
        cond_emb: Tensor, global_cond_emb: Optional[Tensor]=None,
        edge_weight=1.
    ):  # get the KL loss
        b, device = x.shape[0], x.device

        x_start, e_start, o_start = x, e, o
        t, pt = self.sample_time(b, device, self.sample_method)  # (B,)

        log_x_start = index_to_log_onehot(x_start, self.num_node_classes)  # (B, Cx, N)
        log_e_start = index_to_log_onehot(e_start, self.num_edge_classes)  # (B, Ce, N*(N-1)/2)
        log_o_start = index_to_log_onehot(o_start, self.num_objfeat_classes)  # (B, Co, NK)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t, num_classes=self.num_node_classes)
        log_et = self.q_sample(log_x_start=log_e_start, t=t, num_classes=self.num_edge_classes)
        log_ot = self.q_sample(log_x_start=log_o_start, t=t, num_classes=self.num_objfeat_classes)
        xt = log_onehot_to_index(log_xt)
        et = log_onehot_to_index(log_et)
        ot = log_onehot_to_index(log_ot)

        # Go to p_theta function
        log_x0_recon, log_e0_recon, log_o0_recon = self.predict_start(log_xt, log_et, log_ot, t, cond_emb, global_cond_emb)  # P_theta(x0|xt)
        log_model_prob_x = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t, num_classes=self.num_node_classes)  # go through q(xt_1|xt,x0)
        log_model_prob_e = self.q_posterior(log_x_start=log_e0_recon, log_x_t=log_et, t=t, num_classes=self.num_edge_classes)  # go through q(xt_1|xt,x0)
        log_model_prob_o = self.q_posterior(log_x_start=log_o0_recon, log_x_t=log_ot, t=t, num_classes=self.num_objfeat_classes)  # go through q(xt_1|xt,x0)

        # Compute log_true_prob now 
        log_true_prob_x = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t, num_classes=self.num_node_classes)
        log_true_prob_e = self.q_posterior(log_x_start=log_e_start, log_x_t=log_et, t=t, num_classes=self.num_edge_classes)
        log_true_prob_o = self.q_posterior(log_x_start=log_o_start, log_x_t=log_ot, t=t, num_classes=self.num_objfeat_classes)

        # Compute loss
        kl_x = self.multinomial_kl(log_true_prob_x, log_model_prob_x)  # (B, N)
        mask_region_x = (xt == self.num_node_classes-1).float()
        mask_weight_x = mask_region_x * self.mask_weight[0] + (1. - mask_region_x) * self.mask_weight[1]
        kl_x = kl_x * mask_weight_x    # (B, N)
        kl_x = avg_except_batch(kl_x)  # (B,)
        kl_e = self.multinomial_kl(log_true_prob_e, log_model_prob_e)  # (B, N*(N-1)/2)
        mask_region_e = (et == self.num_edge_classes-1).float()
        mask_weight_e = mask_region_e * self.mask_weight[0] + (1. - mask_region_e) * self.mask_weight[1]
        kl_e = kl_e * mask_weight_e    # (B, N*(N-1)/2)
        kl_e = avg_except_batch(kl_e)  # (B,)
        kl_o = self.multinomial_kl(log_true_prob_o, log_model_prob_o)  # (B, NK)
        mask_region_o = (ot == self.num_objfeat_classes-1).float()
        mask_weight_o = mask_region_o * self.mask_weight[0] + (1. - mask_region_o) * self.mask_weight[1]
        kl_o = kl_o * mask_weight_o    # (B, NK)
        kl_o = avg_except_batch(kl_o)  # (B,)

        decoder_nll_x = -log_categorical(log_x_start, log_model_prob_x)  # (B, N)
        decoder_nll_x = avg_except_batch(decoder_nll_x)  # (B,)
        decoder_nll_e = -log_categorical(log_e_start, log_model_prob_e)  # (B, N*(N-1)/2)
        decoder_nll_e = avg_except_batch(decoder_nll_e)  # (B,)
        decoder_nll_o = -log_categorical(log_o_start, log_model_prob_o)  # (B, NK)
        decoder_nll_o = avg_except_batch(decoder_nll_o)  # (B,)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss_x = mask * decoder_nll_x + (1. - mask) * kl_x  # (B,)
        kl_loss_e = mask * decoder_nll_e + (1. - mask) * kl_e  # (B,)
        kl_loss_o = mask * decoder_nll_o + (1. - mask) * kl_o  # (B,)

        # Record for importance sampling
        Lt2 = (kl_loss_x + edge_weight * kl_loss_e + kl_loss_o).pow(2)  # (B,);   # TODO: add `objfeat_weight`
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        loss1_x, loss1_e, loss1_o = kl_loss_x / pt, kl_loss_e / pt, kl_loss_o / pt
        vb_loss_x, vb_loss_e, vb_loss_o = loss1_x, loss1_e, loss1_o
        if self.auxiliary_loss_weight != 0.:
            kl_aux_x = self.multinomial_kl(log_x_start[:, :-1, ...], log_x0_recon[:, :-1, ...])
            kl_aux_x = kl_aux_x * mask_weight_x    # (B, N)
            kl_aux_x = avg_except_batch(kl_aux_x)  # (B,)
            kl_aux_loss_x = mask * decoder_nll_x + (1. - mask) * kl_aux_x
            kl_aux_e = self.multinomial_kl(log_e_start[:, :-1, ...], log_e0_recon[:, :-1, ...])
            kl_aux_e = kl_aux_e * mask_weight_e    # (B, N*(N-1)/2)
            kl_aux_e = avg_except_batch(kl_aux_e)  # (B,)
            kl_aux_loss_e = mask * decoder_nll_e + (1. - mask) * kl_aux_e
            kl_aux_o = self.multinomial_kl(log_o_start[:, :-1, ...], log_o0_recon[:, :-1, ...])
            kl_aux_o = kl_aux_o * mask_weight_o    # (B, NK)
            kl_aux_o = avg_except_batch(kl_aux_o)  # (B,)
            kl_aux_loss_o = mask * decoder_nll_o + (1. - mask) * kl_aux_o

            if self.adaptive_auxiliary_loss:
                addition_loss_weight = (1. - t/self.num_timesteps) + 1.
            else:
                addition_loss_weight = 1.

            loss2_x = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss_x / pt
            vb_loss_x += loss2_x  # (B,)
            loss2_e = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss_e / pt
            vb_loss_e += loss2_e  # (B,)
            loss2_o = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss_o / pt
            vb_loss_o += loss2_o  # (B,)

        return log_model_prob_x, log_model_prob_e, log_model_prob_o, vb_loss_x, vb_loss_e, vb_loss_o

    def truncate(self, log_p_x_0: Tensor, truncation_rate=1.):
        sorted_log_p_x_0, indices = torch.sort(log_p_x_0, dim=1, descending=True)
        sorted_p_x_0 = torch.exp(sorted_log_p_x_0)
        keep_mask = sorted_p_x_0.cumsum(dim=1) < truncation_rate

        # Ensure that at least the largest probability is not zeroed out
        all_true = torch.full_like(keep_mask[:, 0:1, :], True)
        keep_mask = torch.cat((all_true, keep_mask), dim=1)
        keep_mask = keep_mask[:, :-1, :]

        keep_mask = keep_mask.gather(1, indices.argsort(1))

        rv = log_p_x_0.clone()
        rv[~keep_mask] = -torch.inf  # -inf = log(0)
        return rv


################################################################


## Helper functions
def avg_except_batch(x: Tensor, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).mean(dim=-1)


def sum_except_batch(x: Tensor, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(dim=-1)


def log_1_min_a(a: Tensor):
    return torch.log(1. - a.exp() + 1e-40)


def log_add_exp(a: Tensor, b: Tensor):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a: Tensor, t: LongTensor, x_shape: Tuple[int, ...]):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def log_categorical(log_x_start: Tensor, log_prob: Tensor):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x: LongTensor, num_classes: int):
    assert x.max().item() < num_classes, f"Error: {x.max().item()} >= {num_classes}"

    x_onehot: Tensor = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.shape)))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x: Tensor):
    return log_x.argmax(dim=1)


def alpha_schedule(time_step: int, N: int, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999):
    att = np.arange(0, time_step) / (time_step-1) * (att_T - att_1) + att_1
    att = np.concatenate([[1], att])
    at = att[1:] / att[:-1]

    ctt = np.arange(0, time_step) / (time_step-1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate([[0], ctt])
    one_minus_ctt = 1. - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1. - one_minus_ct

    bt = (1. - at - ct) / N

    att = np.concatenate([att[1:], [1]])
    ctt = np.concatenate([ctt[1:], [0]])
    btt = (1. - att - ctt) / N

    return at, bt, ct, att, btt, ctt


################################################################


class SgObjfeatTransformerVQDiffusionWrapper(nn.Module):
    def __init__(self,
        node_dim: int, edge_dim: int,
        num_objfeat_embeds: int, objfeat_dim: int,
        attn_dim=512, t_dim=128,
        global_dim: Optional[int]=None,
        context_dim: Optional[int]=None,
        global_condition_dim: Optional[int]=None,
        n_heads=8, n_layers=5,
        gated_ff=True, dropout=0.1, ada_norm=True,
        use_e_cross_attn=False,
        concat_global_condition=True,
        use_positional_encoding=True
    ):
        super().__init__()
        assert (edge_dim - 2) % 2 == 0, f"Invalide `edge_dim`: {edge_dim}"  # predicate types are symmetric

        if not ada_norm:
            global_dim = t_dim  # not use AdaLN, use global information in graph-attn instead

        self.node_embed = nn.Sequential(
            nn.Embedding(node_dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, attn_dim)
        )

        self.edge_embed = nn.Sequential(
            nn.Embedding(edge_dim, attn_dim//4),
            nn.GELU(),
            nn.Linear(attn_dim//4, attn_dim//4)
        )

        self.time_embed = nn.Sequential(
            Timestep(t_dim),
            TimestepEmbed(t_dim, t_dim)
        )

        self.objfeat_embed = nn.Embedding(num_objfeat_embeds+2, objfeat_dim)  # +2 for empty and [mask] token

        self.objfeat_pool_transformer = nn.ModuleList([
            BasicTransformerBlock(  # self-attn + ff
                objfeat_dim, objfeat_dim, None, None,
                n_heads, gated_ff, dropout
            ) for _ in range(2)  # TODO: make it configurable
        ])
        self.objfeat_pool_norm = nn.LayerNorm(objfeat_dim)
        self.objfeat_pool = nn.Sequential(
            nn.Linear(objfeat_dim, objfeat_dim),
            nn.GELU(),
            nn.Linear(objfeat_dim, attn_dim)
        )

        if global_condition_dim is not None:
            if concat_global_condition:
                self.global_condition_embed = nn.Sequential(
                    nn.Linear(global_condition_dim, context_dim),
                    nn.GELU(),
                    nn.Linear(context_dim, context_dim)
                )
            else:
                self.global_condition_embed = nn.Sequential(
                    nn.Linear(global_condition_dim, t_dim),
                    nn.GELU(),
                    nn.Linear(t_dim, t_dim)
                )

        self.transformer_blocks = nn.ModuleList([
            GraphTransformerBlock(
                attn_dim, attn_dim//4, attn_dim, global_dim,
                context_dim, t_dim,
                n_heads, gated_ff, dropout, ada_norm,
                use_e_cross_attn
            ) for _ in range(n_layers)
        ])

        self.node_proj_out = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, node_dim-1)  # -1 for the [mask] token
        )
        self.edge_proj_out = nn.Sequential(
            nn.LayerNorm(attn_dim//4),
            nn.Linear(attn_dim//4, edge_dim-1)  # -1 for the [mask] token
        )

        self.out_objfeat_tokens = nn.Parameter(torch.empty(4, attn_dim))  # TODO: make `4` configurable
        self.out_objfeat_tokens.data.uniform_(-1./attn_dim, 1./attn_dim)
        self.out_objfeat_transformer = nn.ModuleList([
            BasicTransformerBlock(  # self-attn + cross-attn + ff
                attn_dim, attn_dim, attn_dim, None,
                n_heads, gated_ff, dropout
            ) for _ in range(2)  # TODO: make it configurable
        ])
        self.objfeat_out_norm = nn.LayerNorm(attn_dim)
        self.objfeat_out = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, num_objfeat_embeds+1)  # +1 for empty token
        )

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.use_global_info = global_dim is not None
        self.concat_global_condition = concat_global_condition
        self.use_positional_encoding = use_positional_encoding

    def forward(self,
        x: LongTensor, e: LongTensor, o: LongTensor,
        t: LongTensor, condition: Optional[Tensor]=None,
        global_condition: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, condition_mask: Optional[LongTensor]=None
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

        B, N, E, NK = x.shape[0], x.shape[-1], e.shape[-1], o.shape[-1]
        assert E == N * (N - 1) // 2
        assert NK == N * 4  # TODO: make `4` configurable

        # Symmetrize edges
        e_onehot = F.one_hot(e, self.edge_dim).float()
        e_onehot = scatter_trilist_to_matrix(e_onehot, N)  # (B, N, N, Ce); lower triangle is all zeros
        e_onehot_negative = e_onehot[..., 
            [*range((self.edge_dim-2)//2, self.edge_dim-2)] + \
            [*range(0, (self.edge_dim-2)//2)] + \
            [*range(self.edge_dim-2, e_onehot.shape[-1])]
        ]  # (B, N, N, n_pred_types+2) (+2 for empty and [mask])
        e_onehot = e_onehot + e_onehot_negative.permute(0, 2, 1, 3)
        eye_mask = torch.eye(N, device=e_onehot.device).bool().unsqueeze(0)  # (1, N, N)
        assert torch.all(e_onehot.sum(dim=-1) == (~eye_mask).float())  # every edge is one-hot encoded, except for the diagonal
        e = torch.argmax(e_onehot, dim=-1)  # (B, N, N)

        x_emb = self.node_embed(x)  # (B, N, attn_dim)
        e_emb = self.edge_embed(e)  # (B, N, N, attn_dim//4)
        t_emb = self.time_embed(t)  # (B, N, t_dim)
        if self.use_global_info:
            y_emb = t_emb
        else:
            y_emb =None
        if global_condition is not None:
            if self.concat_global_condition:
                condition = torch.cat([
                    condition,
                    self.global_condition_embed(global_condition).unsqueeze(1)
                ], dim=1)
            else:
                t_emb += self.global_condition_embed(global_condition)

        # Mask out the diagonal
        e_emb = e_emb * (~eye_mask).float().unsqueeze(-1)

        # Object feature embedding
        o = o.reshape(B, N, -1)  # (B, N, K)
        o_emb = self.objfeat_embed(o)  # (B, N, K, Do)
        o_emb = o_emb.reshape(B*N, 4, -1)  # (B*N, K, Do); TODO: make `4` configurable
        o_emb = o_emb + get_1d_sincos_encode(
            torch.arange(o_emb.shape[1], device=o_emb.device),
            o_emb.shape[-1], o_emb.shape[1]
        ).unsqueeze(0)
        for block in self.objfeat_pool_transformer:
            o_emb = block(o_emb)
        o_emb = self.objfeat_pool_norm(o_emb)  # (B*N, K, Do)
        o_emb = torch.mean(o_emb, dim=1).reshape(B, N, -1)  # (B, N, Do); average over objfeat tokens
        o_emb = self.objfeat_pool(o_emb)  # (B, N, attn_dim)

        # Combine object feature embedding with node embedding
        x_emb = x_emb + o_emb

        # Instance ID embeddings
        if self.use_positional_encoding:
            inst_emb_x = get_1d_sincos_encode(
                torch.arange(x_emb.shape[1], device=x_emb.device),
                x_emb.shape[-1], x_emb.shape[1]
            ).unsqueeze(0)  # (1, N, Dx)
            x_emb = x_emb + inst_emb_x
            inst_emb_e = get_1d_sincos_encode(
                torch.arange(e_emb.shape[1], device=e_emb.device),
                e_emb.shape[-1], e_emb.shape[1]
            ).unsqueeze(0)  # (1, N, De)
            e_emb = e_emb + inst_emb_e.unsqueeze(1) + inst_emb_e.unsqueeze(2)

        for block in self.transformer_blocks:
            x_emb, e_emb, y_emb = block(x_emb, e_emb, y_emb, t_emb, condition, mask, condition_mask)

        out_x = self.node_proj_out(x_emb)
        out_e = self.edge_proj_out(e_emb)

        # Output a set of object feature tokens
        out_o_tokens = self.out_objfeat_tokens.unsqueeze(0).repeat(B*N, 1, 1)  # (B*N, K, attn_dim)
        out_o_tokens = out_o_tokens + get_1d_sincos_encode(
            torch.arange(out_o_tokens.shape[1], device=out_o_tokens.device),
            out_o_tokens.shape[-1], out_o_tokens.shape[1]
        ).unsqueeze(0)
        for block in self.out_objfeat_transformer:
            out_o_tokens = block(out_o_tokens, context=x_emb.reshape(B*N, 1, -1))
        out_o_tokens = self.objfeat_out_norm(out_o_tokens)  # (B*N, K, attn_dim)
        out_o = self.objfeat_out(out_o_tokens)  # (B*N, K, num_objfeat_embeds+1)
        out_o = out_o.reshape(B, N, 4, -1)  # (B, N, K, num_objfeat_embeds+1); TODO: make `4` configurable

        if mask is not None:
            x_mask = mask.unsqueeze(-1)    # (B, N, 1)
            e_mask1 = x_mask.unsqueeze(2)  # (B, N, 1, 1)
            e_mask2 = x_mask.unsqueeze(1)  # (B, 1, N, 1)
            out_x = out_x * x_mask
            out_e = out_e * e_mask1 * e_mask2
            out_o = out_o * x_mask.unsqueeze(-1)

        # Mix the predictions of two directions of edges
        out_e_negative = out_e[...,
            [*range((self.edge_dim-2)//2, self.edge_dim-2)] + \
            [*range(0, (self.edge_dim-2)//2)] + \
            [*range(self.edge_dim-2, out_e.shape[-1])]
        ]  # (B, N, N, n_pred_types+2) (+2 for empty and [mask])
        out_e = (out_e + out_e_negative.permute(0, 2, 1, 3)) / 2.

        # Gather upper triangle of edge predictions
        gather_idx = tri_idx_to_full_idx(N)[None, :, None].to(x.device)  # (1, E, 1)
        gather_idx = gather_idx.expand(B, -1, out_e.shape[-1])  # (B, E, Ce)
        out_e = torch.gather(out_e.reshape(B, N*N, -1), dim=1, index=gather_idx)  # (B, E, Ce)
        assert out_e.shape[1] == E

        out_x = out_x.permute(0, 2, 1)  # (B, Cx, N)
        out_e = out_e.permute(0, 2, 1)  # (B, Ce, E)
        out_o = out_o.reshape(B, N*4, -1).permute(0, 2, 1)  # (B, Do, NK); TODO: make `4` configurable

        return out_x, out_e, out_o


################################################################


## Helper functions
def map_upper_triangle_to_list(i: int, j: int, K: int) -> int:
    assert i < j, f"Index `i`({i}) >= `j`({j}): not an upper triangle"
    e_list_idx = i * (2 * K - i - 1) // 2 + j - i - 1
    return e_list_idx


def tri_idx_to_full_idx(K: int) -> LongTensor:
    tri_indices, full_indices = [], []
    for i in range(K):
        for j in range(i+1, K):
            tri_indices.append(map_upper_triangle_to_list(i, j, K))  # for sanity check
            full_indices.append(i * K + j)

    assert tri_indices == [i for i in range(len(tri_indices))]
    return torch.tensor(full_indices).long()  # (K*(K-1)/2,)


def scatter_trilist_to_matrix(buffer: Tensor, K: int) -> Tensor:
    B, E, F = buffer.shape
    assert E == K * (K - 1) // 2

    indices = tri_idx_to_full_idx(K).to(buffer.device)[None, :, None]  # (1, E=K*(K-1)/2, 1)
    indices = indices.expand(B, -1, F)  # (B, E, De)
    ret = torch.zeros(B, K*K, F, dtype=buffer.dtype, device=buffer.device)
    ret = torch.scatter(ret, dim=1, index=indices, src=buffer)
    ret = ret.reshape(B, K, K, F)

    return ret  # lower triangle is all zeros
