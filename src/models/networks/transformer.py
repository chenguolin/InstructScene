from typing import *
from torch import Tensor, LongTensor

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange


class Attention(nn.Module):
    def __init__(self,
        query_dim: int, context_dim: Optional[int]=None,
        n_heads=8, hidden_dim=512, dropout=0.
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, \
            f"Hidden dimension ({hidden_dim}) must be divisible by number of heads ({n_heads})"
        head_dim = hidden_dim // n_heads

        if context_dim is None:
            context_dim = query_dim

        self.scale = head_dim ** -0.5
        self.n_heads = n_heads

        self.to_q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(context_dim, hidden_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self,
        x: Tensor,
        context: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, context_mask: Optional[LongTensor]=None
    ):
        h = self.n_heads

        q = self.to_q(x)  # (b, n, d*h)
        if mask is not None:
            q = q * mask.unsqueeze(-1)

        # If context is not provided, use self-attention
        if context is None:
            context = x
            context_mask = mask

        k = self.to_k(context)  # (b, m, d*h)
        v = self.to_v(context)  # (b, m, d*h)
        if context_mask is not None:
            k = k * context_mask.unsqueeze(-1)
            v = v * context_mask.unsqueeze(-1)

        q, k, v = map(lambda t: rearrange(
            t, "b n (h d) -> (b h) n d", h=h), (q, k, v))  # (b*h, n or m, d)

        sim: Tensor = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale  # (b*h, n, m)

        if context_mask is not None:
            attn_mask = context_mask.unsqueeze(1).unsqueeze(-1)  # (b, 1, m, 1)
            attn_mask = attn_mask.expand(-1, q.shape[1], -1, h)  # (b, n, m, h)
            attn_mask = rearrange(attn_mask, "b n m h -> (b h) n m").bool()  # (b*h, n, m)
            sim = sim.masked_fill(~attn_mask, float("-inf"))
        attn = sim.softmax(dim=-1)  # (b*h, n, m)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)  # (b*h, n, d)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)  # (b, n, d*h)
        out = self.to_out(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1)

        return out


class GraphAttention(nn.Module):
    def __init__(self,
        node_dim: int, edge_dim: int, global_dim: Optional[int]=None,
        n_heads=8, hidden_dim=512, dropout=0.
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, \
            f"Hidden dimension ({hidden_dim}) must be divisible by number of heads ({n_heads})"
        head_dim = hidden_dim // n_heads

        self.scale = head_dim ** -0.5
        self.n_heads = n_heads

        self.to_q = nn.Linear(node_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(node_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(node_dim, hidden_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, node_dim),
            nn.Dropout(dropout)
        )

        # FiLM-style edge and node conditioning
        self.to_e_mul = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.to_e_add = nn.Linear(edge_dim, hidden_dim, bias=False)

        self.to_e_out = nn.Sequential(
            nn.Linear(hidden_dim, edge_dim),
            nn.Dropout(dropout)
        )

        if global_dim is not None:
            # FiLM-style global conditioning
            self.to_y_x_mul = nn.Linear(global_dim, hidden_dim, bias=False)
            self.to_y_x_add = nn.Linear(global_dim, hidden_dim, bias=False)
            self.to_y_e_mul = nn.Linear(global_dim, hidden_dim, bias=False)
            self.to_y_e_add = nn.Linear(global_dim, hidden_dim, bias=False)

            self.to_yx_out = nn.Linear(node_dim, global_dim)
            self.to_ye_out = nn.Linear(edge_dim, global_dim)

        self.use_global_info = global_dim is not None

    def forward(self,
        x: Tensor, e: Tensor, y: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None
    ):
        h = self.n_heads
        if mask is not None:
            x_mask = mask.unsqueeze(-1)     # (b, n) -> (b, n, 1)
            e_mask1 = x_mask.unsqueeze(-1)  # (b, n, 1, 1)
            e_mask2 = x_mask.unsqueeze(1)   # (b, 1, n, 1)

        q = self.to_q(x)  # (b, n, d*h)
        if mask is not None:
            q = q * x_mask

        k = self.to_k(x)  # (b, n, d*h)
        v = self.to_v(x)  # (b, n, d*h)
        if mask is not None:
            k = k * x_mask
            v = v * x_mask

        q, k, v = map(lambda t: rearrange(
            t, "b n (h d) -> (b h) n d", h=h), (q, k, v))  # (b*h, n, d)

        sim: Tensor = q.unsqueeze(2) * k.unsqueeze(1) * self.scale  # (b*h, n, n, d); outer product

        # Incorporate edge information into node similarity matrix
        e_mul = self.to_e_mul(e)  # (b, n, n, d*h)
        e_add = self.to_e_add(e)  # (b, n, n, d*h)
        if mask is not None:
            e_mul = e_mul * e_mask1 * e_mask2
            e_add = e_add * e_mask1 * e_mask2
        e_mul, e_add = map(lambda t: rearrange(
            t, "b n m (h d) -> (b h) n m d", h=h), (e_mul, e_add))  # (b*h, n, n, d)
        sim = (1. + e_mul) * sim + e_add  # (b*h, n, n, d)

        # Incorporate global information into edge features
        e_out = rearrange(sim, "(b h) n m d -> b n m (h d)", h=h)  # (b, n, n, d*h)
        if self.use_global_info:
            ye_mul = self.to_y_e_mul(y).unsqueeze(1).unsqueeze(1)  # (b, 1, 1, d*h)
            ye_add = self.to_y_e_add(y).unsqueeze(1).unsqueeze(1)  # (b, 1, 1, d*h)
            e_out = (1. + ye_mul) * e_out + ye_add  # (b, n, n, d*h)

        # Output new edge features
        e_out = self.to_e_out(e_out)
        if mask is not None:
            e_out = e_out * e_mask1 * e_mask2  # (b, n, n, de)

        # Normalize attention weights
        if mask is not None:
            attn_mask = e_mask2.expand(-1, q.shape[1], -1, h)  # (b, n, n, h)
            attn_mask = rearrange(attn_mask, "b n m h -> (b h) n m ()").bool()  # (b*h, n, n, 1)
            sim = sim.masked_fill(~attn_mask, float("-inf"))
        attn = sim.softmax(dim=2)  # (b*h, n, n, d)

        # Apply attention weights to node features & Incorporate global information into node features
        out = (attn * v.unsqueeze(1)).sum(dim=2)  # (b*h, n, d)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        if self.use_global_info:
            yx_mul = self.to_y_x_mul(y).unsqueeze(1)  # (b, 1, d*h)
            yx_add = self.to_y_x_add(y).unsqueeze(1)  # (b, 1, d*h)
            out = (1. + yx_mul) * out + yx_add  # (b, n, d*h)

        # Output new node features
        out = self.to_out(out)
        if mask is not None:
            out = out * x_mask  # (b, n, dx)

        # Output new global features
        if self.use_global_info:
            if mask is not None:
                yx_out = out.sum(dim=1) / x_mask.expand(-1, -1, out.shape[-1]).sum(dim=1)  # (b, dx)
                ye_out = e_out.sum(dim=(1, 2)) / \
                    (e_mask1 * e_mask2).expand(-1, -1, -1, e_out.shape[-1]).sum(dim=(1, 2))  # (b, de)
            else:
                yx_out = out.mean(dim=1)  # (b, dx)
                ye_out = e_out.mean(dim=(1, 2))  # (b, de)
            yx_out = self.to_yx_out(yx_out)
            ye_out = self.to_ye_out(ye_out)
            y_out = y + yx_out + ye_out
        else:
            y_out = None

        return out, e_out, y_out


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out*2)

    def forward(self, x: Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self,
        dim: int, dim_out: Optional[int]=None,
        mult=4, gated=False, dropout=0.
    ):
        super().__init__()
        hidden_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim

        project_in = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU()
        ) if not gated else GEGLU(dim, hidden_dim)

        self.mlp = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_out)
        )

    def forward(self, x: Tensor):
        return self.mlp(x)


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, t_dim: int):
        super().__init__()
        self.gelu = nn.GELU()
        self.linear = nn.Linear(t_dim, dim*2)
        self.layernorm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x: Tensor, t_emb: Tensor):
        emb: Tensor = self.linear(self.gelu(t_emb)).unsqueeze(1)
        while emb.dim() < x.dim():
            emb = emb.unsqueeze(1)

        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.layernorm(x) * (1. + scale) + shift

        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self,
        dim: int, attn_dim: int,
        context_dim: Optional[int]=None, t_dim: Optional[int]=None,
        n_heads=8, gated_ff=True, dropout=0., ada_norm=False
    ):
        super().__init__()
        if ada_norm:
            assert t_dim is not None, "Parameter `t_dim` must be provided for AdaLN"

        self.attn = Attention(dim, None, n_heads, attn_dim, dropout)
        self.ff = FeedForward(dim, gated=gated_ff, dropout=dropout)

        self.attn_norm = AdaLayerNorm(dim, t_dim) if ada_norm else nn.LayerNorm(dim)
        self.ff_norm = AdaLayerNorm(dim, t_dim) if ada_norm else nn.LayerNorm(dim)

        if context_dim is not None:
            self.cross_attn = Attention(dim, context_dim, n_heads, attn_dim, dropout)
            self.ca_norm = AdaLayerNorm(dim, t_dim) if ada_norm else nn.LayerNorm(dim)

    def forward(self,
        x: Tensor, t_emb: Optional[Tensor]=None,
        context: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, context_mask: Optional[LongTensor]=None
    ):
        # 1. Self-attention
        x_norm = self.attn_norm(x, t_emb) if t_emb is not None else self.attn_norm(x)
        x = self.attn(x_norm, None, mask) + x

        # 2. (Optional) Cross-attention
        if context is not None:
            x_norm = self.ca_norm(x, t_emb) if t_emb is not None else self.ca_norm(x)
            x = self.cross_attn(x_norm, context, mask, context_mask) + x

        # 3. Feed-forward MLPs
        x_norm = self.ff_norm(x, t_emb) if t_emb is not None else self.ff_norm(x)
        x = self.ff(x_norm) + x

        return x


class GraphTransformerBlock(nn.Module):
    def __init__(self,
        node_dim: int, edge_dim: int, attn_dim: int, global_dim: Optional[int]=None,
        context_dim: Optional[int]=None, t_dim: Optional[int]=None,
        n_heads=8, gated_ff=True, dropout=0., ada_norm=False,
        use_e_cross_attn=False
    ):
        super().__init__()
        if ada_norm:
            assert t_dim is not None, "Parameter `t_dim` must be provided for AdaLN"
        # If not use AdaLN, `t_dim` is not really used

        self.graph_attn = GraphAttention(node_dim, edge_dim, global_dim, n_heads, attn_dim, dropout)
        self.ff_x = FeedForward(node_dim, gated=gated_ff, dropout=dropout)
        self.ff_e = FeedForward(edge_dim, gated=gated_ff, dropout=dropout)

        self.ga_x_norm = AdaLayerNorm(node_dim, t_dim) if ada_norm else nn.LayerNorm(node_dim)
        self.ff_x_norm = AdaLayerNorm(node_dim, t_dim) if ada_norm else nn.LayerNorm(node_dim)
        self.ga_e_norm = AdaLayerNorm(edge_dim, t_dim) if ada_norm else nn.LayerNorm(edge_dim)
        self.ff_e_norm = AdaLayerNorm(edge_dim, t_dim) if ada_norm else nn.LayerNorm(edge_dim)

        if context_dim is not None:
            self.cross_attn = Attention(node_dim, context_dim, n_heads, attn_dim, dropout)
            self.ca_norm = AdaLayerNorm(node_dim, t_dim) if ada_norm else nn.LayerNorm(node_dim)
            if use_e_cross_attn:
                self.cross_attn_e = Attention(edge_dim, context_dim, n_heads, attn_dim, dropout)
                self.ca_norm_e = AdaLayerNorm(edge_dim, t_dim) if ada_norm else nn.LayerNorm(edge_dim)

        if global_dim is not None:
            self.ff_y = FeedForward(global_dim, gated=gated_ff, dropout=dropout)
            self.ga_y_norm = AdaLayerNorm(global_dim, t_dim) if ada_norm else nn.LayerNorm(global_dim)
            self.ff_y_norm = AdaLayerNorm(global_dim, t_dim) if ada_norm else nn.LayerNorm(global_dim)

        self.ada_norm = ada_norm
        self.use_e_cross_attn = use_e_cross_attn
        self.with_global_info = global_dim is not None

    def forward(self,
        x: Tensor, e: Tensor, y: Optional[Tensor]=None,
        t_emb: Optional[Tensor]=None, context: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, context_mask: Optional[LongTensor]=None
    ):
        # 1. Graph attention
        x_norm = self.ga_x_norm(x, t_emb) if self.ada_norm else self.ga_x_norm(x)
        e_norm = self.ga_e_norm(e, t_emb) if self.ada_norm else self.ga_e_norm(e)
        if self.with_global_info:
            y_norm = self.ga_y_norm(y, t_emb) if self.ada_norm else self.ga_y_norm(y)
        else:
            y_norm = None
        x_, e_, y_ = self.graph_attn(x_norm, e_norm, y_norm, mask)
        x, e = x_ + x, e_ + e
        if y_ is not None:
            y = y_ + y

        # 2. (Optional) Cross-attention with nodes(edges) and context
        if context is not None:
            x_norm = self.ca_norm(x, t_emb) if self.ada_norm else self.ca_norm(x)
            x = self.cross_attn(x_norm, context, mask, context_mask) + x
            if self.use_e_cross_attn:
                e_norm = self.ca_norm_e(e, t_emb) if self.ada_norm else self.ca_norm_e(e)
                e_norm = rearrange(e_norm, "b n m d -> b (n m) d")
                e_ = self.cross_attn_e(e_norm, context, mask, context_mask)
                e_ = rearrange(e_, "b (n m) d -> b n m d", n=e.shape[1])
                e = e_ + e

        # 3. Node-wise and edge-wise MLPs
        x_norm = self.ff_x_norm(x, t_emb) if self.ada_norm else self.ff_x_norm(x)
        x = self.ff_x(x_norm) + x
        e_norm = self.ff_e_norm(e, t_emb) if self.ada_norm else self.ff_e_norm(e)
        e = self.ff_e(e_norm) + e
        if self.with_global_info:
            y_norm = self.ff_y_norm(y, t_emb) if self.ada_norm else self.ff_y_norm(y)
            y = self.ff_y(y_norm) + y
        else:
            y = None

        return x, e, y
