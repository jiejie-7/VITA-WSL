from __future__ import annotations

from typing import Tuple

import math
import torch
import torch.nn as nn


class VIBGATLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        kl_beta: float,
        bias_coef: float,
        kl_free_bits: float = 0.0,
        attention_only: bool = False,
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.to_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.to_logvar = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.direct_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.post_norm = nn.LayerNorm(latent_dim)
        self.message_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, hidden_dim)
        self.bias_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.bias_coef = bias_coef
        self.kl_beta = kl_beta
        self.kl_free_bits = float(kl_free_bits)
        self.logvar_min = -8.0
        self.logvar_max = 4.0
        self.attention_only = bool(attention_only)

    def encode_messages(
        self,
        neighbor_feat: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neighbor_feat = torch.nan_to_num(neighbor_feat, nan=0.0, posinf=0.0, neginf=0.0)
        norm_neighbors = self.pre_norm(neighbor_feat)
        norm_neighbors = torch.nan_to_num(norm_neighbors, nan=0.0, posinf=0.0, neginf=0.0)

        if self.attention_only:
            messages = self.direct_latent(norm_neighbors)
            kl_raw = torch.zeros(1, device=neighbor_feat.device, dtype=neighbor_feat.dtype)
            kl_scaled = torch.zeros(1, device=neighbor_feat.device, dtype=neighbor_feat.dtype)
            return self.post_norm(messages), kl_scaled, kl_raw

        mu = self.to_mu(norm_neighbors)
        logvar = self.to_logvar(norm_neighbors)
        logvar = logvar.clamp(min=self.logvar_min, max=self.logvar_max)
        if deterministic:
            messages = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            messages = mu + eps * std
        messages = self.post_norm(messages)

        kl_per_edge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        if valid_mask is None:
            edge_mask = torch.ones_like(kl_per_edge)
        else:
            edge_mask = torch.nan_to_num(valid_mask.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        denom = edge_mask.sum().clamp_min(1.0)
        kl_raw = (kl_per_edge * edge_mask).sum() / denom
        kl_raw = kl_raw / math.log(2.0)
        if self.kl_free_bits > 0.0:
            kl_for_loss = torch.clamp(kl_raw - self.kl_free_bits, min=0.0)
        else:
            kl_for_loss = kl_raw
        kl_scaled = self.kl_beta * kl_for_loss
        return messages, kl_scaled, kl_raw

    def decode_messages(self, messages: torch.Tensor) -> torch.Tensor:
        messages = torch.nan_to_num(messages, nan=0.0, posinf=0.0, neginf=0.0)
        feat = self.message_decoder(messages)
        feat = self.message_norm(feat)
        return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    def aggregate_messages(
        self,
        self_feat: torch.Tensor,
        recv_messages: torch.Tensor,
        trust_mask: torch.Tensor,
        comm_mask: torch.Tensor,
        recv_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        recv_messages = torch.nan_to_num(recv_messages, nan=0.0, posinf=0.0, neginf=0.0)
        if recv_features is None:
            recv_features = self.decode_messages(recv_messages)

        query = self.query_proj(self_feat).unsqueeze(1)
        keys = self.key_proj(recv_messages)
        values = self.value_proj(recv_messages)

        attn_logits = torch.matmul(query, keys.transpose(-2, -1)).squeeze(1) / (keys.size(-1) ** 0.5)
        bias_input = torch.abs(self_feat.unsqueeze(1) - recv_features)
        bias = self.bias_mlp(bias_input).squeeze(-1)
        attn_logits = attn_logits - self.bias_coef * bias

        trust_mask = torch.nan_to_num(trust_mask, nan=1e-6, posinf=1.0, neginf=1e-6).clamp(min=1e-6, max=1.0)
        trust_term = torch.log(trust_mask.squeeze(-1))
        attn_logits = attn_logits + trust_term

        neighbor_mask = torch.nan_to_num(comm_mask.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        attn_logits = attn_logits.masked_fill(neighbor_mask < 1e-6, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = attn_weights * neighbor_mask
        norm = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn_weights = (attn_weights / norm).unsqueeze(-2)
        weighted_values = values * comm_mask
        context = torch.matmul(attn_weights, weighted_values).squeeze(-2)
        comm_feat = self.out_proj(context)
        return torch.nan_to_num(comm_feat, nan=0.0, posinf=0.0, neginf=0.0)
