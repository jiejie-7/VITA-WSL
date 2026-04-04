from __future__ import annotations

from typing import Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrustPredictor(nn.Module):
    """Trust estimator with label-free gating and action-prediction supervision."""

    def __init__(self, hidden_dim: int, action_dim: int, gamma: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))
        self.gamma = float(gamma)

    def forward(
        self,
        neighbor_feat: torch.Tensor,
        neighbor_next_action: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict neighbor actions and derive trust from predictive confidence.

        Trust must be computable at rollout time without using ground-truth neighbor
        actions. We therefore use normalized action entropy as a label-free trust
        signal, while `neighbor_next_action` is reserved for the auxiliary loss in
        the caller.
        """
        if neighbor_feat.dim() != 3:
            raise ValueError(f"Expected neighbor_feat to have shape [B, K, H], got {tuple(neighbor_feat.shape)}")

        logits = self.net(neighbor_feat)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        denom = math.log(max(int(logits.size(-1)), 2))
        entropy = entropy / denom
        trust = torch.exp(-self.gamma * entropy)
        trust = torch.nan_to_num(trust, nan=1.0, posinf=1.0, neginf=1.0)
        return logits, trust.clamp(min=1e-6, max=1.0)
