from __future__ import annotations

from typing import Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrustPredictor(nn.Module):
    """Trust estimator with confidence, maliciousness, and utility heads."""

    def __init__(self, hidden_dim: int, action_dim: int, gamma: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.ReLU())
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.malicious_head = nn.Linear(hidden_dim, 1)
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.gamma = float(gamma)

    def forward(
        self,
        neighbor_feat: torch.Tensor,
        neighbor_uncertainty: torch.Tensor | None = None,
        neighbor_next_action: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict trust without requiring labels at rollout time.

        The action head preserves the original confidence-based signal, while the
        malicious head learns a direct edge-level reliability score whenever
        malicious labels are available during PPO updates.
        """
        if neighbor_feat.dim() != 3:
            raise ValueError(f"Expected neighbor_feat to have shape [B, K, H], got {tuple(neighbor_feat.shape)}")
        if neighbor_uncertainty is None:
            neighbor_uncertainty = torch.zeros(
                neighbor_feat.size(0),
                neighbor_feat.size(1),
                1,
                device=neighbor_feat.device,
                dtype=neighbor_feat.dtype,
            )
        x = torch.cat([neighbor_feat, neighbor_uncertainty], dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        hidden = self.net(x)
        logits = self.action_head(hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        denom = math.log(max(int(logits.size(-1)), 2))
        entropy = entropy / denom

        confidence_trust = torch.exp(-self.gamma * entropy)
        confidence_trust = torch.nan_to_num(confidence_trust, nan=1.0, posinf=1.0, neginf=1.0)
        malicious_logits = self.malicious_head(hidden)
        utility_logits = self.utility_head(hidden)
        return logits, confidence_trust.clamp(min=1e-6, max=1.0), malicious_logits, utility_logits
