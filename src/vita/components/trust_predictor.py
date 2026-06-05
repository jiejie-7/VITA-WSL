from __future__ import annotations

from typing import Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrustPredictor(nn.Module):
    """Trust estimator with confidence, maliciousness, and utility heads."""

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        gamma: float,
        *,
        pair_product: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.ReLU())
        self.receiver_context_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.pair_fusion = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU())
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.malicious_head = nn.Linear(hidden_dim, 1)
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.gamma = float(gamma)
        self.pair_product = bool(pair_product)

    def forward(
        self,
        neighbor_feat: torch.Tensor,
        neighbor_uncertainty: torch.Tensor | None = None,
        receiver_context: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict trust without requiring labels at rollout time.

        The action head preserves the original confidence-based signal, while the
        malicious and utility heads condition on both sender-side messages and the
        receiver context. This helps separate true communication corruption from
        harmless heterogeneity or battle-state-dependent exploration.
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
        if receiver_context is None:
            receiver_context = torch.zeros(
                neighbor_feat.size(0),
                neighbor_feat.size(-1),
                device=neighbor_feat.device,
                dtype=neighbor_feat.dtype,
            )
        if receiver_context.dim() != 2:
            raise ValueError(f"Expected receiver_context to have shape [B, H], got {tuple(receiver_context.shape)}")
        x = torch.cat([neighbor_feat, neighbor_uncertainty], dim=-1)
        sender_hidden = self.net(x)
        receiver_hidden = self.receiver_context_net(receiver_context).unsqueeze(1).expand(-1, neighbor_feat.size(1), -1)
        if self.pair_product:
            pair_relation = sender_hidden * receiver_hidden
        else:
            pair_relation = torch.abs(sender_hidden - receiver_hidden)
        pair_input = torch.cat([sender_hidden, receiver_hidden, pair_relation], dim=-1)
        hidden = sender_hidden + self.pair_fusion(pair_input)
        logits = self.action_head(hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        denom = math.log(max(int(logits.size(-1)), 2))
        entropy = entropy / denom

        confidence_trust = torch.exp(-self.gamma * entropy)
        malicious_logits = self.malicious_head(hidden)
        utility_logits = self.utility_head(hidden)
        return logits, confidence_trust.clamp(min=1e-6, max=1.0), malicious_logits, utility_logits
