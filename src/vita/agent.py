from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .components import FeatureEncoder, VIBGATLayer, GatedResidualBlock, TrustPredictor


@dataclass
class VITAAgentConfig:
    obs_dim: int
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    latent_dim: int = 64
    trust_gamma: float = 1.0
    kl_beta: float = 1e-3
    kl_free_bits: float = 0.0
    trust_lambda: float = 0.1
    trust_malicious_weight: float = 1.0
    trust_margin_weight: float = 0.5
    trust_margin: float = 0.1
    trust_reliability_mix: float = 0.5
    trust_counterfactual_mix: float = 0.5
    trust_counterfactual_margin: float = 0.02
    trust_counterfactual_weight: float = 1.0
    max_neighbors: int = 5
    comm_dropout: float = 0.1
    enable_trust: bool = True
    enable_kl: bool = True
    vib_deterministic: bool = False
    attention_only: bool = False
    trust_hard_topk: bool = False
    trust_topk_k: int = 0
    trust_gate_floor: float = 0.0
    attn_bias_coef: float = 1.0
    trust_malicious_gate_coef: float = 1.0
    allocation_sharpness: float = 1.0
    allocation_floor: float = 0.0


class VITAAgent(torch.nn.Module):
    def __init__(self, cfg: VITAAgentConfig):
        super().__init__()
        self.cfg = cfg
        self.actor_encoder = FeatureEncoder(cfg.obs_dim, cfg.hidden_dim)
        self.critic_encoder = FeatureEncoder(cfg.state_dim, cfg.hidden_dim)
        self.comm_encoder = FeatureEncoder(cfg.obs_dim, cfg.hidden_dim)
        self.trust_predictor = TrustPredictor(cfg.hidden_dim, cfg.action_dim, cfg.trust_gamma)
        self.vib_gat = VIBGATLayer(
            cfg.hidden_dim,
            cfg.latent_dim,
            cfg.kl_beta,
            cfg.attn_bias_coef,
            cfg.kl_free_bits,
            attention_only=cfg.attention_only,
        )
        self.residual = GatedResidualBlock(cfg.hidden_dim)
        self.neighbor_norm = torch.nn.LayerNorm(cfg.hidden_dim)
        self.comm_dropout = torch.nn.Dropout(cfg.comm_dropout)
        critic_hidden = max(cfg.hidden_dim, 256)
        self.critic_mlp = torch.nn.Sequential(
            torch.nn.Linear(cfg.hidden_dim, critic_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(critic_hidden, critic_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(critic_hidden, cfg.hidden_dim),
        )
        self.policy_head = torch.nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.value_head = torch.nn.Linear(cfg.hidden_dim, 1)
        self.comm_enabled = True
        self.comm_strength = 0.0
        self.trust_strength = 0.0

    def load_state_dict(self, state_dict, strict: bool = True):
        has_trust = any(key.startswith("trust_predictor.") for key in state_dict)
        has_utility_head = any(key.startswith("trust_predictor.utility_head.") for key in state_dict)
        if (not has_trust) or (not has_utility_head):
            return super().load_state_dict(state_dict, strict=False)
        return super().load_state_dict(state_dict, strict=strict)

    def set_comm_enabled(self, enabled: bool) -> None:
        self.comm_enabled = enabled
        if not enabled:
            self.comm_strength = 0.0

    def set_comm_strength(self, strength: float) -> None:
        self.comm_strength = float(max(0.0, min(1.0, strength)))

    def set_trust_strength(self, strength: float) -> None:
        if not self.cfg.enable_trust:
            self.trust_strength = 0.0
            return
        self.trust_strength = float(max(0.0, min(1.0, strength)))

    def set_trust_active(self, active: bool) -> None:
        self.set_trust_strength(1.0 if active else 0.0)

    @property
    def rnn_hidden_dim(self) -> int:
        return self.cfg.hidden_dim

    def _zero_scalar(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, device=ref.device, dtype=ref.dtype)

    def _encode_neighbors(self, neighbor_seq: torch.Tensor) -> torch.Tensor:
        bsz, k, t, dim = neighbor_seq.shape
        flat = neighbor_seq.view(bsz * k, t, dim)
        feat, _ = self.comm_encoder(flat, None, None)
        feat = self.neighbor_norm(feat)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat.view(bsz, k, -1)

    def _mask_logits(self, logits: torch.Tensor, avail_actions: torch.Tensor | None) -> torch.Tensor:
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        if avail_actions is None:
            return logits
        avail_actions = torch.nan_to_num(avail_actions, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        logits = logits + avail_actions * 1e-6
        mask = avail_actions < 0.5
        all_masked = mask.all(dim=-1, keepdim=True)
        if all_masked.any():
            idx = torch.argmax(avail_actions, dim=-1, keepdim=True)
            mask = mask.clone()
            mask.scatter_(dim=-1, index=idx, value=False)
        return logits.masked_fill(mask, -1e9)

    def _compute_utility(
        self,
        recv_feat: torch.Tensor,
        neighbor_uncertainty: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trust_logits, confidence_trust, malicious_logits, utility_logits = self.trust_predictor(
            recv_feat,
            neighbor_uncertainty,
        )
        malicious_prob = torch.sigmoid(malicious_logits)
        malicious_prob = torch.nan_to_num(malicious_prob, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        confidence_trust = torch.nan_to_num(confidence_trust, nan=1.0, posinf=1.0, neginf=1.0).clamp(1e-6, 1.0)
        mix = float(max(0.0, min(1.0, getattr(self.cfg, "trust_reliability_mix", 0.5))))
        malicious_gate_coef = float(max(0.0, getattr(self.cfg, "trust_malicious_gate_coef", 1.0)))
        malicious_gate = (1.0 - malicious_gate_coef * malicious_prob).clamp(0.0, 1.0)
        reliability = (1.0 - mix) * confidence_trust + mix * malicious_gate
        reliability = torch.nan_to_num(reliability, nan=1.0, posinf=1.0, neginf=0.0).clamp(1e-6, 1.0)
        utility_score = torch.sigmoid(utility_logits)
        utility_score = torch.nan_to_num(utility_score, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return trust_logits, malicious_logits, utility_logits, reliability, utility_score

    def _compute_allocation(self, utility_scores: torch.Tensor) -> torch.Tensor:
        utility_scores = torch.nan_to_num(utility_scores, nan=1.0, posinf=1.0, neginf=1.0).clamp(1e-6, 1.0)
        sharpness = float(max(1e-6, getattr(self.cfg, "allocation_sharpness", 1.0)))
        allocation = utility_scores.pow(sharpness)
        floor = float(getattr(self.cfg, "allocation_floor", 0.0))
        if floor <= 0.0:
            floor = float(getattr(self.cfg, "trust_gate_floor", 0.0))
        floor = float(max(0.0, min(1.0, floor)))
        if floor > 0.0:
            allocation = floor + (1.0 - floor) * allocation
        return allocation.clamp(1e-6, 1.0)

    def _apply_hard_topk(self, allocation_scores: torch.Tensor, recv_mask: torch.Tensor) -> torch.Tensor:
        if float(self.trust_strength) <= 0.0:
            return allocation_scores
        topk = int(getattr(self.cfg, "trust_topk_k", 0))
        if topk <= 0:
            return allocation_scores
        scores = allocation_scores.squeeze(-1)
        valid = recv_mask.squeeze(-1) > 0.5
        if not bool(valid.any()):
            return allocation_scores
        masked = scores.masked_fill(~valid, -1e9)
        k = min(topk, masked.size(-1))
        if k <= 0:
            return allocation_scores
        _, indices = torch.topk(masked, k=k, dim=-1)
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(dim=-1, index=indices, value=1.0)
        hard_mask = hard_mask * valid.float()
        return hard_mask.unsqueeze(-1)

    def _blend_allocation(self, raw_allocation: torch.Tensor, recv_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.enable_trust:
            allocation = torch.ones_like(recv_mask)
            return allocation, recv_mask

        if bool(getattr(self.cfg, "trust_hard_topk", False)):
            hard_mask = self._apply_hard_topk(raw_allocation, recv_mask)
            allocation = (1.0 - float(self.trust_strength)) * recv_mask + float(self.trust_strength) * hard_mask
            aggregate_mask = recv_mask * allocation
            return allocation.clamp(0.0, 1.0), aggregate_mask.clamp(0.0, 1.0)

        allocation = (1.0 - float(self.trust_strength)) + float(self.trust_strength) * raw_allocation
        allocation = allocation * recv_mask
        return allocation.clamp(0.0, 1.0), allocation.clamp(0.0, 1.0)

    def _trust_loss(
        self,
        trust_logits: torch.Tensor,
        malicious_logits: torch.Tensor,
        utility_logits: torch.Tensor | None,
        neighbor_next_actions: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None = None,
        neighbor_malicious: torch.Tensor | None = None,
        utility_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self._zero_scalar(trust_logits)

        if neighbor_next_actions is not None:
            has_label = neighbor_next_actions.sum(dim=-1, keepdim=True) > 1e-6
            valid = has_label
            if neighbor_mask is not None:
                valid = valid & (neighbor_mask > 0.5)
            if bool(valid.any()):
                log_probs = F.log_softmax(trust_logits, dim=-1)
                ce = -(neighbor_next_actions * log_probs).sum(dim=-1, keepdim=True)
                ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
                denom = max(float(torch.log(torch.tensor(max(int(trust_logits.size(-1)), 2), device=trust_logits.device))), 1.0)
                ce = ce / denom
                loss = loss + (ce * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

        if neighbor_malicious is None:
            if utility_logits is not None and utility_targets is not None:
                util_mask = torch.ones_like(utility_logits, dtype=torch.bool)
                if neighbor_mask is not None:
                    util_mask = util_mask & (neighbor_mask > 0.5)
                if bool(util_mask.any()):
                    util_target = utility_targets.clamp(0.0, 1.0)
                    bce = F.binary_cross_entropy_with_logits(utility_logits, util_target, reduction="none")
                    w = float(max(0.0, getattr(self.cfg, "trust_counterfactual_weight", 1.0)))
                    loss = loss + w * (bce * util_mask.float()).sum() / util_mask.float().sum().clamp_min(1.0)
                    if bool((util_target[util_mask] > 0.5).any()) and bool((util_target[util_mask] <= 0.5).any()):
                        util_prob = torch.sigmoid(utility_logits)
                        pos_mean = util_prob[(util_target > 0.5) & util_mask].mean()
                        neg_mean = util_prob[(util_target <= 0.5) & util_mask].mean()
                        margin = float(max(0.0, getattr(self.cfg, "trust_counterfactual_margin", 0.02)))
                        loss = loss + w * F.relu(margin - (pos_mean - neg_mean))
            return loss

        mal_mask = torch.ones_like(malicious_logits, dtype=torch.bool)
        if neighbor_mask is not None:
            mal_mask = mal_mask & (neighbor_mask > 0.5)
        if not bool(mal_mask.any()):
            return loss

        target = neighbor_malicious.clamp(0.0, 1.0)
        pos = target[mal_mask].sum().clamp_min(1.0)
        neg = ((1.0 - target)[mal_mask]).sum().clamp_min(1.0)
        pos_weight = (neg / pos).detach()
        bce = F.binary_cross_entropy_with_logits(
            malicious_logits,
            target,
            reduction="none",
            pos_weight=pos_weight,
        )
        weight = float(getattr(self.cfg, "trust_malicious_weight", 1.0))
        loss = loss + weight * (bce * mal_mask.float()).sum() / mal_mask.float().sum().clamp_min(1.0)

        mal_bool = (target > 0.5) & mal_mask
        clean_bool = (~(target > 0.5)) & mal_mask
        if bool(mal_bool.any()) and bool(clean_bool.any()):
            malicious_prob = torch.sigmoid(malicious_logits)
            mal_mean = malicious_prob[mal_bool].mean()
            clean_mean = malicious_prob[clean_bool].mean()
            margin = float(max(0.0, getattr(self.cfg, "trust_margin", 0.1)))
            margin_weight = float(max(0.0, getattr(self.cfg, "trust_margin_weight", 0.5)))
            loss = loss + margin_weight * F.relu(margin - (mal_mean - clean_mean))

        if utility_logits is not None and utility_targets is not None:
            util_mask = mal_mask if neighbor_mask is None else (neighbor_mask > 0.5)
            util_target = utility_targets.clamp(0.0, 1.0)
            if bool(util_mask.any()):
                bce = F.binary_cross_entropy_with_logits(utility_logits, util_target, reduction="none")
                w = float(max(0.0, getattr(self.cfg, "trust_counterfactual_weight", 1.0)))
                loss = loss + w * (bce * util_mask.float()).sum() / util_mask.float().sum().clamp_min(1.0)
                if bool((util_target[util_mask] > 0.5).any()) and bool((util_target[util_mask] <= 0.5).any()):
                    util_prob = torch.sigmoid(utility_logits)
                    pos_mean = util_prob[(util_target > 0.5) & util_mask].mean()
                    neg_mean = util_prob[(util_target <= 0.5) & util_mask].mean()
                    margin = float(max(0.0, getattr(self.cfg, "trust_counterfactual_margin", 0.02)))
                    loss = loss + w * F.relu(margin - (pos_mean - neg_mean))
        return loss

    def _compute_counterfactual_utility(
        self,
        self_feat: torch.Tensor,
        recv_messages: torch.Tensor,
        base_trust_mask: torch.Tensor,
        base_comm_mask: torch.Tensor,
        recv_mask: torch.Tensor,
        recv_feat: torch.Tensor,
        avail_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        if recv_messages.size(1) == 0:
            return torch.zeros(self_feat.size(0), 0, 1, device=self_feat.device, dtype=self_feat.dtype)
        utility_scores = []
        base_msg = recv_messages
        base_feat = recv_feat
        base_comm = self.vib_gat.aggregate_messages(
            self_feat,
            base_msg,
            base_trust_mask,
            base_comm_mask,
            recv_features=base_feat,
        )
        base_policy_input = self.residual(self_feat, base_comm, self.comm_enabled, self.comm_strength)
        base_logits = self._mask_logits(self.policy_head(base_policy_input), avail_actions)
        base_logp = torch.log_softmax(base_logits, dim=-1)
        base_prob = torch.softmax(base_logits, dim=-1)

        for i in range(base_msg.size(1)):
            edge_alive = base_comm_mask[:, i, 0]
            msg_i = base_msg.clone()
            msg_i[:, i : i + 1, :] = 0.0
            trust_mask_i = base_trust_mask.clone()
            trust_mask_i[:, i : i + 1, :] = 0.0
            comm_mask_i = base_comm_mask.clone()
            comm_mask_i[:, i : i + 1, :] = 0.0
            feat_i = self.vib_gat.decode_messages(msg_i)
            comm_i = self.vib_gat.aggregate_messages(
                self_feat,
                msg_i,
                trust_mask_i,
                comm_mask_i,
                recv_features=feat_i,
            )
            policy_i = self.residual(self_feat, comm_i, self.comm_enabled, self.comm_strength)
            logits_i = self._mask_logits(self.policy_head(policy_i), avail_actions)
            logp_i = torch.log_softmax(logits_i, dim=-1)
            utility = (base_prob * (base_logp - logp_i)).sum(dim=-1).abs()
            utility_scores.append(utility * edge_alive)
        if not utility_scores:
            return torch.zeros(self_feat.size(0), 0, 1, device=self_feat.device, dtype=self_feat.dtype)
        utility_scores = torch.stack(utility_scores, dim=1)
        utility_scores = utility_scores / utility_scores.max(dim=1, keepdim=True).values.clamp_min(1e-6)
        utility_scores = utility_scores.unsqueeze(-1) * recv_mask
        utility_scores = torch.nan_to_num(utility_scores, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return utility_scores

    def _comm_forward(
        self,
        self_feat: torch.Tensor,
        neighbor_seq: torch.Tensor,
        neighbor_mask: torch.Tensor | None,
        channel_mask: torch.Tensor | None,
        channel_noise: torch.Tensor | None,
        neighbor_next_actions: torch.Tensor | None,
        neighbor_malicious: torch.Tensor | None,
        avail_actions: torch.Tensor | None,
        *,
        deterministic: bool,
        training: bool,
    ) -> Dict[str, torch.Tensor]:
        if (not self.comm_enabled) or (self.comm_strength <= 0.0):
            zero = self._zero_scalar(self_feat)
            return {
                "comm_feat": torch.zeros_like(self_feat),
                "kl_loss": zero,
                "kl_raw": zero,
                "trust_loss": zero,
                "trust_score_mean": zero,
                "trust_score_p10": zero,
                "trust_score_p50": zero,
                "trust_score_p90": zero,
                "trust_gate_ratio": zero,
                "comm_valid_neighbors": zero,
                "comm_kept_neighbors": zero,
                "comm_malicious_ratio": zero,
                "trust_malicious_gap": zero,
                "malicious_score_mean": zero,
                "malicious_score_mal": zero,
                "malicious_score_clean": zero,
                "allocation_score_mean": zero,
                "allocation_score_mal": zero,
                "allocation_score_clean": zero,
                "allocation_malicious_gap": zero,
                "utility_score_mean": zero,
                "utility_score_mal": zero,
                "utility_score_clean": zero,
                "utility_malicious_gap": zero,
            }

        neighbor_feat = self._encode_neighbors(neighbor_seq)
        if neighbor_mask is None:
            neighbor_mask = torch.ones(neighbor_feat.size(0), neighbor_feat.size(1), 1, device=neighbor_feat.device, dtype=neighbor_feat.dtype)
        sender_mask = torch.nan_to_num(neighbor_mask.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if channel_mask is None:
            recv_mask = sender_mask
        else:
            recv_mask = torch.nan_to_num(channel_mask.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            recv_mask = recv_mask * sender_mask

        sender_messages, kl_loss, kl_raw, sender_uncertainty = self.vib_gat.encode_messages(
            neighbor_feat,
            sender_mask,
            deterministic=deterministic,
        )
        if channel_noise is None:
            recv_messages = sender_messages
        else:
            recv_messages = sender_messages + torch.nan_to_num(channel_noise, nan=0.0, posinf=0.0, neginf=0.0)
        recv_messages = recv_messages * recv_mask
        recv_feat = self.vib_gat.decode_messages(recv_messages)

        trust_logits, malicious_logits, utility_logits, reliability, utility_score = self._compute_utility(
            recv_feat,
            sender_uncertainty,
        )
        raw_allocation = self._compute_allocation(reliability)
        utility_bias = float(max(0.0, getattr(self.cfg, "trust_counterfactual_mix", 0.5)))
        if utility_score.numel() > 0:
            raw_allocation = (1.0 - utility_bias) * raw_allocation + utility_bias * utility_score
        raw_allocation = raw_allocation * recv_mask
        allocation_scores, aggregate_mask = self._blend_allocation(raw_allocation, recv_mask)
        attn_trust_mask = allocation_scores * recv_mask + (1e-6 * (1.0 - recv_mask))

        utility_targets = None
        if training and self.cfg.enable_trust:
            with torch.no_grad():
                utility_targets = self._compute_counterfactual_utility(
                    self_feat,
                    recv_messages,
                    recv_mask,
                    recv_mask,
                    recv_mask,
                    recv_feat,
                    avail_actions,
                )

        comm_feat = self.vib_gat.aggregate_messages(
            self_feat,
            recv_messages,
            attn_trust_mask,
            aggregate_mask,
            recv_features=recv_feat,
        )
        comm_feat = self.comm_dropout(comm_feat)

        kl_loss = torch.nan_to_num(kl_loss, nan=0.0, posinf=0.0, neginf=0.0)
        kl_raw = torch.nan_to_num(kl_raw, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.cfg.enable_kl:
            kl_loss = self._zero_scalar(self_feat)
        elif training:
            kl_loss = kl_loss * float(self.comm_strength)

        trust_loss = self._zero_scalar(self_feat)
        if training and self.cfg.enable_trust:
            trust_loss = self._trust_loss(
                trust_logits,
                malicious_logits,
                utility_logits,
                neighbor_next_actions,
                neighbor_mask=recv_mask,
                neighbor_malicious=neighbor_malicious,
                utility_targets=utility_targets,
            )

        valid_mask = (recv_mask > 0.5).squeeze(-1)
        sender_valid_mask = (sender_mask > 0.5).squeeze(-1)
        utility_values = reliability.squeeze(-1)[valid_mask]

        zero = self._zero_scalar(self_feat)
        if utility_values.numel() == 0:
            trust_score_mean = zero
            trust_score_p10 = zero
            trust_score_p50 = zero
            trust_score_p90 = zero
        else:
            trust_score_mean = utility_values.mean()
            q = torch.tensor([0.1, 0.5, 0.9], device=utility_values.device, dtype=utility_values.dtype)
            qv = torch.quantile(utility_values, q)
            trust_score_p10 = qv[0]
            trust_score_p50 = qv[1]
            trust_score_p90 = qv[2]

        comm_valid_neighbors = sender_valid_mask.float().sum(dim=-1).mean()
        comm_kept_neighbors = aggregate_mask.squeeze(-1).sum(dim=-1).mean()
        trust_gate_ratio = comm_kept_neighbors / recv_mask.squeeze(-1).sum(dim=-1).mean().clamp_min(1.0)

        malicious_prob = torch.sigmoid(malicious_logits)
        malicious_prob = torch.nan_to_num(malicious_prob, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        comm_malicious_ratio = zero
        trust_malicious_gap = zero
        malicious_score_mean = zero
        malicious_score_mal = zero
        malicious_score_clean = zero
        allocation_score_mean = zero
        allocation_score_mal = zero
        allocation_score_clean = zero
        allocation_malicious_gap = zero
        utility_score_mean = zero
        utility_score_mal = zero
        utility_score_clean = zero
        utility_malicious_gap = zero

        if utility_values.numel() > 0:
            allocation_score_mean = aggregate_mask.squeeze(-1)[valid_mask].mean()
            malicious_score_mean = malicious_prob.squeeze(-1)[valid_mask].mean()
            utility_score_mean = utility_score.squeeze(-1)[valid_mask].mean()

        if neighbor_malicious is not None:
            mal_mask = (neighbor_malicious > 0.5).squeeze(-1) & valid_mask
            if bool(valid_mask.any()):
                comm_malicious_ratio = mal_mask.float().sum() / valid_mask.float().sum().clamp_min(1.0)
            clean_mask = valid_mask & (~mal_mask)
            if bool(mal_mask.any()):
                malicious_score_mal = malicious_prob.squeeze(-1)[mal_mask].mean()
                allocation_score_mal = aggregate_mask.squeeze(-1)[mal_mask].mean()
                utility_score_mal = utility_score.squeeze(-1)[mal_mask].mean()
                trust_mal = reliability.squeeze(-1)[mal_mask].mean()
            else:
                trust_mal = zero
            if bool(clean_mask.any()):
                malicious_score_clean = malicious_prob.squeeze(-1)[clean_mask].mean()
                allocation_score_clean = aggregate_mask.squeeze(-1)[clean_mask].mean()
                utility_score_clean = utility_score.squeeze(-1)[clean_mask].mean()
                trust_clean = reliability.squeeze(-1)[clean_mask].mean()
            else:
                trust_clean = zero
            if bool(mal_mask.any()) and bool(clean_mask.any()):
                trust_malicious_gap = trust_clean - trust_mal
                allocation_malicious_gap = allocation_score_clean - allocation_score_mal
                utility_malicious_gap = utility_score_clean - utility_score_mal

        return {
            "comm_feat": comm_feat,
            "kl_loss": kl_loss,
            "kl_raw": kl_raw,
            "trust_loss": trust_loss,
            "trust_score_mean": trust_score_mean,
            "trust_score_p10": trust_score_p10,
            "trust_score_p50": trust_score_p50,
            "trust_score_p90": trust_score_p90,
            "trust_gate_ratio": trust_gate_ratio,
            "comm_valid_neighbors": comm_valid_neighbors,
            "comm_kept_neighbors": comm_kept_neighbors,
            "comm_malicious_ratio": comm_malicious_ratio,
            "trust_malicious_gap": trust_malicious_gap,
            "malicious_score_mean": malicious_score_mean,
            "malicious_score_mal": malicious_score_mal,
            "malicious_score_clean": malicious_score_clean,
            "allocation_score_mean": allocation_score_mean,
            "allocation_score_mal": allocation_score_mal,
            "allocation_score_clean": allocation_score_clean,
            "allocation_malicious_gap": allocation_malicious_gap,
            "utility_score_mean": utility_score_mean,
            "utility_score_mal": utility_score_mal,
            "utility_score_clean": utility_score_clean,
            "utility_malicious_gap": utility_malicious_gap,
        }

    def _critic_forward(
        self,
        state: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        critic_feat, next_critic = self.critic_encoder(state.unsqueeze(1), rnn_states_critic.unsqueeze(0), masks)
        critic_feat = self.critic_mlp(critic_feat)
        values = self.value_head(critic_feat)
        return values, next_critic.squeeze(0)

    def act(
        self,
        obs_seq: torch.Tensor,
        state: torch.Tensor,
        neighbor_seq: torch.Tensor,
        neighbor_mask: torch.Tensor | None,
        channel_mask: torch.Tensor | None,
        channel_noise: torch.Tensor | None,
        neighbor_next_actions: torch.Tensor | None,
        neighbor_malicious: torch.Tensor | None,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
        avail_actions: torch.Tensor | None,
        deterministic: bool = False,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self_feat, next_actor = self.actor_encoder(obs_seq, rnn_states_actor.unsqueeze(0), masks)
        self_feat = torch.nan_to_num(self_feat, nan=0.0, posinf=0.0, neginf=0.0)
        comm_out = self._comm_forward(
            self_feat,
            neighbor_seq,
            neighbor_mask,
            channel_mask,
            channel_noise,
            neighbor_next_actions,
            neighbor_malicious=neighbor_malicious,
            avail_actions=avail_actions,
            deterministic=bool(deterministic) or bool(self.cfg.vib_deterministic),
            training=False,
        )

        fused = self.residual(self_feat, comm_out["comm_feat"], self.comm_enabled, self.comm_strength)
        logits = self._mask_logits(self.policy_head(fused), avail_actions)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            actions = dist.sample().unsqueeze(-1)
        log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        values, next_critic = self._critic_forward(state, rnn_states_critic, masks)

        out = {
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "entropy": entropy,
            "next_actor_state": next_actor.squeeze(0),
            "next_critic_state": next_critic,
            "kl_loss": comm_out["kl_loss"],
            "kl_raw": comm_out["kl_raw"],
        }
        if return_debug:
            out.update(comm_out)
        return out

    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        state: torch.Tensor,
        neighbor_seq: torch.Tensor,
        neighbor_mask: torch.Tensor | None,
        channel_mask: torch.Tensor | None,
        channel_noise: torch.Tensor | None,
        neighbor_next_actions: torch.Tensor | None,
        neighbor_malicious: torch.Tensor | None,
        actions: torch.Tensor,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
        avail_actions: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        self_feat, next_actor = self.actor_encoder(obs_seq, rnn_states_actor.unsqueeze(0), masks)
        self_feat = torch.nan_to_num(self_feat, nan=0.0, posinf=0.0, neginf=0.0)
        comm_out = self._comm_forward(
            self_feat,
            neighbor_seq,
            neighbor_mask,
            channel_mask,
            channel_noise,
            neighbor_next_actions,
            neighbor_malicious,
            avail_actions=avail_actions,
            deterministic=bool(self.cfg.vib_deterministic),
            training=True,
        )

        fused = self.residual(self_feat, comm_out["comm_feat"], self.comm_enabled, self.comm_strength)
        logits = self._mask_logits(self.policy_head(fused), avail_actions)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        values, next_critic = self._critic_forward(state, rnn_states_critic, masks)

        if (not self.comm_enabled) or (self.comm_strength <= 0.0):
            residual_gate_mean = self._zero_scalar(self_feat)
            residual_gate_max = self._zero_scalar(self_feat)
            residual_comm_ratio = self._zero_scalar(self_feat)
        else:
            with torch.no_grad():
                strength = torch.as_tensor(float(self.comm_strength), device=self_feat.device, dtype=self_feat.dtype).clamp(0.0, 1.0)
                gate_input = torch.cat([self_feat, comm_out["comm_feat"]], dim=-1)
                gate = torch.sigmoid(self.residual.gate(gate_input)) * strength
                residual_gate_mean = gate.mean()
                residual_gate_max = gate.max()
                comm_contrib = (gate * comm_out["comm_feat"]).norm(dim=-1)
                self_norm = self_feat.norm(dim=-1).clamp_min(1e-6)
                residual_comm_ratio = (comm_contrib / self_norm).mean()

        out = {
            "log_probs": log_probs,
            "entropy": entropy,
            "values": values,
            "next_actor_state": next_actor.squeeze(0),
            "next_critic_state": next_critic,
            "comm_strength": torch.tensor(float(self.comm_strength), device=obs_seq.device, dtype=obs_seq.dtype),
            "trust_strength": torch.tensor(float(self.trust_strength), device=obs_seq.device, dtype=obs_seq.dtype),
            "comm_enabled": torch.tensor(float(self.comm_enabled), device=obs_seq.device, dtype=obs_seq.dtype),
            "residual_gate_mean": residual_gate_mean,
            "residual_gate_max": residual_gate_max,
            "residual_comm_ratio": residual_comm_ratio,
        }
        out.update(comm_out)
        return out

    def get_values(
        self,
        state: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._critic_forward(state, rnn_states_critic, masks)
