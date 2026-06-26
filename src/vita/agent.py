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
    vib_consistency_weight: float = 0.0
    vib_consistency_noise_std: float = 0.0
    trust_lambda: float = 0.1
    trust_malicious_weight: float = 1.0
    trust_margin_weight: float = 0.5
    trust_margin: float = 0.1
    trust_action_loss_weight: float = 1.0
    trust_reliability_mix: float = 0.5
    trust_utility_mix: float = 0.6
    trust_counterfactual_mix: float = 0.5
    trust_counterfactual_margin: float = 0.02
    trust_counterfactual_weight: float = 1.0
    trust_consistency_mix: float = 0.0
    trust_consistency_weight: float = 0.0
    trust_consistency_margin: float = 0.05
    trust_consistency_noise_std: float = 0.0
    max_neighbors: int = 5
    comm_dropout: float = 0.1
    enable_trust: bool = True
    enable_kl: bool = True
    bypass_vib: bool = False
    vib_deterministic: bool = False
    attention_only: bool = False
    trust_hard_topk: bool = False
    trust_topk_k: int = 0
    trust_gate_floor: float = 0.0
    attn_bias_coef: float = 1.0
    trust_malicious_gate_coef: float = 1.0
    allocation_sharpness: float = 1.0
    allocation_floor: float = 0.0
    trust_pair_product: bool = False
    trust_gate_product: bool = False
    trust_decouple_allocation: bool = False
    trust_use_utility_for_gate: bool = True
    trust_gate_threshold: float = 0.0
    enable_belief_router: bool = False
    belief_router_tau: float = 3.0
    belief_router_strength: float = 1.0
    belief_router_self_floor: float = 0.1
    belief_router_prior_weight: float = 0.5
    belief_router_social_weight: float = 1.0
    belief_router_comm_quantile: float = 0.1
    belief_router_social_cap: float = 0.6
    belief_prior_loss_weight: float = 0.0
    belief_prior_loss_min_conf: float = 0.05


class VITAAgent(torch.nn.Module):
    def __init__(self, cfg: VITAAgentConfig):
        super().__init__()
        self.cfg = cfg
        self.actor_encoder = FeatureEncoder(cfg.obs_dim, cfg.hidden_dim)
        self.critic_encoder = FeatureEncoder(cfg.state_dim, cfg.hidden_dim)
        self.comm_encoder = FeatureEncoder(cfg.obs_dim, cfg.hidden_dim)
        self.trust_predictor = TrustPredictor(
            cfg.hidden_dim,
            cfg.action_dim,
            cfg.trust_gamma,
            pair_product=cfg.trust_pair_product,
        )
        message_dim = cfg.hidden_dim if bool(cfg.bypass_vib) else cfg.latent_dim
        self.vib_gat = VIBGATLayer(
            cfg.hidden_dim,
            message_dim,
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
        self.prior_norm = torch.nn.LayerNorm(cfg.hidden_dim)
        self.prior_predictor = torch.nn.Sequential(
            torch.nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        torch.nn.init.zeros_(self.prior_predictor[-1].weight)
        torch.nn.init.zeros_(self.prior_predictor[-1].bias)
        self.comm_enabled = True
        self.comm_strength = 0.0
        self.trust_strength = 0.0

    def load_state_dict(self, state_dict, strict: bool = True):
        has_trust = any(key.startswith("trust_predictor.") for key in state_dict)
        has_utility_head = any(key.startswith("trust_predictor.utility_head.") for key in state_dict)
        has_receiver_ctx = any(key.startswith("trust_predictor.receiver_context_net.") for key in state_dict)
        has_pair_fusion = any(key.startswith("trust_predictor.pair_fusion.") for key in state_dict)
        if (not has_trust) or (not has_utility_head) or (not has_receiver_ctx) or (not has_pair_fusion):
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

    def _zero_belief_debug(self, ref: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = self._zero_scalar(ref)
        return {
            "belief_prior_loss": zero,
            "belief_error": zero,
            "belief_self_conf": zero,
            "belief_comm_conf": zero,
            "route_w_self": zero,
            "route_w_social": zero,
            "route_w_prior": zero,
        }

    def _encode_neighbors(self, neighbor_seq: torch.Tensor) -> torch.Tensor:
        bsz, k, t, dim = neighbor_seq.shape
        flat = neighbor_seq.view(bsz * k, t, dim)
        feat, _ = self.comm_encoder(flat, None, None)
        feat = self.neighbor_norm(feat)
        return feat.view(bsz, k, -1)

    def _predict_prior(
        self,
        rnn_states_actor: torch.Tensor,
        masks: torch.Tensor | None,
    ) -> torch.Tensor:
        if masks is None:
            mask = torch.ones(
                rnn_states_actor.size(0),
                1,
                device=rnn_states_actor.device,
                dtype=rnn_states_actor.dtype,
            )
        else:
            mask = torch.nan_to_num(masks, nan=0.0, posinf=1.0, neginf=0.0).view(rnn_states_actor.size(0), -1)
            mask = mask[:, :1].clamp(0.0, 1.0)
        prior_input = rnn_states_actor * mask
        delta = self.prior_predictor(self.prior_norm(prior_input))
        return prior_input + delta

    def _compute_belief_prior(
        self,
        self_feat: torch.Tensor,
        rnn_states_actor: torch.Tensor,
        masks: torch.Tensor | None,
        *,
        training: bool,
    ) -> Dict[str, torch.Tensor]:
        if masks is None:
            mask = torch.ones(
                rnn_states_actor.size(0),
                1,
                device=rnn_states_actor.device,
                dtype=rnn_states_actor.dtype,
            )
        else:
            mask = torch.nan_to_num(masks, nan=0.0, posinf=1.0, neginf=0.0).view(rnn_states_actor.size(0), -1)
            mask = mask[:, :1].clamp(0.0, 1.0)
        prior_input = rnn_states_actor * mask
        prior_available = prior_input.norm(dim=-1, keepdim=True) > 1e-6

        prior_feat = self._predict_prior(rnn_states_actor, masks)
        self_norm = F.normalize(torch.nan_to_num(self_feat), p=2, dim=-1)
        prior_norm = F.normalize(torch.nan_to_num(prior_feat), p=2, dim=-1)
        cosine = (self_norm * prior_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        error = (0.5 * (1.0 - cosine)).clamp(0.0, 1.0)

        reset_mask = mask <= 0.5
        no_prior_mask = reset_mask | (~prior_available)
        error = torch.where(no_prior_mask, torch.zeros_like(error), error)

        tau = float(max(1e-6, getattr(self.cfg, "belief_router_tau", 3.0)))
        self_conf = torch.exp(-tau * error).clamp(0.0, 1.0)
        self_conf = torch.where(no_prior_mask, torch.ones_like(self_conf), self_conf)
        context_conf = self_conf.detach()
        receiver_context = context_conf * self_feat + (1.0 - context_conf) * prior_feat

        prior_loss = self._zero_scalar(self_feat)
        loss_weight = float(max(0.0, getattr(self.cfg, "belief_prior_loss_weight", 0.0)))
        if training and loss_weight > 0.0 and bool(getattr(self.cfg, "enable_belief_router", False)):
            min_conf = float(max(0.0, min(1.0, getattr(self.cfg, "belief_prior_loss_min_conf", 0.05))))
            available_weight = prior_available.float().detach()
            weight = self_conf.detach().clamp(min=min_conf, max=1.0) * available_weight
            mse = (prior_norm - self_norm.detach()).pow(2).mean(dim=-1, keepdim=True)
            denom = weight.sum().clamp_min(1.0)
            prior_loss = loss_weight * (weight * mse).sum() / denom

        return {
            "prior_feat": prior_feat,
            "receiver_context": receiver_context,
            "self_conf": self_conf,
            "belief_error": error,
            "belief_prior_loss": prior_loss,
        }

    def _belief_route(
        self,
        self_feat: torch.Tensor,
        comm_feat: torch.Tensor,
        comm_out: Dict[str, torch.Tensor],
        belief_out: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not bool(getattr(self.cfg, "enable_belief_router", False)):
            fused = self.residual(self_feat, comm_feat, self.comm_enabled, self.comm_strength)
            debug = self._zero_belief_debug(self_feat)
            debug["belief_prior_loss"] = belief_out.get("belief_prior_loss", self._zero_scalar(self_feat))
            return fused, debug

        prior_feat = belief_out["prior_feat"]
        receiver_context = belief_out["receiver_context"]
        self_conf = belief_out["self_conf"]
        belief_error = belief_out["belief_error"]

        social_feat = self.residual(receiver_context, comm_feat, self.comm_enabled, self.comm_strength)
        edge_reliability = comm_out.get("edge_reliability")
        edge_malicious_prob = comm_out.get("edge_malicious_prob")
        edge_consistency = comm_out.get("edge_consistency")
        edge_valid = comm_out.get("edge_valid_mask")
        if edge_reliability is None or edge_valid is None:
            comm_conf = torch.zeros_like(self_conf)
        else:
            valid = torch.nan_to_num(edge_valid, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
            reliability = torch.nan_to_num(edge_reliability, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            denom = valid.sum(dim=1).clamp_min(1.0)
            mean_reliability = (reliability * valid).sum(dim=1) / denom
            valid_bool = valid.squeeze(-1) > 0.5
            masked_reliability = reliability.squeeze(-1).masked_fill(~valid_bool, 2.0)
            q = float(max(0.0, min(1.0, getattr(self.cfg, "belief_router_comm_quantile", 0.1))))
            sorted_reliability, _ = torch.sort(masked_reliability, dim=1)
            valid_count = valid_bool.sum(dim=1, keepdim=True)
            q_index = torch.floor(q * (valid_count.float() - 1.0).clamp_min(0.0)).long()
            low_reliability = sorted_reliability.gather(1, q_index).clamp(0.0, 1.0)
            low_reliability = torch.where(valid_count > 0, low_reliability, torch.zeros_like(low_reliability))
            comm_conf = 0.5 * mean_reliability + 0.5 * low_reliability
            if edge_malicious_prob is not None:
                malicious_prob = torch.nan_to_num(edge_malicious_prob, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                clean_prob = (1.0 - malicious_prob).clamp(0.0, 1.0)
                clean_conf = (clean_prob * valid).sum(dim=1) / denom
                comm_conf = comm_conf * clean_conf
            if edge_consistency is not None:
                consistency = torch.nan_to_num(edge_consistency, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                consistency_conf = (consistency * valid).sum(dim=1) / denom
                comm_conf = comm_conf * consistency_conf
            has_valid = (valid.sum(dim=1) > 0.5).float()
            comm_conf = comm_conf * has_valid
        if (not self.comm_enabled) or float(self.comm_strength) <= 0.0:
            comm_conf = torch.zeros_like(self_conf)
        else:
            comm_conf = comm_conf * float(self.comm_strength)
        comm_conf = comm_conf.clamp(0.0, 1.0)

        strength = float(max(0.0, min(1.0, getattr(self.cfg, "belief_router_strength", 1.0))))
        self_floor = float(max(0.0, min(1.0, getattr(self.cfg, "belief_router_self_floor", 0.1))))
        prior_weight = float(max(0.0, getattr(self.cfg, "belief_router_prior_weight", 0.5)))
        social_weight = float(max(0.0, getattr(self.cfg, "belief_router_social_weight", 1.0)))
        social_cap = float(max(0.0, min(1.0, getattr(self.cfg, "belief_router_social_cap", 0.6))))

        route_self_conf = self_conf.detach()
        route_comm_conf = comm_conf.detach()
        uncertainty = (1.0 - route_self_conf).clamp(0.0, 1.0)
        self_score = self_floor + (1.0 - self_floor) * route_self_conf
        social_score = social_weight * uncertainty * route_comm_conf
        prior_score = prior_weight * uncertainty * (1.0 - route_comm_conf)
        scores = torch.cat([self_score, social_score, prior_score], dim=-1).clamp_min(1e-6)
        weights = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        w_self = weights[:, 0:1]
        w_social = weights[:, 1:2]
        w_prior = weights[:, 2:3]
        if social_cap < 1.0:
            capped_social = w_social.clamp(max=social_cap)
            overflow = w_social - capped_social
            non_social = (w_self + w_prior).clamp_min(1e-6)
            w_self = w_self + overflow * (w_self / non_social)
            w_prior = w_prior + overflow * (w_prior / non_social)
            w_social = capped_social
            route_norm = (w_self + w_social + w_prior).clamp_min(1e-6)
            w_self = w_self / route_norm
            w_social = w_social / route_norm
            w_prior = w_prior / route_norm

        routed = w_self * self_feat + w_social * social_feat + w_prior * prior_feat
        default_fused = self.residual(self_feat, comm_feat, self.comm_enabled, self.comm_strength)
        fused = (1.0 - strength) * default_fused + strength * routed
        debug = {
            "belief_prior_loss": belief_out.get("belief_prior_loss", self._zero_scalar(self_feat)),
            "belief_error": belief_error.mean(),
            "belief_self_conf": self_conf.mean(),
            "belief_comm_conf": comm_conf.mean(),
            "route_w_self": w_self.mean(),
            "route_w_social": w_social.mean(),
            "route_w_prior": w_prior.mean(),
        }
        return fused, debug

    def _vib_consistency_loss(
        self,
        neighbor_seq: torch.Tensor,
        sender_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = self._zero_scalar(neighbor_seq)
        weight = float(max(0.0, getattr(self.cfg, "vib_consistency_weight", 0.0)))
        noise_std = float(max(0.0, getattr(self.cfg, "vib_consistency_noise_std", 0.0)))
        if (
            weight <= 0.0
            or noise_std <= 0.0
            or (not self.cfg.enable_kl)
            or bool(self.cfg.bypass_vib)
            or bool(self.cfg.attention_only)
            or neighbor_seq.numel() == 0
        ):
            return zero, zero

        edge_mask = torch.nan_to_num(sender_mask.squeeze(-1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if not bool((edge_mask > 0.5).any()):
            return zero, zero

        noise_mask = edge_mask.view(edge_mask.size(0), edge_mask.size(1), 1, 1)
        view1 = neighbor_seq + torch.randn_like(neighbor_seq) * noise_std * noise_mask
        view2 = neighbor_seq + torch.randn_like(neighbor_seq) * noise_std * noise_mask
        mu1 = self.vib_gat.encode_mean_latent(self._encode_neighbors(view1))
        mu2 = self.vib_gat.encode_mean_latent(self._encode_neighbors(view2))
        mu1 = F.normalize(mu1, p=2, dim=-1)
        mu2 = F.normalize(mu2, p=2, dim=-1)
        per_edge = (mu1 - mu2).pow(2).mean(dim=-1)
        raw_loss = (per_edge * edge_mask).sum() / edge_mask.sum().clamp_min(1.0)
        scaled_loss = raw_loss * weight * float(self.comm_strength)
        return scaled_loss, raw_loss.detach()

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

    def _compute_consistency_score(
        self,
        self_feat: torch.Tensor,
        recv_feat: torch.Tensor,
        neighbor_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        noise_std = float(max(0.0, getattr(self.cfg, "trust_consistency_noise_std", 0.0)))
        if noise_std <= 0.0:
            return torch.ones(
                recv_feat.size(0),
                recv_feat.size(1),
                1,
                device=recv_feat.device,
                dtype=recv_feat.dtype,
            )

        direction = torch.tanh(recv_feat)
        direction_scale = direction.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = noise_std * direction / direction_scale

        logits_pos, _, malicious_pos, _ = self.trust_predictor(
            recv_feat + delta,
            neighbor_uncertainty,
            receiver_context=self_feat,
        )
        logits_neg, _, malicious_neg, _ = self.trust_predictor(
            recv_feat - delta,
            neighbor_uncertainty,
            receiver_context=self_feat,
        )

        logp_pos = F.log_softmax(logits_pos, dim=-1)
        logp_neg = F.log_softmax(logits_neg, dim=-1)
        prob_pos = logp_pos.exp()
        prob_neg = logp_neg.exp()
        mean_prob = (0.5 * (prob_pos + prob_neg)).clamp_min(1e-6)
        log_mean = mean_prob.log()
        js = 0.5 * (
            (prob_pos * (logp_pos - log_mean)).sum(dim=-1, keepdim=True)
            + (prob_neg * (logp_neg - log_mean)).sum(dim=-1, keepdim=True)
        )
        denom = max(
            float(torch.log(torch.tensor(max(int(logits_pos.size(-1)), 2), device=recv_feat.device))),
            1.0,
        )
        js = js / denom

        malicious_gap = (torch.sigmoid(malicious_pos) - torch.sigmoid(malicious_neg)).abs()
        instability = 0.5 * js + 0.5 * malicious_gap
        return (1.0 - instability).clamp(1e-6, 1.0)

    def _compute_utility(
        self,
        self_feat: torch.Tensor,
        recv_feat: torch.Tensor,
        neighbor_uncertainty: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trust_logits, confidence_trust, malicious_logits, utility_logits = self.trust_predictor(
            recv_feat,
            neighbor_uncertainty,
            receiver_context=self_feat,
        )
        malicious_prob = torch.sigmoid(malicious_logits).clamp(0.0, 1.0)
        confidence_trust = confidence_trust.clamp(1e-6, 1.0)
        mix = float(max(0.0, min(1.0, getattr(self.cfg, "trust_reliability_mix", 0.5))))
        malicious_gate_coef = float(max(0.0, getattr(self.cfg, "trust_malicious_gate_coef", 1.0)))
        malicious_gate = (1.0 - malicious_gate_coef * malicious_prob).clamp(0.0, 1.0)
        utility_score = torch.sigmoid(utility_logits).clamp(0.0, 1.0)
        base_reliability = ((1.0 - mix) * confidence_trust + mix * malicious_gate).clamp(1e-6, 1.0)

        consistency_mix = float(max(0.0, min(1.0, getattr(self.cfg, "trust_consistency_mix", 0.0))))
        consistency_score = self._compute_consistency_score(self_feat, recv_feat, neighbor_uncertainty)
        gate_reliability = ((1.0 - consistency_mix) * base_reliability + consistency_mix * consistency_score).clamp(
            1e-6, 1.0
        )

        if bool(getattr(self.cfg, "trust_use_utility_for_gate", True)):
            utility_mix = float(max(0.0, min(1.0, getattr(self.cfg, "trust_utility_mix", 0.6))))
            if bool(getattr(self.cfg, "trust_gate_product", False)):
                reliability = gate_reliability * ((1.0 - utility_mix) + utility_mix * utility_score)
            else:
                reliability = (1.0 - utility_mix) * gate_reliability + utility_mix * utility_score
        else:
            reliability = gate_reliability
        reliability = reliability.clamp(1e-6, 1.0)
        return trust_logits, malicious_logits, utility_logits, reliability, utility_score, consistency_score

    def _compute_allocation(self, utility_scores: torch.Tensor) -> torch.Tensor:
        utility_scores = utility_scores.clamp(1e-6, 1.0)
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
        valid = recv_mask.squeeze(-1) > 1e-6
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

    def _apply_gate_threshold(self, gate_scores: torch.Tensor) -> torch.Tensor:
        threshold = float(max(0.0, min(0.999, getattr(self.cfg, "trust_gate_threshold", 0.0))))
        if threshold <= 0.0:
            return gate_scores.clamp(0.0, 1.0)
        denom = max(1e-6, 1.0 - threshold)
        return ((gate_scores - threshold) / denom).clamp(0.0, 1.0)

    def _blend_allocation(
        self,
        raw_allocation: torch.Tensor,
        recv_mask: torch.Tensor,
        gate_scores: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.enable_trust:
            allocation = torch.ones_like(recv_mask)
            return allocation, recv_mask

        if gate_scores is None:
            gate_scores = recv_mask
        gate_scores = self._apply_gate_threshold(gate_scores) * recv_mask

        if bool(getattr(self.cfg, "trust_hard_topk", False)):
            hard_mask = self._apply_hard_topk(raw_allocation * gate_scores, gate_scores)
            trust_branch = gate_scores * hard_mask
            allocation = (1.0 - float(self.trust_strength)) * recv_mask + float(self.trust_strength) * trust_branch
            aggregate_mask = recv_mask * allocation
            return allocation.clamp(0.0, 1.0), aggregate_mask.clamp(0.0, 1.0)

        trust_branch = gate_scores * raw_allocation
        allocation = (1.0 - float(self.trust_strength)) * recv_mask + float(self.trust_strength) * trust_branch
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
        consistency_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self._zero_scalar(trust_logits)

        action_weight = float(max(0.0, getattr(self.cfg, "trust_action_loss_weight", 1.0)))
        if neighbor_next_actions is not None and action_weight > 0.0:
            has_label = neighbor_next_actions.sum(dim=-1, keepdim=True) > 1e-6
            valid = has_label
            if neighbor_mask is not None:
                valid = valid & (neighbor_mask > 0.5)
            if bool(valid.any()):
                log_probs = F.log_softmax(trust_logits, dim=-1)
                ce = -(neighbor_next_actions * log_probs).sum(dim=-1, keepdim=True)
                denom = max(float(torch.log(torch.tensor(max(int(trust_logits.size(-1)), 2), device=trust_logits.device))), 1.0)
                ce = ce / denom
                loss = loss + action_weight * (ce * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

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

        consistency_weight = float(max(0.0, getattr(self.cfg, "trust_consistency_weight", 0.0)))
        if consistency_scores is not None and consistency_weight > 0.0:
            consistency_target = (1.0 - target).clamp(0.0, 1.0)
            mse = F.mse_loss(consistency_scores, consistency_target, reduction="none")
            loss = loss + consistency_weight * (mse * mal_mask.float()).sum() / mal_mask.float().sum().clamp_min(1.0)
            if bool(mal_bool.any()) and bool(clean_bool.any()):
                clean_consistency = consistency_scores[clean_bool].mean()
                mal_consistency = consistency_scores[mal_bool].mean()
                margin = float(max(0.0, getattr(self.cfg, "trust_consistency_margin", 0.05)))
                loss = loss + consistency_weight * F.relu(margin - (clean_consistency - mal_consistency))

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

        B, K, D = base_msg.shape
        eye = torch.eye(K, device=base_msg.device, dtype=base_msg.dtype).view(1, K, K, 1)
        msg_batch = base_msg.unsqueeze(1).expand(B, K, K, D).clone()
        msg_batch = msg_batch * (1.0 - eye)
        trust_batch = base_trust_mask.unsqueeze(1).expand(B, K, K, 1).clone()
        trust_batch = trust_batch * (1.0 - eye)
        comm_batch = base_comm_mask.unsqueeze(1).expand(B, K, K, 1).clone()
        comm_batch = comm_batch * (1.0 - eye)
        if bool(self.cfg.bypass_vib):
            feat_batch = msg_batch.reshape(B * K, K, D)
        else:
            feat_batch = self.vib_gat.decode_messages(msg_batch.reshape(B * K, K, D))
        self_batch = self_feat.unsqueeze(1).expand(B, K, self_feat.size(-1)).reshape(B * K, self_feat.size(-1))
        comm_flat = self.vib_gat.aggregate_messages(
            self_batch,
            msg_batch.reshape(B * K, K, D),
            trust_batch.reshape(B * K, K, 1),
            comm_batch.reshape(B * K, K, 1),
            recv_features=feat_batch,
        )
        policy_flat = self.residual(self_batch, comm_flat, self.comm_enabled, self.comm_strength)
        logits_flat = self._mask_logits(self.policy_head(policy_flat), None if avail_actions is None else avail_actions.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1))
        logp_flat = torch.log_softmax(logits_flat, dim=-1)
        utility_scores = (base_prob.unsqueeze(1) * (base_logp.unsqueeze(1) - logp_flat.view(B, K, -1))).sum(dim=-1).abs()
        utility_scores = utility_scores * base_comm_mask.squeeze(-1)
        utility_scores = utility_scores / utility_scores.max(dim=1, keepdim=True).values.clamp_min(1e-6)
        utility_scores = utility_scores.unsqueeze(-1) * recv_mask
        utility_scores = utility_scores.clamp(0.0, 1.0)
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
        receiver_context: torch.Tensor | None = None,
        *,
        deterministic: bool,
        training: bool,
    ) -> Dict[str, torch.Tensor]:
        if (not self.comm_enabled) or (self.comm_strength <= 0.0):
            zero = self._zero_scalar(self_feat)
            bsz = self_feat.size(0)
            num_neighbors = neighbor_seq.size(1) if neighbor_seq.dim() >= 2 else 0
            edge_zeros = torch.zeros(bsz, num_neighbors, 1, device=self_feat.device, dtype=self_feat.dtype)
            return {
                "comm_feat": torch.zeros_like(self_feat),
                "kl_loss": zero,
                "kl_raw": zero,
                "vib_consistency_loss": zero,
                "vib_consistency_raw": zero,
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
                "consistency_score_mean": zero,
                "consistency_score_mal": zero,
                "consistency_score_clean": zero,
                "consistency_malicious_gap": zero,
                "edge_reliability": edge_zeros,
                "edge_allocation": edge_zeros,
                "edge_utility": edge_zeros,
                "edge_consistency": edge_zeros,
                "edge_malicious_prob": edge_zeros,
                "edge_valid_mask": edge_zeros,
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

        if bool(self.cfg.bypass_vib):
            sender_messages = neighbor_feat
            kl_loss = self._zero_scalar(self_feat)
            kl_raw = self._zero_scalar(self_feat)
            sender_uncertainty = torch.zeros(
                (*sender_messages.shape[:2], 1),
                device=sender_messages.device,
                dtype=sender_messages.dtype,
            )
        else:
            sender_messages, kl_loss, kl_raw, sender_uncertainty = self.vib_gat.encode_messages(
                neighbor_feat,
                sender_mask,
                deterministic=deterministic,
            )
        if training and not bool(self.cfg.bypass_vib):
            vib_consistency_loss, vib_consistency_raw = self._vib_consistency_loss(neighbor_seq, sender_mask)
        else:
            vib_consistency_loss = self._zero_scalar(self_feat)
            vib_consistency_raw = self._zero_scalar(self_feat)
        if channel_noise is None:
            recv_messages = sender_messages
        else:
            if channel_noise.size(-1) != sender_messages.size(-1):
                raise ValueError(
                    f"channel_noise dim {channel_noise.size(-1)} does not match message dim {sender_messages.size(-1)}. "
                    "For bypass_vib, set model.latent_dim to model.hidden_dim."
                )
            recv_messages = sender_messages + torch.nan_to_num(channel_noise, nan=0.0, posinf=0.0, neginf=0.0)
        recv_messages = recv_messages * recv_mask
        if bool(self.cfg.bypass_vib):
            recv_feat = recv_messages
        else:
            recv_feat = self.vib_gat.decode_messages(recv_messages)

        trust_context = receiver_context if receiver_context is not None else self_feat
        trust_logits, malicious_logits, utility_logits, reliability, utility_score, consistency_score = self._compute_utility(
            trust_context,
            recv_feat,
            sender_uncertainty,
        )
        allocation_source = utility_score if bool(getattr(self.cfg, "trust_decouple_allocation", False)) else reliability
        raw_allocation = self._compute_allocation(allocation_source)
        raw_allocation = raw_allocation * recv_mask
        allocation_scores, aggregate_mask = self._blend_allocation(raw_allocation, recv_mask, gate_scores=reliability)
        attn_trust_mask = allocation_scores * recv_mask + (1e-6 * (1.0 - recv_mask))

        utility_targets = None
        if training and self.cfg.enable_trust:
            with torch.no_grad():
                utility_targets = self._compute_counterfactual_utility(
                    trust_context,
                    recv_messages,
                    recv_mask,
                    recv_mask,
                    recv_mask,
                    recv_feat,
                    avail_actions,
                )

        comm_feat = self.vib_gat.aggregate_messages(
            trust_context,
            recv_messages,
            attn_trust_mask,
            aggregate_mask,
            recv_features=recv_feat,
        )
        comm_feat = self.comm_dropout(comm_feat)

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
                consistency_scores=consistency_score,
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

        malicious_prob = torch.sigmoid(malicious_logits).clamp(0.0, 1.0)

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
        consistency_score_mean = zero
        consistency_score_mal = zero
        consistency_score_clean = zero
        consistency_malicious_gap = zero

        if utility_values.numel() > 0:
            allocation_score_mean = aggregate_mask.squeeze(-1)[valid_mask].mean()
            malicious_score_mean = malicious_prob.squeeze(-1)[valid_mask].mean()
            utility_score_mean = utility_score.squeeze(-1)[valid_mask].mean()
            consistency_score_mean = consistency_score.squeeze(-1)[valid_mask].mean()

        if neighbor_malicious is not None:
            mal_mask = (neighbor_malicious > 0.5).squeeze(-1) & valid_mask
            if bool(valid_mask.any()):
                comm_malicious_ratio = mal_mask.float().sum() / valid_mask.float().sum().clamp_min(1.0)
            clean_mask = valid_mask & (~mal_mask)
            if bool(mal_mask.any()):
                malicious_score_mal = malicious_prob.squeeze(-1)[mal_mask].mean()
                allocation_score_mal = aggregate_mask.squeeze(-1)[mal_mask].mean()
                utility_score_mal = utility_score.squeeze(-1)[mal_mask].mean()
                consistency_score_mal = consistency_score.squeeze(-1)[mal_mask].mean()
                trust_mal = reliability.squeeze(-1)[mal_mask].mean()
            else:
                trust_mal = zero
            if bool(clean_mask.any()):
                malicious_score_clean = malicious_prob.squeeze(-1)[clean_mask].mean()
                allocation_score_clean = aggregate_mask.squeeze(-1)[clean_mask].mean()
                utility_score_clean = utility_score.squeeze(-1)[clean_mask].mean()
                consistency_score_clean = consistency_score.squeeze(-1)[clean_mask].mean()
                trust_clean = reliability.squeeze(-1)[clean_mask].mean()
            else:
                trust_clean = zero
            if bool(mal_mask.any()) and bool(clean_mask.any()):
                trust_malicious_gap = trust_clean - trust_mal
                allocation_malicious_gap = allocation_score_clean - allocation_score_mal
                utility_malicious_gap = utility_score_clean - utility_score_mal
                consistency_malicious_gap = consistency_score_clean - consistency_score_mal

        return {
            "comm_feat": comm_feat,
            "kl_loss": kl_loss,
            "kl_raw": kl_raw,
            "vib_consistency_loss": vib_consistency_loss,
            "vib_consistency_raw": vib_consistency_raw,
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
            "consistency_score_mean": consistency_score_mean,
            "consistency_score_mal": consistency_score_mal,
            "consistency_score_clean": consistency_score_clean,
            "consistency_malicious_gap": consistency_malicious_gap,
            "edge_reliability": reliability,
            "edge_allocation": aggregate_mask,
            "edge_utility": utility_score,
            "edge_consistency": consistency_score,
            "edge_malicious_prob": malicious_prob,
            "edge_valid_mask": recv_mask,
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
        belief_out = self._compute_belief_prior(
            self_feat,
            rnn_states_actor,
            masks,
            training=False,
        )
        receiver_context = belief_out["receiver_context"] if bool(getattr(self.cfg, "enable_belief_router", False)) else None
        comm_out = self._comm_forward(
            self_feat,
            neighbor_seq,
            neighbor_mask,
            channel_mask,
            channel_noise,
            neighbor_next_actions,
            neighbor_malicious=neighbor_malicious,
            avail_actions=avail_actions,
            receiver_context=receiver_context,
            deterministic=bool(deterministic) or bool(self.cfg.vib_deterministic),
            training=False,
        )

        fused, belief_debug = self._belief_route(self_feat, comm_out["comm_feat"], comm_out, belief_out)
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
            "belief_prior_loss": belief_debug["belief_prior_loss"],
        }
        if return_debug:
            out.update(comm_out)
            out.update(belief_debug)
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
        belief_out = self._compute_belief_prior(
            self_feat,
            rnn_states_actor,
            masks,
            training=True,
        )
        receiver_context = belief_out["receiver_context"] if bool(getattr(self.cfg, "enable_belief_router", False)) else None
        comm_out = self._comm_forward(
            self_feat,
            neighbor_seq,
            neighbor_mask,
            channel_mask,
            channel_noise,
            neighbor_next_actions,
            neighbor_malicious,
            avail_actions=avail_actions,
            receiver_context=receiver_context,
            deterministic=bool(self.cfg.vib_deterministic),
            training=True,
        )

        fused, belief_debug = self._belief_route(self_feat, comm_out["comm_feat"], comm_out, belief_out)
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
        out.update(belief_debug)
        return out

    def get_values(
        self,
        state: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._critic_forward(state, rnn_states_critic, masks)
