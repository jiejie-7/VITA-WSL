from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math

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
    max_neighbors: int = 5
    comm_dropout: float = 0.1
    enable_trust: bool = True
    enable_kl: bool = True
    vib_deterministic: bool = False
    attention_only: bool = False
    trust_gate_floor: float = 0.0
    attn_bias_coef: float = 1.0


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
        if not has_trust:
            return super().load_state_dict(state_dict, strict=False)
        return super().load_state_dict(state_dict, strict=strict)

    def set_comm_enabled(self, enabled: bool) -> None:
        self.comm_enabled = enabled
        if not enabled:
            self.comm_strength = 0.0

    def set_comm_strength(self, strength: float) -> None:
        strength = float(max(0.0, min(1.0, strength)))
        self.comm_strength = strength

    def set_trust_strength(self, strength: float) -> None:
        if not self.cfg.enable_trust:
            self.trust_strength = 0.0
            return
        strength = float(max(0.0, min(1.0, strength)))
        self.trust_strength = strength

    def set_trust_active(self, active: bool) -> None:
        self.set_trust_strength(1.0 if active else 0.0)

    @property
    def rnn_hidden_dim(self) -> int:
        return self.cfg.hidden_dim

    def _encode_neighbors(self, neighbor_seq: torch.Tensor) -> torch.Tensor:
        # neighbor_seq: [B, K, T, obs_dim]
        B, K, T, D = neighbor_seq.shape
        flat = neighbor_seq.view(B * K, T, D)
        feat, _ = self.comm_encoder(flat, None, None)
        feat = self.neighbor_norm(feat)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat.view(B, K, -1)

    def _mask_logits(self, logits: torch.Tensor, avail_actions: torch.Tensor | None) -> torch.Tensor:
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        if avail_actions is None:
            return logits
        avail_actions = torch.nan_to_num(avail_actions, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        logits = logits + (avail_actions * 1e-6)
        mask = (avail_actions < 0.5)
        all_masked = mask.all(dim=-1, keepdim=True)
        if all_masked.any():
            # Ensure at least one action remains valid per row.
            idx = torch.argmax(avail_actions, dim=-1, keepdim=True)
            mask = mask.clone()
            mask.scatter_(dim=-1, index=idx, value=False)
        return logits.masked_fill(mask, -1e9)

    def _predict_trust(
        self,
        neighbor_feat: torch.Tensor,
        neighbor_next_actions: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, trust = self.trust_predictor(neighbor_feat, neighbor_next_actions)
        trust = torch.nan_to_num(trust, nan=1.0, posinf=1.0, neginf=1.0)
        return logits, trust.clamp(min=1e-6, max=1.0)

    def _trust_loss(
        self,
        trust_logits: torch.Tensor,
        trust_scores: torch.Tensor | None,
        neighbor_next_actions: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None = None,
        neighbor_malicious: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if neighbor_next_actions is None:
            return torch.zeros(1, device=trust_logits.device)
        has_label = neighbor_next_actions.sum(dim=-1, keepdim=True) > 1e-6
        if not bool(has_label.any()):
            return torch.zeros(1, device=trust_logits.device)
        log_probs = F.log_softmax(trust_logits, dim=-1)
        ce = -(neighbor_next_actions * log_probs).sum(dim=-1, keepdim=True)
        ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
        denom = math.log(max(int(trust_logits.size(-1)), 2))
        ce = ce / denom
        valid = has_label
        if neighbor_mask is not None:
            valid = valid & (neighbor_mask > 0.5)
        base_loss = (ce * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
        aux_loss = torch.zeros(1, device=trust_logits.device)
        if neighbor_malicious is not None and trust_scores is not None:
            sup_mask = torch.ones_like(trust_scores, dtype=torch.bool)
            if neighbor_mask is not None:
                sup_mask = sup_mask & (neighbor_mask > 0.5)
            if bool(sup_mask.any()):
                target = (1.0 - neighbor_malicious).clamp(0.0, 1.0)
                trust_scores = torch.nan_to_num(trust_scores, nan=1.0, posinf=1.0, neginf=1.0)
                trust_scores = trust_scores.clamp(1e-6, 1.0 - 1e-6)
                bce = F.binary_cross_entropy(trust_scores, target, reduction="none")
                aux_loss = (bce * sup_mask.float()).sum() / sup_mask.float().sum().clamp_min(1.0)
        weight = float(getattr(self.cfg, "trust_malicious_weight", 1.0))
        return base_loss + weight * aux_loss

    def act(
        self,
        obs_seq: torch.Tensor,
        state: torch.Tensor,
        neighbor_seq: torch.Tensor,
        neighbor_mask: torch.Tensor | None,
        channel_mask: torch.Tensor | None,
        channel_noise: torch.Tensor | None,
        neighbor_next_actions: torch.Tensor | None,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
        avail_actions: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self_feat, next_actor = self.actor_encoder(obs_seq, rnn_states_actor.unsqueeze(0), masks)
        self_feat = torch.nan_to_num(self_feat, nan=0.0, posinf=0.0, neginf=0.0)
        if (not self.comm_enabled) or (self.comm_strength <= 0.0):
            comm_feat = torch.zeros_like(self_feat)
            kl_loss = torch.zeros(1, device=self_feat.device)
            kl_raw = torch.zeros(1, device=self_feat.device)
        else:
            neighbor_feat = self._encode_neighbors(neighbor_seq)
            if neighbor_mask is None:
                neighbor_mask = torch.ones(neighbor_feat.size(0), neighbor_feat.size(1), 1, device=neighbor_feat.device)
            sender_mask = torch.nan_to_num(neighbor_mask.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            if channel_mask is None:
                recv_mask = sender_mask
            else:
                recv_mask = torch.nan_to_num(channel_mask.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                recv_mask = recv_mask * sender_mask
            vib_deterministic = bool(deterministic) or bool(self.cfg.vib_deterministic)
            sender_messages, kl_loss, kl_raw = self.vib_gat.encode_messages(
                neighbor_feat,
                sender_mask,
                deterministic=vib_deterministic,
            )
            if channel_noise is None:
                recv_messages = sender_messages
            else:
                recv_messages = sender_messages + torch.nan_to_num(channel_noise, nan=0.0, posinf=0.0, neginf=0.0)
            recv_messages = recv_messages * recv_mask
            recv_feat = self.vib_gat.decode_messages(recv_messages)
            use_trust = self.cfg.enable_trust
            if use_trust:
                _, trust_scores = self._predict_trust(recv_feat, neighbor_next_actions)
                trust_scores = trust_scores.detach()
                trust_scores = (1.0 - float(self.trust_strength)) + float(self.trust_strength) * trust_scores
            else:
                trust_scores = torch.ones_like(recv_mask)
            trust_gate_soft = trust_scores.clamp(min=0.0, max=1.0)
            gate_floor = float(max(0.0, min(1.0, float(self.cfg.trust_gate_floor))))
            if gate_floor > 0.0:
                trust_gate_soft = gate_floor + (1.0 - gate_floor) * trust_gate_soft
            aggregate_mask = recv_mask * trust_gate_soft
            trust_scores = trust_scores * recv_mask + (1e-6 * (1.0 - recv_mask))
            comm_feat = self.vib_gat.aggregate_messages(
                self_feat,
                recv_messages,
                trust_scores,
                aggregate_mask,
                recv_features=recv_feat,
            )
            kl_loss = torch.nan_to_num(kl_loss, nan=0.0, posinf=0.0, neginf=0.0)
            kl_raw = torch.nan_to_num(kl_raw, nan=0.0, posinf=0.0, neginf=0.0)
            if not self.cfg.enable_kl:
                kl_loss = torch.zeros(1, device=self_feat.device)
            comm_feat = self.comm_dropout(comm_feat)
        fused = self.residual(self_feat, comm_feat, self.comm_enabled, self.comm_strength)
        logits = self.policy_head(fused)
        logits = self._mask_logits(logits, avail_actions)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            actions = dist.sample().unsqueeze(-1)
        log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        critic_feat, next_critic = self.critic_encoder(state.unsqueeze(1), rnn_states_critic.unsqueeze(0), masks)
        critic_feat = self.critic_mlp(critic_feat)
        values = self.value_head(critic_feat)

        return {
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "entropy": entropy,
            "next_actor_state": next_actor.squeeze(0),
            "next_critic_state": next_critic.squeeze(0),
            "kl_loss": kl_loss,
            "kl_raw": kl_raw,
        }

    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        state: torch.Tensor,
        neighbor_seq: torch.Tensor,
        neighbor_mask: torch.Tensor | None,
        channel_mask: torch.Tensor | None,
        channel_noise: torch.Tensor | None,
        neighbor_next_actions: torch.Tensor,
        neighbor_malicious: torch.Tensor | None,
        actions: torch.Tensor,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self_feat, next_actor = self.actor_encoder(obs_seq, rnn_states_actor.unsqueeze(0), masks)
        self_feat = torch.nan_to_num(self_feat, nan=0.0, posinf=0.0, neginf=0.0)
        if (not self.comm_enabled) or (self.comm_strength <= 0.0):
            comm_feat = torch.zeros_like(self_feat)
            kl_loss = torch.zeros(1, device=self_feat.device)
            kl_raw = torch.zeros(1, device=self_feat.device)
            trust_loss = torch.zeros(1, device=obs_seq.device)
            trust_score_mean = torch.zeros(1, device=obs_seq.device)
            trust_score_p10 = torch.zeros(1, device=obs_seq.device)
            trust_score_p50 = torch.zeros(1, device=obs_seq.device)
            trust_score_p90 = torch.zeros(1, device=obs_seq.device)
            trust_gate_ratio = torch.zeros(1, device=obs_seq.device)
            comm_valid_neighbors = torch.zeros(1, device=obs_seq.device)
            comm_kept_neighbors = torch.zeros(1, device=obs_seq.device)
            comm_malicious_ratio = torch.zeros(1, device=obs_seq.device)
            trust_malicious_gap = torch.zeros(1, device=obs_seq.device)
        else:
            neighbor_feat = self._encode_neighbors(neighbor_seq)
            if neighbor_mask is None:
                neighbor_mask = torch.ones(neighbor_feat.size(0), neighbor_feat.size(1), 1, device=neighbor_feat.device)
            sender_mask = torch.nan_to_num(neighbor_mask.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            if channel_mask is None:
                recv_mask = sender_mask
            else:
                recv_mask = torch.nan_to_num(channel_mask.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                recv_mask = recv_mask * sender_mask
            sender_messages, kl_loss, kl_raw = self.vib_gat.encode_messages(
                neighbor_feat,
                sender_mask,
                deterministic=bool(self.cfg.vib_deterministic),
            )
            if channel_noise is None:
                recv_messages = sender_messages
            else:
                recv_messages = sender_messages + torch.nan_to_num(channel_noise, nan=0.0, posinf=0.0, neginf=0.0)
            recv_messages = recv_messages * recv_mask
            recv_feat = self.vib_gat.decode_messages(recv_messages)

            trust_loss = torch.zeros(1, device=obs_seq.device)
            trust_scores_raw = None
            use_trust = self.cfg.enable_trust
            if use_trust:
                trust_logits, trust_scores_raw = self._predict_trust(recv_feat, neighbor_next_actions)
                trust_loss = self._trust_loss(
                    trust_logits,
                    trust_scores_raw,
                    neighbor_next_actions,
                    neighbor_mask=recv_mask,
                    neighbor_malicious=neighbor_malicious,
                )
                trust_scores_raw = trust_scores_raw.detach()
                trust_scores = (1.0 - float(self.trust_strength)) + float(self.trust_strength) * trust_scores_raw
            else:
                trust_scores = torch.ones_like(recv_mask)
                trust_scores_raw = None
            trust_gate_soft = trust_scores.clamp(min=0.0, max=1.0)
            gate_floor = float(max(0.0, min(1.0, float(self.cfg.trust_gate_floor))))
            if gate_floor > 0.0:
                trust_gate_soft = gate_floor + (1.0 - gate_floor) * trust_gate_soft
            aggregate_mask = recv_mask * trust_gate_soft
            trust_scores = trust_scores * recv_mask + (1e-6 * (1.0 - recv_mask))
            comm_feat = self.vib_gat.aggregate_messages(
                self_feat,
                recv_messages,
                trust_scores,
                aggregate_mask,
                recv_features=recv_feat,
            )
            kl_loss = torch.nan_to_num(kl_loss, nan=0.0, posinf=0.0, neginf=0.0)
            kl_raw = torch.nan_to_num(kl_raw, nan=0.0, posinf=0.0, neginf=0.0)
            kl_loss = kl_loss * float(self.comm_strength)
            if not self.cfg.enable_kl:
                kl_loss = torch.zeros(1, device=self_feat.device)
            comm_feat = self.comm_dropout(comm_feat)

            valid_mask = (recv_mask > 0.5).squeeze(-1)
            sender_valid_mask = (sender_mask > 0.5).squeeze(-1)
            comm_valid_neighbors = sender_valid_mask.float().sum(dim=-1).mean()
            comm_kept_neighbors = aggregate_mask.squeeze(-1).sum(dim=-1).mean()
            trust_values = (trust_scores_raw if trust_scores_raw is not None else trust_scores).squeeze(-1)[valid_mask]
            if trust_values.numel() == 0:
                trust_score_mean = torch.zeros(1, device=obs_seq.device)
                trust_score_p10 = torch.zeros(1, device=obs_seq.device)
                trust_score_p50 = torch.zeros(1, device=obs_seq.device)
                trust_score_p90 = torch.zeros(1, device=obs_seq.device)
                trust_gate_ratio = torch.zeros(1, device=obs_seq.device)
            else:
                trust_score_mean = trust_values.mean()
                q = torch.tensor([0.1, 0.5, 0.9], device=trust_values.device, dtype=trust_values.dtype)
                qv = torch.quantile(trust_values, q)
                trust_score_p10 = qv[0]
                trust_score_p50 = qv[1]
                trust_score_p90 = qv[2]
                kept = aggregate_mask.squeeze(-1)[valid_mask].sum()
                trust_gate_ratio = kept / valid_mask.float().sum().clamp_min(1.0)
            comm_malicious_ratio = torch.zeros(1, device=obs_seq.device)
            trust_malicious_gap = torch.zeros(1, device=obs_seq.device)
            mal_mask = None
            if neighbor_malicious is not None:
                mal_mask = (neighbor_malicious > 0.5).squeeze(-1) & valid_mask
                if bool(valid_mask.any()):
                    comm_malicious_ratio = mal_mask.float().sum() / valid_mask.float().sum().clamp_min(1.0)
            if mal_mask is not None and trust_scores_raw is not None:
                if bool(mal_mask.any()):
                    trust_mal = trust_scores_raw.squeeze(-1)[mal_mask].mean()
                else:
                    trust_mal = torch.zeros(1, device=obs_seq.device)
                clean_mask = valid_mask & (~mal_mask)
                if bool(clean_mask.any()):
                    trust_clean = trust_scores_raw.squeeze(-1)[clean_mask].mean()
                else:
                    trust_clean = torch.zeros(1, device=obs_seq.device)
                trust_malicious_gap = trust_clean - trust_mal
        fused = self.residual(self_feat, comm_feat, self.comm_enabled, self.comm_strength)
        if (not self.comm_enabled) or (self.comm_strength <= 0.0):
            residual_gate_mean = torch.zeros(1, device=obs_seq.device)
            residual_gate_max = torch.zeros(1, device=obs_seq.device)
            residual_comm_ratio = torch.zeros(1, device=obs_seq.device)
        else:
            with torch.no_grad():
                strength = float(max(0.0, min(1.0, float(self.comm_strength))))
                strength_t = torch.as_tensor(strength, device=self_feat.device, dtype=self_feat.dtype)
                gate_input = torch.cat([self_feat, comm_feat], dim=-1)
                gate = torch.sigmoid(self.residual.gate(gate_input)) * strength_t
                residual_gate_mean = gate.mean()
                residual_gate_max = gate.max()
                comm_contrib = (gate * comm_feat).norm(dim=-1)
                self_norm = self_feat.norm(dim=-1).clamp_min(1e-6)
                residual_comm_ratio = (comm_contrib / self_norm).mean()
        logits = self.policy_head(fused)
        logits = self._mask_logits(logits, avail_actions)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        critic_feat, next_critic = self.critic_encoder(state.unsqueeze(1), rnn_states_critic.unsqueeze(0), masks)
        critic_feat = self.critic_mlp(critic_feat)
        values = self.value_head(critic_feat)
        return {
            "log_probs": log_probs,
            "entropy": entropy,
            "values": values,
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
            "comm_strength": torch.tensor(float(self.comm_strength), device=obs_seq.device),
            "trust_strength": torch.tensor(float(self.trust_strength), device=obs_seq.device),
            "comm_enabled": torch.tensor(float(self.comm_enabled), device=obs_seq.device),
            "residual_gate_mean": residual_gate_mean,
            "residual_gate_max": residual_gate_max,
            "residual_comm_ratio": residual_comm_ratio,
            "next_actor_state": next_actor.squeeze(0),
            "next_critic_state": next_critic.squeeze(0),
        }

    def get_values(
        self,
        state: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        critic_feat, next_state = self.critic_encoder(state.unsqueeze(1), rnn_states_critic.unsqueeze(0), masks)
        critic_feat = self.critic_mlp(critic_feat)
        values = self.value_head(critic_feat)
        return values, next_state.squeeze(0)
