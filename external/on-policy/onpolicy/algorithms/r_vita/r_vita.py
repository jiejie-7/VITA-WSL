import numpy as np
import torch
import torch.nn as nn

from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm


def _schedule_coeff(elapsed: int, warmup: int) -> float:
    if elapsed <= 0:
        return 0.0
    if warmup <= 0:
        return 1.0
    return min(1.0, elapsed / float(warmup))


def _resolve_schedule(primary: int, fallback: int) -> int:
    primary = int(primary)
    fallback = int(fallback)
    return fallback if primary < 0 else primary


class R_VITA:
    """Trainer for VITA, implemented on top of the official RMAPPo pipeline."""

    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        if self._use_popart:
            raise NotImplementedError("R_VITA does not support PopArt in this repository.")

        self.value_normalizer = ValueNorm(1, device=self.device) if self._use_valuenorm else None

        self._enable_trust = not bool(getattr(args, "vita_disable_trust", False))
        self._enable_kl = not bool(getattr(args, "vita_disable_kl", False))

        self._trust_lambda = float(getattr(args, "vita_trust_lambda", 0.1))
        legacy_trust_warmup = int(getattr(args, "vita_trust_warmup_updates", 0))
        legacy_trust_delay = int(getattr(args, "vita_trust_delay_updates", 0))
        self._trust_loss_warmup = _resolve_schedule(
            getattr(args, "vita_trust_loss_warmup_updates", -1),
            legacy_trust_warmup,
        )
        self._trust_loss_delay = _resolve_schedule(
            getattr(args, "vita_trust_loss_delay_updates", -1),
            legacy_trust_delay,
        )
        self._kl_warmup = int(getattr(args, "vita_kl_warmup_updates", 0))
        self._kl_delay = int(getattr(args, "vita_kl_delay_updates", 0))

        self._current_update = 0

    def set_update(self, update: int) -> None:
        self._current_update = int(update)
        if hasattr(self.policy, "update_schedules"):
            self.policy.update_schedules(self._current_update)

    def _trust_coeff(self) -> float:
        if not self._enable_trust:
            return 0.0
        elapsed = self._current_update - self._trust_loss_delay
        return self._trust_lambda * _schedule_coeff(elapsed, self._trust_loss_warmup)

    def _kl_coeff(self) -> float:
        if not self._enable_kl:
            return 0.0
        elapsed = self._current_update - self._kl_delay
        return _schedule_coeff(elapsed, self._kl_warmup)

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        value_loss = torch.max(value_loss_original, value_loss_clipped) if self._use_clipped_value_loss else value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum().clamp_min(1.0)
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        (
            share_obs_batch,
            obs_batch,
            neighbor_obs_batch,
            neighbor_actions_batch,
            neighbor_masks_batch,
            neighbor_channel_masks_batch,
            neighbor_channel_noise_batch,
            neighbor_malicious_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        (
            values,
            action_log_probs,
            dist_entropy,
            kl_loss,
            trust_loss,
            vib_consistency_loss,
            belief_prior_loss,
            debug,
        ) = self.policy.evaluate_actions_vita(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            neighbor_obs_batch,
            neighbor_actions_batch,
            neighbor_masks_batch,
            neighbor_channel_masks_batch,
            neighbor_channel_noise_batch,
            neighbor_malicious_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum().clamp_min(1.0)
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        trust_coeff = float(self._trust_coeff())
        kl_coeff = float(self._kl_coeff())

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            (
                policy_action_loss
                - dist_entropy * self.entropy_coef
                + kl_coeff * kl_loss
                + trust_coeff * trust_loss
                + vib_consistency_loss
                + belief_prior_loss
            ).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor_parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor_parameters())
        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic_parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic_parameters())
        self.policy.critic_optimizer.step()

        return (
            value_loss,
            critic_grad_norm,
            policy_action_loss,
            dist_entropy,
            actor_grad_norm,
            imp_weights,
            kl_loss,
            trust_loss,
            vib_consistency_loss,
            belief_prior_loss,
            debug,
        )

    def train(self, buffer, update_actor=True):
        if self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "dist_entropy": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "ratio": 0.0,
            "kl": 0.0,
            "kl_raw": 0.0,
            "vib_consistency_loss": 0.0,
            "vib_consistency_raw": 0.0,
            "belief_prior_loss": 0.0,
            "belief_error": 0.0,
            "belief_self_conf": 0.0,
            "belief_comm_conf": 0.0,
            "route_w_self": 0.0,
            "route_w_social": 0.0,
            "route_w_prior": 0.0,
            "trust_loss": 0.0,
            "trust_score_mean": 0.0,
            "trust_score_p10": 0.0,
            "trust_score_p50": 0.0,
            "trust_score_p90": 0.0,
            "trust_gate_ratio": 0.0,
            "comm_valid_neighbors": 0.0,
            "comm_kept_neighbors": 0.0,
            "comm_malicious_ratio": 0.0,
            "trust_malicious_gap": 0.0,
            "consistency_score_mean": 0.0,
            "consistency_score_mal": 0.0,
            "consistency_score_clean": 0.0,
            "consistency_malicious_gap": 0.0,
            "comm_strength": 0.0,
            "trust_strength": 0.0,
            "comm_enabled": 0.0,
            "residual_gate_mean": 0.0,
            "residual_gate_max": 0.0,
            "residual_comm_ratio": 0.0,
        }

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                    kl_loss,
                    trust_loss,
                    vib_consistency_loss,
                    belief_prior_loss,
                    debug,
                ) = self.ppo_update(sample, update_actor)

                train_info["value_loss"] += float(value_loss.item())
                train_info["policy_loss"] += float(policy_loss.item())
                train_info["dist_entropy"] += float(dist_entropy.item())
                train_info["actor_grad_norm"] += float(actor_grad_norm)
                train_info["critic_grad_norm"] += float(critic_grad_norm)
                train_info["ratio"] += float(imp_weights.mean().item())
                train_info["kl"] += float(kl_loss.item())
                train_info["trust_loss"] += float(trust_loss.item())
                train_info["vib_consistency_loss"] += float(vib_consistency_loss.item())
                train_info["belief_prior_loss"] += float(belief_prior_loss.item())
                train_info["kl_raw"] += float(debug.get("kl_raw", 0.0))
                train_info["vib_consistency_raw"] += float(debug.get("vib_consistency_raw", 0.0))
                train_info["belief_error"] += float(debug.get("belief_error", 0.0))
                train_info["belief_self_conf"] += float(debug.get("belief_self_conf", 0.0))
                train_info["belief_comm_conf"] += float(debug.get("belief_comm_conf", 0.0))
                train_info["route_w_self"] += float(debug.get("route_w_self", 0.0))
                train_info["route_w_social"] += float(debug.get("route_w_social", 0.0))
                train_info["route_w_prior"] += float(debug.get("route_w_prior", 0.0))
                train_info["trust_score_mean"] += float(debug.get("trust_score_mean", 0.0))
                train_info["trust_score_p10"] += float(debug.get("trust_score_p10", 0.0))
                train_info["trust_score_p50"] += float(debug.get("trust_score_p50", 0.0))
                train_info["trust_score_p90"] += float(debug.get("trust_score_p90", 0.0))
                train_info["trust_gate_ratio"] += float(debug.get("trust_gate_ratio", 0.0))
                train_info["comm_valid_neighbors"] += float(debug.get("comm_valid_neighbors", 0.0))
                train_info["comm_kept_neighbors"] += float(debug.get("comm_kept_neighbors", 0.0))
                train_info["comm_malicious_ratio"] += float(debug.get("comm_malicious_ratio", 0.0))
                train_info["trust_malicious_gap"] += float(debug.get("trust_malicious_gap", 0.0))
                train_info["consistency_score_mean"] += float(debug.get("consistency_score_mean", 0.0))
                train_info["consistency_score_mal"] += float(debug.get("consistency_score_mal", 0.0))
                train_info["consistency_score_clean"] += float(debug.get("consistency_score_clean", 0.0))
                train_info["consistency_malicious_gap"] += float(debug.get("consistency_malicious_gap", 0.0))
                train_info["comm_strength"] += float(debug.get("comm_strength", 0.0))
                train_info["trust_strength"] += float(debug.get("trust_strength", 0.0))
                train_info["comm_enabled"] += float(debug.get("comm_enabled", 0.0))
                train_info["residual_gate_mean"] += float(debug.get("residual_gate_mean", 0.0))
                train_info["residual_gate_max"] += float(debug.get("residual_gate_max", 0.0))
                train_info["residual_comm_ratio"] += float(debug.get("residual_comm_ratio", 0.0))

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= float(max(1, num_updates))
        return train_info

    def prep_training(self):
        self.policy.agent.train()

    def prep_rollout(self):
        self.policy.agent.eval()
