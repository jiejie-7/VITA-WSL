import numpy as np
import torch
import torch.nn as nn

from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import get_gard_norm


class R_TARMAC(R_MAPPO):
    """PPO trainer for the MAPPO-backbone TarMAC baseline."""

    def ppo_update(self, sample, update_actor=True):
        (
            share_obs_batch,
            obs_batch,
            neighbor_obs_batch,
            neighbor_rnn_states_batch,
            _neighbor_actions_batch,
            neighbor_masks_batch,
            neighbor_channel_masks_batch,
            neighbor_channel_noise_batch,
            _neighbor_malicious_batch,
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

        values, action_log_probs, dist_entropy, debug = self.policy.evaluate_actions_tarmac(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            neighbor_obs_batch,
            neighbor_rnn_states_batch,
            neighbor_masks_batch,
            neighbor_channel_masks_batch,
            neighbor_channel_noise_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum().clamp_min(1.0)
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, debug

    def train(self, buffer, update_actor=True):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        advantages = (advantages - np.nanmean(advantages_copy)) / (np.nanstd(advantages_copy) + 1e-5)

        train_info = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "dist_entropy": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "ratio": 0.0,
            "comm_env_neighbors": 0.0,
            "comm_effective_neighbors": 0.0,
            "comm_attention_entropy": 0.0,
        }

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, debug = (
                    self.ppo_update(sample, update_actor)
                )

                train_info["value_loss"] += float(value_loss.item())
                train_info["policy_loss"] += float(policy_loss.item())
                train_info["dist_entropy"] += float(dist_entropy.item())
                train_info["actor_grad_norm"] += float(actor_grad_norm)
                train_info["critic_grad_norm"] += float(critic_grad_norm)
                train_info["ratio"] += float(imp_weights.mean().item())
                train_info["comm_env_neighbors"] += float(debug.get("comm_env_neighbors", 0.0))
                train_info["comm_effective_neighbors"] += float(debug.get("comm_effective_neighbors", 0.0))
                train_info["comm_attention_entropy"] += float(debug.get("comm_attention_entropy", 0.0))

        num_updates = float(max(1, self.ppo_epoch * self.num_mini_batch))
        for key in train_info:
            train_info[key] /= num_updates
        return train_info
