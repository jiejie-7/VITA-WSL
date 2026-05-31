import numpy as np
import torch

from onpolicy.utils.shared_buffer import SharedReplayBuffer, _cast, _flatten


class SharedVITAReplayBuffer(SharedReplayBuffer):
    """SharedReplayBuffer with extra neighbor tensors for VITA."""

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        super().__init__(args, num_agents, obs_space, cent_obs_space, act_space)

        max_neighbors = int(getattr(args, "vita_max_neighbors", 0) or 0)
        if num_agents > 1:
            max_neighbors = max(1, min(max_neighbors, num_agents - 1))
        else:
            max_neighbors = 1
        self.vita_max_neighbors = max_neighbors

        obs_dim = int(self.obs.shape[-1])
        if act_space.__class__.__name__ != "Discrete":
            raise NotImplementedError("VITA buffer currently supports Discrete action spaces only.")
        act_dim = int(act_space.n)
        latent_dim = int(getattr(args, "vita_latent_dim", 64) or 64)

        self.neighbor_obs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, self.vita_max_neighbors, obs_dim),
            dtype=np.float32,
        )
        self.neighbor_masks = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, self.vita_max_neighbors, 1),
            dtype=np.float32,
        )
        self.neighbor_actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, self.vita_max_neighbors, act_dim),
            dtype=np.float32,
        )

        self.neighbor_malicious = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, self.vita_max_neighbors, 1),
            dtype=np.float32,
        )
        self.neighbor_channel_masks = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, self.vita_max_neighbors, 1),
            dtype=np.float32,
        )
        self.neighbor_channel_noise = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, self.vita_max_neighbors, latent_dim),
            dtype=np.float32,
        )

    def insert(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks=None,
        active_masks=None,
        available_actions=None,
        *,
        neighbor_obs,
        neighbor_actions,
        neighbor_masks,
        neighbor_malicious,
        neighbor_channel_masks,
        neighbor_channel_noise,
    ):
        step = int(self.step)
        super().insert(
            share_obs,
            obs,
            rnn_states_actor,
            rnn_states_critic,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            masks,
            bad_masks,
            active_masks,
            available_actions,
        )

        self.neighbor_obs[step] = neighbor_obs.copy()
        self.neighbor_actions[step] = neighbor_actions.copy()
        self.neighbor_masks[step] = neighbor_masks.copy()
        self.neighbor_malicious[step] = neighbor_malicious.copy()
        self.neighbor_channel_masks[step] = neighbor_channel_masks.copy()
        self.neighbor_channel_noise[step] = neighbor_channel_noise.copy()

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({}).".format(
                    n_rollout_threads,
                    episode_length,
                    num_agents,
                    n_rollout_threads * episode_length * num_agents,
                    num_mini_batch,
                )
            )
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        neighbor_obs = self.neighbor_obs.reshape(-1, *self.neighbor_obs.shape[3:])
        neighbor_actions = self.neighbor_actions.reshape(-1, *self.neighbor_actions.shape[3:])
        neighbor_masks = self.neighbor_masks.reshape(-1, *self.neighbor_masks.shape[3:])
        neighbor_malicious = self.neighbor_malicious.reshape(-1, *self.neighbor_malicious.shape[3:])
        neighbor_channel_masks = self.neighbor_channel_masks.reshape(-1, *self.neighbor_channel_masks.shape[3:])
        neighbor_channel_noise = self.neighbor_channel_noise.reshape(-1, *self.neighbor_channel_noise.shape[3:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            neighbor_obs_batch = neighbor_obs[indices]
            neighbor_actions_batch = neighbor_actions[indices]
            neighbor_masks_batch = neighbor_masks[indices]
            neighbor_malicious_batch = neighbor_malicious[indices]
            neighbor_channel_masks_batch = neighbor_channel_masks[indices]
            neighbor_channel_noise_batch = neighbor_channel_noise[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield (
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
            )

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        assert n_rollout_threads * num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_agents, num_mini_batch)
        )
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()

        share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
        obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        neighbor_obs = self.neighbor_obs.reshape(-1, batch_size, *self.neighbor_obs.shape[3:])
        neighbor_actions = self.neighbor_actions.reshape(-1, batch_size, *self.neighbor_actions.shape[3:])
        neighbor_masks = self.neighbor_masks.reshape(-1, batch_size, *self.neighbor_masks.shape[3:])
        neighbor_malicious = self.neighbor_malicious.reshape(-1, batch_size, *self.neighbor_malicious.shape[3:])
        neighbor_channel_masks = self.neighbor_channel_masks.reshape(-1, batch_size, *self.neighbor_channel_masks.shape[3:])
        neighbor_channel_noise = self.neighbor_channel_noise.reshape(-1, batch_size, *self.neighbor_channel_noise.shape[3:])
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(-1, batch_size, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(-1, batch_size, self.available_actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(-1, batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            neighbor_obs_batch = []
            neighbor_actions_batch = []
            neighbor_masks_batch = []
            neighbor_malicious_batch = []
            neighbor_channel_masks_batch = []
            neighbor_channel_noise_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(share_obs[:-1, ind])
                obs_batch.append(obs[:-1, ind])
                neighbor_obs_batch.append(neighbor_obs[:, ind])
                neighbor_actions_batch.append(neighbor_actions[:, ind])
                neighbor_masks_batch.append(neighbor_masks[:, ind])
                neighbor_malicious_batch.append(neighbor_malicious[:, ind])
                neighbor_channel_masks_batch.append(neighbor_channel_masks[:, ind])
                neighbor_channel_noise_batch.append(neighbor_channel_noise[:, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.episode_length, num_envs_per_batch
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            neighbor_obs_batch = np.stack(neighbor_obs_batch, 1)
            neighbor_actions_batch = np.stack(neighbor_actions_batch, 1)
            neighbor_masks_batch = np.stack(neighbor_masks_batch, 1)
            neighbor_malicious_batch = np.stack(neighbor_malicious_batch, 1)
            neighbor_channel_masks_batch = np.stack(neighbor_channel_masks_batch, 1)
            neighbor_channel_noise_batch = np.stack(neighbor_channel_noise_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            neighbor_obs_batch = _flatten(T, N, neighbor_obs_batch)
            neighbor_actions_batch = _flatten(T, N, neighbor_actions_batch)
            neighbor_masks_batch = _flatten(T, N, neighbor_masks_batch)
            neighbor_malicious_batch = _flatten(T, N, neighbor_malicious_batch)
            neighbor_channel_masks_batch = _flatten(T, N, neighbor_channel_masks_batch)
            neighbor_channel_noise_batch = _flatten(T, N, neighbor_channel_noise_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield (
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
            )

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 4:
            share_obs = self.share_obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        neighbor_obs = self.neighbor_obs.transpose(1, 2, 0, 3, 4).reshape(-1, *self.neighbor_obs.shape[3:])
        neighbor_actions = self.neighbor_actions.transpose(1, 2, 0, 3, 4).reshape(-1, *self.neighbor_actions.shape[3:])
        neighbor_masks = self.neighbor_masks.transpose(1, 2, 0, 3, 4).reshape(-1, *self.neighbor_masks.shape[3:])
        neighbor_malicious = self.neighbor_malicious.transpose(1, 2, 0, 3, 4).reshape(-1, *self.neighbor_malicious.shape[3:])
        neighbor_channel_masks = self.neighbor_channel_masks.transpose(1, 2, 0, 3, 4).reshape(-1, *self.neighbor_channel_masks.shape[3:])
        neighbor_channel_noise = self.neighbor_channel_noise.transpose(1, 2, 0, 3, 4).reshape(-1, *self.neighbor_channel_noise.shape[3:])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 2, 0, 3, 4).reshape(
            -1, *self.rnn_states_critic.shape[3:]
        )

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            neighbor_obs_batch = []
            neighbor_actions_batch = []
            neighbor_masks_batch = []
            neighbor_malicious_batch = []
            neighbor_channel_masks_batch = []
            neighbor_channel_noise_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                obs_batch.append(obs[ind : ind + data_chunk_length])
                neighbor_obs_batch.append(neighbor_obs[ind : ind + data_chunk_length])
                neighbor_actions_batch.append(neighbor_actions[ind : ind + data_chunk_length])
                neighbor_masks_batch.append(neighbor_masks[ind : ind + data_chunk_length])
                neighbor_malicious_batch.append(neighbor_malicious[ind : ind + data_chunk_length])
                neighbor_channel_masks_batch.append(neighbor_channel_masks[ind : ind + data_chunk_length])
                neighbor_channel_noise_batch.append(neighbor_channel_noise[ind : ind + data_chunk_length])
                actions_batch.append(actions[ind : ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind : ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind : ind + data_chunk_length])
                adv_targ.append(advantages[ind : ind + data_chunk_length])
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)
            neighbor_obs_batch = np.stack(neighbor_obs_batch, axis=1)
            neighbor_actions_batch = np.stack(neighbor_actions_batch, axis=1)
            neighbor_masks_batch = np.stack(neighbor_masks_batch, axis=1)
            neighbor_malicious_batch = np.stack(neighbor_malicious_batch, axis=1)
            neighbor_channel_masks_batch = np.stack(neighbor_channel_masks_batch, axis=1)
            neighbor_channel_noise_batch = np.stack(neighbor_channel_noise_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            neighbor_obs_batch = _flatten(L, N, neighbor_obs_batch)
            neighbor_actions_batch = _flatten(L, N, neighbor_actions_batch)
            neighbor_masks_batch = _flatten(L, N, neighbor_masks_batch)
            neighbor_malicious_batch = _flatten(L, N, neighbor_malicious_batch)
            neighbor_channel_masks_batch = _flatten(L, N, neighbor_channel_masks_batch)
            neighbor_channel_noise_batch = _flatten(L, N, neighbor_channel_noise_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield (
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
            )

