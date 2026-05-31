from collections import defaultdict
import os
import time

import imageio
import numpy as np
import torch
try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None else str(value)


class FootballRunner(Runner):
    def __init__(self, config):
        super(FootballRunner, self).__init__(config)
        self.env_infos = defaultdict(list)

        self._vita_positions = None
        self._vita_alive = None
        self._vita_comm_rng = None
        self._vita_comm_noise_std = 0.0
        self._vita_comm_drop_prob = 0.0
        self._vita_comm_malicious_prob = 0.0
        self._vita_comm_malicious_noise_scale = 3.0
        self._vita_comm_malicious_mode = "bernoulli"
        self._vita_comm_malicious_fixed_agent_id = 0
        self._vita_comm_malicious_senders = None
        self._vita_latent_dim = 64

        if self.algorithm_name == "rvita":
            seed = int(getattr(self.all_args, "seed", 0))
            self._vita_comm_rng = np.random.RandomState(seed)
            self._vita_comm_noise_std = _env_float("ONPOLICY_COMM_NOISE_STD", 0.0)
            self._vita_comm_drop_prob = _env_float("ONPOLICY_COMM_PACKET_DROP_PROB", 0.0)
            self._vita_comm_malicious_prob = _env_float("ONPOLICY_COMM_MALICIOUS_AGENT_PROB", 0.0)
            self._vita_comm_malicious_noise_scale = _env_float("ONPOLICY_COMM_MALICIOUS_NOISE_SCALE", 3.0)
            self._vita_comm_malicious_mode = _env_str("ONPOLICY_COMM_MALICIOUS_MODE", "bernoulli").strip().lower()
            self._vita_comm_malicious_fixed_agent_id = _env_int("ONPOLICY_COMM_MALICIOUS_FIXED_AGENT_ID", 0)
            self._vita_latent_dim = int(getattr(self.all_args, "vita_latent_dim", 64) or 64)

    def _vita_sample_comm_malicious_senders(self, n_envs: int, n_agents: int):
        if self._vita_comm_rng is None:
            return None
        prob = float(self._vita_comm_malicious_prob)
        if prob <= 0.0 or n_envs <= 0 or n_agents <= 0:
            return None
        prob = float(min(1.0, max(0.0, prob)))

        mode = str(getattr(self, "_vita_comm_malicious_mode", "bernoulli") or "bernoulli").strip().lower()
        if mode in {"fixed", "fixed_agent", "fixed_sender"}:
            agent_id = int(getattr(self, "_vita_comm_malicious_fixed_agent_id", 0))
            if agent_id < 0 or agent_id >= n_agents:
                agent_id = 0
            mask = np.zeros((n_envs, n_agents), dtype=bool)
            if prob >= 1.0:
                mask[:, agent_id] = True
            else:
                active = self._vita_comm_rng.rand(n_envs) < prob
                if active.any():
                    mask[active, agent_id] = True
            return mask

        if mode in {"one", "one_per_episode", "episode", "sample_one", "random_one"}:
            mask = np.zeros((n_envs, n_agents), dtype=bool)
            active = np.ones((n_envs,), dtype=bool) if prob >= 1.0 else (self._vita_comm_rng.rand(n_envs) < prob)
            if active.any():
                env_sel = np.where(active)[0]
                agent_sel = self._vita_comm_rng.randint(0, n_agents, size=env_sel.size)
                mask[env_sel, agent_sel] = True
            return mask

        return self._vita_comm_rng.rand(n_envs, n_agents) < prob

    def _vita_comm_range(self) -> float:
        comm_range = float(getattr(self.all_args, "vita_comm_sight_range", 0.0) or 0.0)
        return comm_range if comm_range > 0.0 else 10.0

    def _vita_extract_positions(self, info):
        if not isinstance(info, dict):
            return None
        left_team = info.get("left_team")
        if left_team is None:
            return None
        left_team = np.asarray(left_team, dtype=np.float32)
        if left_team.ndim != 2 or left_team.shape[1] < 2 or left_team.shape[0] <= 0:
            return None

        indices = info.get("active")
        if indices is None:
            indices = info.get("designated")
        if indices is None:
            return left_team[:self.num_agents, :2].copy()

        indices = np.asarray(indices)
        if indices.ndim == 0:
            indices = np.full((self.num_agents,), int(indices), dtype=np.int64)

        selected = np.zeros((self.num_agents, 2), dtype=np.float32)
        for agent_i in range(self.num_agents):
            raw = indices[agent_i] if agent_i < indices.shape[0] else agent_i
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                idx = agent_i
            idx = min(max(idx, 0), left_team.shape[0] - 1)
            selected[agent_i] = left_team[idx, :2]
        return selected

    def _vita_compute_neighbors(self, obs, *, positions=None, alive=None, comm_malicious_senders=None):
        obs = np.asarray(obs, dtype=np.float32)
        positions = self._vita_positions if positions is None else positions
        alive = self._vita_alive if alive is None else alive

        n_envs, n_agents, obs_dim = obs.shape
        max_neighbors = int(getattr(self.buffer, "vita_max_neighbors", max(1, n_agents - 1)))
        max_neighbors = max(1, min(max_neighbors, n_agents - 1)) if n_agents > 1 else 1

        neighbor_idx = np.zeros((n_envs, n_agents, max_neighbors), dtype=np.int64)
        neighbor_masks = np.zeros((n_envs, n_agents, max_neighbors, 1), dtype=np.float32)
        neighbor_obs = np.zeros((n_envs, n_agents, max_neighbors, obs_dim), dtype=np.float32)
        neighbor_malicious = np.zeros((n_envs, n_agents, max_neighbors, 1), dtype=np.float32)

        if n_agents <= 1:
            return neighbor_obs, neighbor_masks, neighbor_idx, neighbor_malicious

        if positions is None or alive is None:
            positions = np.zeros((n_envs, n_agents, 2), dtype=np.float32)
            alive = np.ones((n_envs, n_agents), dtype=np.float32)

        positions = np.asarray(positions, dtype=np.float32)
        alive = np.asarray(alive, dtype=np.float32)
        if alive.ndim == 3 and alive.shape[-1] == 1:
            alive = alive.squeeze(-1)
        if alive.ndim == 1:
            alive = np.broadcast_to(alive[None, :], (n_envs, n_agents))

        diff = positions[:, :, None, :] - positions[:, None, :, :]
        dist = np.linalg.norm(diff, axis=-1).astype(np.float32)
        for agent_id in range(n_agents):
            dist[:, agent_id, agent_id] = 1e9
        neighbor_idx = np.argsort(dist, axis=-1)[..., :max_neighbors].astype(np.int64)
        dist_k = np.take_along_axis(dist, neighbor_idx, axis=-1)

        comm_range = self._vita_comm_range()
        within = dist_k <= (comm_range + 1e-6)
        env_ids = np.arange(n_envs)[:, None, None]
        neighbor_alive = alive[env_ids, neighbor_idx] > 0.5
        self_alive = alive > 0.5
        combined = within & neighbor_alive & self_alive[..., None]
        neighbor_masks = combined.astype(np.float32)[..., None]

        neighbor_obs = obs[env_ids, neighbor_idx]

        if comm_malicious_senders is None and self._vita_comm_malicious_prob > 0.0:
            comm_malicious_senders = self._vita_comm_malicious_senders
            if comm_malicious_senders is None or comm_malicious_senders.shape[:2] != (n_envs, n_agents):
                comm_malicious_senders = self._vita_sample_comm_malicious_senders(n_envs, n_agents)
        if comm_malicious_senders is not None:
            mal_mask = comm_malicious_senders[env_ids, neighbor_idx][..., None]
            mal_mask = mal_mask & (neighbor_masks > 0.5)
            neighbor_malicious = mal_mask.astype(np.float32)

        neighbor_malicious = neighbor_malicious * neighbor_masks
        return neighbor_obs, neighbor_masks, neighbor_idx, neighbor_malicious

    def _vita_sample_channel_effects(self, neighbor_masks, neighbor_malicious):
        neighbor_masks = np.asarray(neighbor_masks, dtype=np.float32)
        neighbor_malicious = np.asarray(neighbor_malicious, dtype=np.float32)
        n_envs, n_agents, max_neighbors, _ = neighbor_masks.shape
        channel_masks = neighbor_masks.copy()
        channel_noise = np.zeros((n_envs, n_agents, max_neighbors, self._vita_latent_dim), dtype=np.float32)

        if self._vita_comm_rng is None:
            return channel_masks, channel_noise

        if self._vita_comm_drop_prob > 0.0:
            drop_mask = (self._vita_comm_rng.rand(n_envs, n_agents, max_neighbors) < self._vita_comm_drop_prob)[..., None]
            drop_mask = drop_mask & (channel_masks > 0.5)
            if drop_mask.any():
                channel_masks = channel_masks * (1.0 - drop_mask.astype(np.float32))

        if self._vita_comm_noise_std > 0.0:
            noise = self._vita_comm_rng.normal(0.0, self._vita_comm_noise_std, size=channel_noise.shape).astype(np.float32)
            channel_noise += noise

        if np.any(neighbor_malicious > 0.5):
            base_std = self._vita_comm_noise_std if self._vita_comm_noise_std > 0.0 else 1.0
            std = float(max(1e-6, base_std) * max(0.0, self._vita_comm_malicious_noise_scale))
            mal_noise = self._vita_comm_rng.normal(0.0, std, size=channel_noise.shape).astype(np.float32)
            channel_noise += mal_noise * neighbor_malicious

        channel_noise = channel_noise * channel_masks
        return channel_masks, channel_noise

    def _vita_gather_neighbor_actions(self, actions, neighbor_idx):
        actions = np.asarray(actions)
        act_dim = int(self.envs.action_space[0].n)
        act_idx = actions.squeeze(-1).astype(np.int64)
        one_hot = np.eye(act_dim, dtype=np.float32)[act_idx]
        env_ids = np.arange(one_hot.shape[0])[:, None, None]
        return one_hot[env_ids, neighbor_idx]

    def _vita_update_meta(self, infos, *, dones_env=None, positions=None, alive=None) -> None:
        positions = self._vita_positions if positions is None else positions
        alive = self._vita_alive if alive is None else alive
        is_training_meta = (positions is self._vita_positions) and (alive is self._vita_alive)
        if positions is None or alive is None:
            return

        for env_i, info in enumerate(infos):
            pos = self._vita_extract_positions(info)
            if pos is not None:
                positions[env_i] = pos
            alive[env_i] = 1.0

        if dones_env is not None:
            positions[dones_env == True] = 0.0
            alive[dones_env == True] = 1.0
            if is_training_meta:
                idx = np.asarray(dones_env, dtype=bool)
                if idx.any() and self._vita_comm_malicious_senders is not None:
                    sampled = self._vita_sample_comm_malicious_senders(int(idx.sum()), self.num_agents)
                    if sampled is None:
                        self._vita_comm_malicious_senders[idx] = False
                    else:
                        self._vita_comm_malicious_senders[idx] = sampled

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.algorithm_name == "rvita" and hasattr(self.trainer, "set_update"):
                self.trainer.set_update(episode + 1)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                if self.algorithm_name == "rvita":
                    values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, neighbor_obs, neighbor_masks, neighbor_idx, neighbor_malicious, neighbor_channel_masks, neighbor_channel_noise = self.collect(step)
                else:
                    values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                obs, rewards, dones, infos = self.envs.step(actions_env)

                if self.algorithm_name == "rvita":
                    neighbor_actions = self._vita_gather_neighbor_actions(actions, neighbor_idx)
                    data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, neighbor_obs, neighbor_masks, neighbor_actions, neighbor_malicious, neighbor_channel_masks, neighbor_channel_noise
                else:
                    data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

            self.compute()
            train_infos = self.train()
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if total_num_steps % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
            train_metrics = {
                "phase": "train",
                "update": episode + 1,
                "total_env_steps": int(total_num_steps),
                "average_episode_rewards": float(train_infos["average_episode_rewards"]),
            }
            for key, value in train_infos.items():
                if np.isscalar(value):
                    train_metrics[key] = float(value)
            for key in ("goal", "win_rate", "steps"):
                values = self.env_infos.get(key)
                if values:
                    train_metrics[key] = float(np.mean(values))
            self.log_json(train_metrics, step=episode + 1)

            if total_num_steps % self.log_interval == 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.env_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.env_infos = defaultdict(list)

            if total_num_steps % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        obs = self.envs.reset()
        self.buffer.share_obs[0] = obs.copy()
        self.buffer.obs[0] = obs.copy()

        if self.algorithm_name == "rvita":
            self._vita_positions = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.float32)
            self._vita_alive = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
            self._vita_comm_malicious_senders = self._vita_sample_comm_malicious_senders(self.n_rollout_threads, self.num_agents)

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        if self.algorithm_name == "rvita":
            obs_now = self.buffer.obs[step]
            neighbor_obs, neighbor_masks, neighbor_idx, neighbor_malicious = self._vita_compute_neighbors(obs_now)
            neighbor_channel_masks, neighbor_channel_noise = self._vita_sample_channel_effects(neighbor_masks, neighbor_malicious)
            act_dim = int(self.envs.action_space[0].n)
            neighbor_action_labels = np.zeros((neighbor_obs.shape[0], neighbor_obs.shape[1], neighbor_obs.shape[2], act_dim), dtype=np.float32)
            self.trainer.policy.set_step_context(
                neighbor_obs.reshape(-1, neighbor_obs.shape[2], neighbor_obs.shape[3]),
                neighbor_masks.reshape(-1, neighbor_masks.shape[2], 1),
                neighbor_action_labels.reshape(-1, neighbor_action_labels.shape[2], neighbor_action_labels.shape[3]),
                neighbor_malicious.reshape(-1, neighbor_malicious.shape[2], 1),
                neighbor_channel_masks.reshape(-1, neighbor_channel_masks.shape[2], 1),
                neighbor_channel_noise.reshape(-1, neighbor_channel_noise.shape[2], neighbor_channel_noise.shape[3]),
            )

        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.available_actions[step]),
        )

        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        actions_env = [actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

        if self.algorithm_name == "rvita":
            return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, neighbor_obs, neighbor_masks, neighbor_idx, neighbor_malicious, neighbor_channel_masks, neighbor_channel_noise
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        if self.algorithm_name == "rvita":
            obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, neighbor_obs, neighbor_masks, neighbor_actions, neighbor_malicious, neighbor_channel_masks, neighbor_channel_noise = data
        else:
            obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=-1)
        if np.any(dones_env):
            for done, info in zip(dones_env, infos):
                if done:
                    self.env_infos["goal"].append(info["score_reward"])
                    self.env_infos["win_rate"].append(1 if info["score_reward"] > 0 else 0)
                    self.env_infos["steps"].append(info["max_steps"] - info["steps_left"])

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        if self.algorithm_name == "rvita":
            self.buffer.insert(
                share_obs=obs,
                obs=obs,
                rnn_states_actor=rnn_states,
                rnn_states_critic=rnn_states_critic,
                actions=actions,
                action_log_probs=action_log_probs,
                value_preds=values,
                rewards=rewards,
                masks=masks,
                active_masks=active_masks,
                neighbor_obs=neighbor_obs,
                neighbor_actions=neighbor_actions,
                neighbor_masks=neighbor_masks,
                neighbor_malicious=neighbor_malicious,
                neighbor_channel_masks=neighbor_channel_masks,
                neighbor_channel_noise=neighbor_channel_noise,
            )
            self._vita_update_meta(infos, dones_env=dones_env)
        else:
            self.buffer.insert(
                share_obs=obs,
                obs=obs,
                rnn_states_actor=rnn_states,
                rnn_states_critic=rnn_states_critic,
                actions=actions,
                action_log_probs=action_log_probs,
                value_preds=values,
                rewards=rewards,
                masks=masks,
                active_masks=active_masks,
            )

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    if wandb is None:
                        raise ImportError("wandb is required when `--use_wandb` is enabled. Please `pip install wandb`.")
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_obs = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        num_done = 0
        eval_goals = np.zeros(self.all_args.eval_episodes)
        eval_win_rates = np.zeros(self.all_args.eval_episodes)
        eval_steps = np.zeros(self.all_args.eval_episodes)
        step = 0
        quo = self.all_args.eval_episodes // self.n_eval_rollout_threads
        rem = self.all_args.eval_episodes % self.n_eval_rollout_threads
        done_episodes_per_thread = np.zeros(self.n_eval_rollout_threads, dtype=int)
        eval_episodes_per_thread = done_episodes_per_thread + quo
        eval_episodes_per_thread[:rem] += 1
        unfinished_thread = done_episodes_per_thread != eval_episodes_per_thread

        eval_positions = None
        eval_alive = None
        eval_prev_actions = None
        eval_comm_malicious_senders = None
        if self.algorithm_name == "rvita":
            eval_positions = np.zeros((self.n_eval_rollout_threads, self.num_agents, 2), dtype=np.float32)
            eval_alive = np.ones((self.n_eval_rollout_threads, self.num_agents), dtype=np.float32)
            eval_comm_malicious_senders = self._vita_sample_comm_malicious_senders(self.n_eval_rollout_threads, self.num_agents)

        while num_done < self.all_args.eval_episodes and step < self.episode_length:
            self.trainer.prep_rollout()
            if self.algorithm_name == "rvita":
                neighbor_obs, neighbor_masks, neighbor_idx, neighbor_malicious = self._vita_compute_neighbors(
                    eval_obs,
                    positions=eval_positions,
                    alive=eval_alive,
                    comm_malicious_senders=eval_comm_malicious_senders,
                )
                neighbor_channel_masks, neighbor_channel_noise = self._vita_sample_channel_effects(neighbor_masks, neighbor_malicious)
                act_dim = int(self.eval_envs.action_space[0].n)
                if eval_prev_actions is not None:
                    neighbor_prev_actions = self._vita_gather_neighbor_actions(eval_prev_actions, neighbor_idx)
                else:
                    neighbor_prev_actions = np.zeros((neighbor_obs.shape[0], neighbor_obs.shape[1], neighbor_obs.shape[2], act_dim), dtype=np.float32)
                self.trainer.policy.set_step_context(
                    neighbor_obs.reshape(-1, neighbor_obs.shape[2], neighbor_obs.shape[3]),
                    neighbor_masks.reshape(-1, neighbor_masks.shape[2], 1),
                    neighbor_prev_actions.reshape(-1, neighbor_prev_actions.shape[2], neighbor_prev_actions.shape[3]),
                    neighbor_malicious.reshape(-1, neighbor_malicious.shape[2], 1),
                    neighbor_channel_masks.reshape(-1, neighbor_channel_masks.shape[2], 1),
                    neighbor_channel_noise.reshape(-1, neighbor_channel_noise.shape[2], neighbor_channel_noise.shape[3]),
                )

            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                np.concatenate(np.ones((self.n_eval_rollout_threads, self.num_agents, self.eval_envs.action_space[0].n), dtype=np.float32)),
                deterministic=self.all_args.eval_deterministic,
            )

            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            eval_prev_actions = eval_actions

            eval_actions_env = [eval_actions[idx, :, 0] for idx in range(self.n_eval_rollout_threads)]
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)

            eval_dones_env = np.all(eval_dones, axis=-1)
            if self.algorithm_name == "rvita":
                self._vita_update_meta(eval_infos, dones_env=eval_dones_env, positions=eval_positions, alive=eval_alive)
                if eval_comm_malicious_senders is not None:
                    idx = np.asarray(eval_dones_env, dtype=bool)
                    if idx.any():
                        sampled = self._vita_sample_comm_malicious_senders(int(idx.sum()), self.num_agents)
                        if sampled is None:
                            eval_comm_malicious_senders[idx] = False
                        else:
                            eval_comm_malicious_senders[idx] = sampled

            eval_dones_unfinished_env = eval_dones_env[unfinished_thread]
            if np.any(eval_dones_unfinished_env):
                for idx_env in range(self.n_eval_rollout_threads):
                    if unfinished_thread[idx_env] and eval_dones_env[idx_env]:
                        eval_goals[num_done] = eval_infos[idx_env]["score_reward"]
                        eval_win_rates[num_done] = 1 if eval_infos[idx_env]["score_reward"] > 0 else 0
                        eval_steps[num_done] = eval_infos[idx_env]["max_steps"] - eval_infos[idx_env]["steps_left"]
                        num_done += 1
                        done_episodes_per_thread[idx_env] += 1
            unfinished_thread = done_episodes_per_thread != eval_episodes_per_thread

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            step += 1

        eval_goal = np.mean(eval_goals)
        eval_win_rate = np.mean(eval_win_rates)
        eval_step = np.mean(eval_steps)

        print("eval expected goal is {}.".format(eval_goal))
        if self.use_wandb:
            if wandb is None:
                raise ImportError("wandb is required when `--use_wandb` is enabled. Please `pip install wandb`.")
            wandb.log({"eval_goal": eval_goal}, step=total_num_steps)
            wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
            wandb.log({"eval_step": eval_step}, step=total_num_steps)
        else:
            self.writter.add_scalars("eval_goal", {"expected_goal": eval_goal}, total_num_steps)
            self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
            self.writter.add_scalars("eval_step", {"expected_step": eval_step}, total_num_steps)

        update = int(total_num_steps // (self.episode_length * self.n_rollout_threads))
        self.log_json({
            "phase": "eval",
            "update": update,
            "total_env_steps": int(total_num_steps),
            "eval_goal": float(eval_goal),
            "eval_win_rate": float(eval_win_rate),
            "eval_step": float(eval_step),
        }, step=update)

    @torch.no_grad()
    def render(self):
        render_env = self.envs

        render_goals = np.zeros(self.all_args.render_episodes)
        for i_episode in range(self.all_args.render_episodes):
            render_obs = render_env.reset()
            render_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            render_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            if self.all_args.save_gifs:
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[0]["frame"]
                frames.append(image)

            render_dones = False
            while not np.any(render_dones):
                self.trainer.prep_rollout()
                if self.algorithm_name == "rvita":
                    raise NotImplementedError("render for rvita on football is not implemented in this repository.")
                render_actions, render_rnn_states = self.trainer.policy.act(
                    np.concatenate(render_obs),
                    np.concatenate(render_rnn_states),
                    np.concatenate(render_masks),
                    deterministic=True,
                )

                render_actions = np.array(np.split(_t2n(render_actions), self.n_rollout_threads))
                render_rnn_states = np.array(np.split(_t2n(render_rnn_states), self.n_rollout_threads))

                render_actions_env = [render_actions[idx, :, 0] for idx in range(self.n_rollout_threads)]
                render_obs, render_rewards, render_dones, render_infos = render_env.step(render_actions_env)

                if self.all_args.save_gifs:
                    image = render_infos[0]["frame"]
                    frames.append(image)

            render_goals[i_episode] = render_rewards[0, 0]
            print("goal in episode {}: {}".format(i_episode, render_rewards[0, 0]))

            if self.all_args.save_gifs:
                imageio.mimsave(
                    uri="{}/episode{}.gif".format(str(self.gif_dir), i_episode),
                    ims=frames,
                    format="GIF",
                    duration=self.all_args.ifi,
                )

        print("expected goal: {}".format(np.mean(render_goals)))
