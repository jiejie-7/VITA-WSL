import time
import numpy as np
import torch
from gym import spaces
try:
    from inspect import getfullargspec as getargspec
except ImportError:  # Python < 3
    from inspect import getargspec


def _sample_sender_mask(rng, n_agents, prob, mode="bernoulli", fixed_agent_id=0):
    if rng is None or n_agents <= 0:
        return None
    prob = float(max(0.0, min(1.0, prob)))
    if prob <= 0.0:
        return None

    mode = str(mode or "bernoulli").strip().lower()
    if mode in {"fixed", "fixed_agent", "fixed_sender"}:
        agent_id = int(fixed_agent_id)
        if agent_id < 0 or agent_id >= n_agents:
            agent_id = 0
        mask = np.zeros((n_agents,), dtype=bool)
        if prob >= 1.0 or rng.rand() < prob:
            mask[agent_id] = True
        return mask

    if mode in {"one", "one_per_episode", "episode", "sample_one", "random_one"}:
        mask = np.zeros((n_agents,), dtype=bool)
        if prob >= 1.0 or rng.rand() < prob:
            mask[int(rng.randint(0, n_agents))] = True
        return mask

    return (rng.rand(n_agents) < prob)

class GymWrapper(object):
    '''
    for multi-agent
    '''
    def __init__(self, env, args=None):
        self.env = env
        self.args = args
        self._noise_rng = np.random.RandomState()
        self._malicious_mask = None
        self._comm_malicious_mask = None
        self._noise_step = 0

        self.obs_noise_std = 0.0
        self.packet_drop_prob = 0.0
        self.malicious_agent_prob = 0.0
        self.malicious_obs_noise_scale = 0.0
        self.malicious_obs_mode = "add"
        self.noise_warmup_steps = 0
        self.start_at_full_noise = False
        self.obs_noise_normalize = False
        self.obs_noise_norm_eps = 1e-6
        self.comm_malicious_agent_prob = 0.0
        self.comm_malicious_mode = "bernoulli"
        self.comm_malicious_fixed_agent_id = 0
        self._obs_mean = None
        self._obs_M2 = None
        self._obs_count = 0
        if args is not None:
            self.obs_noise_std = float(getattr(args, 'obs_noise_std', 0.0))
            self.packet_drop_prob = float(getattr(args, 'packet_drop_prob', 0.0))
            self.malicious_agent_prob = float(getattr(args, 'malicious_agent_prob', 0.0))
            self.malicious_obs_noise_scale = float(getattr(args, 'malicious_obs_noise_scale', 0.0))
            self.malicious_obs_mode = str(getattr(args, 'malicious_obs_mode', 'add')).lower()
            self.noise_warmup_steps = int(max(0, getattr(args, 'noise_warmup_steps', 0)))
            self.obs_noise_normalize = bool(getattr(args, 'obs_noise_normalize', False))
            self.obs_noise_norm_eps = float(getattr(args, 'obs_noise_norm_eps', 1e-6))
            self.comm_malicious_agent_prob = float(getattr(args, 'comm_malicious_agent_prob', 0.0))
            self.comm_malicious_mode = str(getattr(args, 'comm_malicious_mode', 'bernoulli')).lower()
            self.comm_malicious_fixed_agent_id = int(getattr(args, 'comm_malicious_fixed_agent_id', 0))
            seed = getattr(args, 'seed', None)
            if seed is not None:
                try:
                    seed = int(seed)
                except Exception:
                    seed = None
            if seed is not None and seed >= 0:
                self._noise_rng = np.random.RandomState(seed)

    @property
    def observation_dim(self):
        '''
        for multi-agent, this is the obs per agent
        '''

        # tuple space
        if hasattr(self.env.observation_space, 'spaces'):
            total_obs_dim = 0
            for space in self.env.observation_space.spaces:
                if hasattr(self.env.action_space, 'shape'):
                    total_obs_dim += int(np.prod(space.shape))
                else: # Discrete
                    total_obs_dim += 1
            return total_obs_dim
        else:
            return int(np.prod(self.env.observation_space.shape))

    @property
    def num_actions(self):
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return int(self.env.action_space.nvec[0])
        elif hasattr(self.env.action_space, 'n'):
            # Discrete
            return self.env.action_space.n

    @property
    def dim_actions(self):
        # for multi-agent, this is the number of action per agent
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return self.env.action_space.shape[0]
            # return len(self.env.action_space.shape)
        elif hasattr(self.env.action_space, 'n'):
            # Discrete => only 1 action takes place at a time.
            return 1

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, epoch):
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            obs = self.env.reset(epoch)
        else:
            obs = self.env.reset()

        obs = self._flatten_obs_numpy(obs)
        self._sample_malicious_mask(obs)
        self._sample_comm_malicious_mask(obs)
        obs = self._apply_obs_noise(obs)
        self._last_noise_info = self._noise_info()
        obs = torch.from_numpy(obs).double()
        return obs

    def display(self):
        self.env.render()
        time.sleep(0.5)

    def end_display(self):
        self.env.exit_render()

    def step(self, action):
        # TODO: Modify all environments to take list of action
        # instead of doing this
        if self.dim_actions == 1:
            action = action[0]
        obs, r, done, info = self.env.step(action)
        self._noise_step += 1
        obs = self._flatten_obs_numpy(obs)
        obs = self._apply_obs_noise(obs)
        info = self._inject_noise_info(info)
        obs = torch.from_numpy(obs).double()
        return (obs, r, done, info)

    def reward_terminal(self):
        if hasattr(self.env, 'reward_terminal'):
            return self.env.reward_terminal()
        else:
            return np.zeros(1)

    def _flatten_obs_numpy(self, obs):
        if isinstance(obs, tuple):
            _obs=[]
            for agent in obs: #list/tuple of observations.
                ag_obs = []
                for obs_kind in agent:
                    ag_obs.append(np.array(obs_kind).flatten())
                _obs.append(np.concatenate(ag_obs))
            obs = np.stack(_obs)
        obs = np.asarray(obs, dtype=np.float32)
        obs = obs.reshape(1, -1, self.observation_dim)
        return obs

    def _flatten_obs(self, obs):
        obs = self._flatten_obs_numpy(obs)
        obs = torch.from_numpy(obs).double()
        return obs

    def _noise_coeff(self):
        warmup = int(self.noise_warmup_steps)
        if warmup <= 0:
            return 1.0
        return float(min(1.0, self._noise_step / float(warmup)))

    def _sample_malicious_mask(self, obs_arr):
        self._malicious_mask = None
        coeff = self._noise_coeff()
        eff_mal_prob = float(max(0.0, self.malicious_agent_prob) * coeff)
        if eff_mal_prob <= 0.0:
            return
        if obs_arr is None:
            return
        n_agents = obs_arr.shape[1] if obs_arr.ndim == 3 else obs_arr.shape[0]
        if n_agents <= 0:
            return
        self._malicious_mask = (self._noise_rng.rand(n_agents) < eff_mal_prob)

    def _sample_comm_malicious_mask(self, obs_arr):
        self._comm_malicious_mask = None
        coeff = self._noise_coeff()
        eff_mal_prob = float(max(0.0, self.comm_malicious_agent_prob) * coeff)
        if eff_mal_prob <= 0.0:
            return
        if obs_arr is None:
            return
        n_agents = obs_arr.shape[1] if obs_arr.ndim == 3 else obs_arr.shape[0]
        if n_agents <= 0:
            return
        self._comm_malicious_mask = _sample_sender_mask(
            self._noise_rng,
            n_agents,
            eff_mal_prob,
            mode=self.comm_malicious_mode,
            fixed_agent_id=self.comm_malicious_fixed_agent_id,
        )

    def _update_obs_stats(self, view):
        if not self.obs_noise_normalize:
            return
        arr = np.asarray(view, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.size == 0:
            return
        count = int(arr.shape[0])
        batch_mean = arr.mean(axis=0)
        batch_M2 = ((arr - batch_mean) ** 2).sum(axis=0)
        if self._obs_count <= 0 or self._obs_mean is None or self._obs_M2 is None:
            self._obs_mean = batch_mean
            self._obs_M2 = batch_M2
            self._obs_count = count
            return
        delta = batch_mean - self._obs_mean
        total = self._obs_count + count
        self._obs_mean = self._obs_mean + delta * (count / total)
        self._obs_M2 = self._obs_M2 + batch_M2 + (delta ** 2) * (self._obs_count * count / total)
        self._obs_count = total

    def _obs_std(self, obs_dim):
        if self._obs_count <= 1 or self._obs_mean is None or self._obs_M2 is None:
            return np.ones((obs_dim,), dtype=np.float32)
        var = self._obs_M2 / float(max(1, self._obs_count))
        std = np.sqrt(var)
        eps = float(max(0.0, self.obs_noise_norm_eps))
        if eps > 0.0:
            std = np.maximum(std, eps)
        return std.astype(np.float32)

    def _apply_obs_noise(self, obs):
        if obs is None:
            return obs
        coeff = self._noise_coeff()
        obs_noise_std = float(max(0.0, self.obs_noise_std) * coeff)
        packet_drop_prob = float(max(0.0, self.packet_drop_prob) * coeff)
        malicious_obs_noise_scale = float(max(0.0, self.malicious_obs_noise_scale) * coeff)
        if (
            obs_noise_std <= 0.0
            and packet_drop_prob <= 0.0
            and (self._malicious_mask is None or not self._malicious_mask.any() or malicious_obs_noise_scale <= 0.0)
        ):
            return obs

        obs_arr = np.asarray(obs, dtype=np.float32).copy()
        if obs_arr.ndim == 3 and obs_arr.shape[0] == 1:
            view = obs_arr[0]
        else:
            view = obs_arr

        std = None
        if self.obs_noise_normalize and (obs_noise_std > 0.0 or malicious_obs_noise_scale > 0.0):
            self._update_obs_stats(view)
            std = self._obs_std(view.shape[-1])
            if view.ndim == 2:
                std = std.reshape(1, -1)

        if obs_noise_std > 0.0:
            if std is None:
                view += self._noise_rng.normal(0.0, obs_noise_std, size=view.shape).astype(np.float32)
            else:
                view += self._noise_rng.normal(0.0, obs_noise_std, size=view.shape).astype(np.float32) * std
        if packet_drop_prob > 0.0 and view.ndim >= 2:
            drop_mask = self._noise_rng.rand(view.shape[0]) < packet_drop_prob
            view[drop_mask] = 0.0
        if (
            self._malicious_mask is not None
            and self._malicious_mask.any()
            and view.ndim >= 2
            and view.shape[0] >= self._malicious_mask.shape[0]
        ):
            base_std = obs_noise_std if obs_noise_std > 0.0 else 1.0
            scale = float(max(1e-6, base_std) * malicious_obs_noise_scale)
            mal = self._malicious_mask.astype(bool)
            noise = self._noise_rng.normal(0.0, scale, size=view[mal].shape).astype(np.float32)
            if std is not None:
                noise = noise * std
            if self.malicious_obs_mode == 'add':
                view[mal] = view[mal] + noise
            else:
                view[mal] = noise

        if obs_arr.ndim == 3 and obs_arr.shape[0] == 1:
            obs_arr[0] = view
        else:
            obs_arr = view
        return obs_arr

    def _noise_info(self):
        info = {}
        if self._malicious_mask is not None:
            info['malicious_mask'] = self._malicious_mask.copy()
        if self._comm_malicious_mask is not None:
            info['comm_malicious_mask'] = self._comm_malicious_mask.copy()
        return info

    def _inject_noise_info(self, info):
        if not isinstance(info, dict):
            info = {'env_info': info}
        info.update(self._noise_info())
        return info

    def get_noise_info(self):
        return self._noise_info()

    def get_stat(self):
        if hasattr(self.env, 'stat'):
            self.env.stat.pop('steps_taken', None)
            return self.env.stat
        else:
            return dict()


class SmacWrapper(object):
    """SMAC adapter with VITA-style observation noise."""

    def __init__(self, env, args=None):
        self.env = env
        self.args = args
        self.n_agents = int(getattr(env, "n_agents", 0))
        self.episode_limit = int(getattr(env, "episode_limit", 0))
        self._noise_rng = np.random.RandomState()
        self._malicious_mask = None
        self._comm_malicious_mask = None
        self._noise_step = 0
        self._stat = {}
        self._last_battles_won = 0.0
        self._last_battles_game = 0.0
        self._last_avail_actions = None

        self.obs_noise_std = 0.0
        self.packet_drop_prob = 0.0
        self.malicious_agent_prob = 0.0
        self.malicious_obs_noise_scale = 0.0
        self.malicious_obs_mode = "add"
        self.noise_warmup_steps = 0
        self.start_at_full_noise = False
        self.obs_noise_normalize = False
        self.obs_noise_norm_eps = 1e-6
        self.comm_malicious_agent_prob = 0.0
        self.comm_malicious_mode = "bernoulli"
        self.comm_malicious_fixed_agent_id = 0
        self._obs_mean = None
        self._obs_M2 = None
        self._obs_count = 0
        if args is not None:
            self.obs_noise_std = float(getattr(args, "obs_noise_std", 0.0))
            self.packet_drop_prob = float(getattr(args, "packet_drop_prob", 0.0))
            self.malicious_agent_prob = float(getattr(args, "malicious_agent_prob", 0.0))
            self.malicious_obs_noise_scale = float(getattr(args, "malicious_obs_noise_scale", 0.0))
            self.malicious_obs_mode = str(getattr(args, "malicious_obs_mode", "add")).lower()
            self.noise_warmup_steps = int(max(0, getattr(args, "noise_warmup_steps", 0)))
            self.start_at_full_noise = bool(getattr(args, "start_at_full_noise", False))
            self.obs_noise_normalize = bool(getattr(args, "obs_noise_normalize", False))
            self.obs_noise_norm_eps = float(getattr(args, "obs_noise_norm_eps", 1e-6))
            self.comm_malicious_agent_prob = float(getattr(args, "comm_malicious_agent_prob", 0.0))
            self.comm_malicious_mode = str(getattr(args, "comm_malicious_mode", "bernoulli")).lower()
            self.comm_malicious_fixed_agent_id = int(getattr(args, "comm_malicious_fixed_agent_id", 0))
            seed = getattr(args, "seed", None)
            if seed is not None:
                try:
                    seed = int(seed)
                except Exception:
                    seed = None
            if seed is not None and seed >= 0:
                self._noise_rng = np.random.RandomState(seed)

    @property
    def observation_dim(self):
        size = self.env.get_obs_size()
        if isinstance(size, (list, tuple, np.ndarray)):
            if len(size) == 0:
                return 0
            size = size[0]
            if isinstance(size, (list, tuple, np.ndarray)):
                if len(size) == 0:
                    return 0
                size = size[0]
        return int(size)

    @property
    def num_actions(self):
        return int(self.env.get_total_actions())

    @property
    def dim_actions(self):
        return 1

    @property
    def action_space(self):
        return spaces.Discrete(self.num_actions)

    @property
    def observation_space(self):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents, self.observation_dim),
            dtype=np.float32,
        )

    def reset(self, epoch=None):
        self._noise_step = 0
        if self.start_at_full_noise and self.noise_warmup_steps > 0:
            self._noise_step = self.noise_warmup_steps
        self._stat = {}
        reset_args = getargspec(self.env.reset).args
        if "epoch" in reset_args:
            obs, _, avail_actions = self.env.reset(epoch)
        else:
            obs, _, avail_actions = self.env.reset()
        self._last_avail_actions = avail_actions
        obs = self._flatten_obs_numpy(obs)
        self._sample_malicious_mask(obs)
        self._sample_comm_malicious_mask(obs)
        obs = self._apply_obs_noise(obs)
        self._last_noise_info = self._noise_info()
        obs = torch.from_numpy(obs).double()
        return obs

    def step(self, action):
        if isinstance(action, (list, tuple)) and len(action) == 1:
            action = action[0]
        action = np.asarray(action).reshape(-1)
        action = action.astype(np.int64).tolist()
        action, avail_actions, fix_count = self._sanitize_actions(action)

        obs, _, rewards, dones, infos, next_avail = self.env.step(action)
        self._last_avail_actions = next_avail
        self._noise_step += 1
        obs = self._flatten_obs_numpy(obs)
        obs = self._apply_obs_noise(obs)
        reward = self._format_rewards(rewards)
        done = bool(np.all(dones))
        info = self._merge_info(infos, next_avail)
        if fix_count:
            info["action_fix_count"] = fix_count
        if done:
            self._update_stat(infos)
        info = self._inject_noise_info(info)
        obs = torch.from_numpy(obs).double()
        return (obs, reward, done, info)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def display(self):
        if hasattr(self.env, "render"):
            self.env.render()

    def reward_terminal(self):
        return np.zeros(self.n_agents, dtype=np.float32)

    def get_noise_info(self):
        return self._noise_info()

    def get_stat(self):
        stat = dict(self._stat)
        self._stat = {}
        return stat

    def _flatten_obs_numpy(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        obs = obs.reshape(1, self.n_agents, self.observation_dim)
        return obs

    def _format_rewards(self, rewards):
        arr = np.asarray(rewards, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full((self.n_agents,), float(arr))
        elif arr.ndim == 1:
            if arr.size == 1:
                arr = np.full((self.n_agents,), float(arr[0]))
        else:
            arr = arr.reshape(self.n_agents, -1)
            if arr.shape[1] > 0:
                arr = arr[:, 0]
            else:
                arr = arr.reshape(self.n_agents)
        return arr

    def _merge_info(self, infos, avail_actions):
        info = {}
        if isinstance(infos, list) and infos:
            info.update(infos[0])
            if "alive_mask" in infos[0]:
                info["alive_mask"] = infos[0]["alive_mask"]
        if avail_actions is not None:
            info["avail_actions"] = avail_actions
        return info

    def _update_stat(self, infos):
        won = False
        battles_won = self._last_battles_won
        battles_game = self._last_battles_game
        if isinstance(infos, list) and infos:
            won = bool(infos[0].get("won", False))
            if "battles_won" in infos[0]:
                battles_won = float(infos[0]["battles_won"])
            if "battles_game" in infos[0]:
                battles_game = float(infos[0]["battles_game"])

        delta_won = max(0.0, battles_won - self._last_battles_won)
        delta_game = max(0.0, battles_game - self._last_battles_game)
        self._last_battles_won = battles_won
        self._last_battles_game = battles_game

        self._stat["battles_won"] = delta_won
        self._stat["battles_game"] = delta_game
        self._stat["success"] = 1.0 if won else 0.0
        self._stat["won"] = 1.0 if won else 0.0

    def _sanitize_actions(self, actions):
        fix_count = 0
        avail_actions = None
        try:
            avail_actions = self.env.get_avail_actions()
        except Exception:
            return actions, None, fix_count
        fixed = list(actions)
        for i, act in enumerate(actions):
            if i >= len(avail_actions):
                continue
            avail = np.asarray(avail_actions[i], dtype=np.float32)
            if act < 0 or act >= avail.size or avail[act] < 0.5:
                if avail.sum() > 0:
                    fixed[i] = int(np.argmax(avail))
                    fix_count += 1
                else:
                    fixed[i] = 0
                    fix_count += 1
        return fixed, avail_actions, fix_count

    def _noise_coeff(self):
        warmup = int(self.noise_warmup_steps)
        if warmup <= 0:
            return 1.0
        return float(min(1.0, self._noise_step / float(warmup)))

    def _sample_malicious_mask(self, obs_arr):
        self._malicious_mask = None
        coeff = self._noise_coeff()
        eff_mal_prob = float(max(0.0, self.malicious_agent_prob) * coeff)
        if eff_mal_prob <= 0.0:
            return
        if obs_arr is None:
            return
        n_agents = obs_arr.shape[1] if obs_arr.ndim == 3 else obs_arr.shape[0]
        if n_agents <= 0:
            return
        self._malicious_mask = (self._noise_rng.rand(n_agents) < eff_mal_prob)

    def _sample_comm_malicious_mask(self, obs_arr):
        self._comm_malicious_mask = None
        coeff = self._noise_coeff()
        eff_mal_prob = float(max(0.0, self.comm_malicious_agent_prob) * coeff)
        if eff_mal_prob <= 0.0:
            return
        if obs_arr is None:
            return
        n_agents = obs_arr.shape[1] if obs_arr.ndim == 3 else obs_arr.shape[0]
        if n_agents <= 0:
            return
        self._comm_malicious_mask = _sample_sender_mask(
            self._noise_rng,
            n_agents,
            eff_mal_prob,
            mode=self.comm_malicious_mode,
            fixed_agent_id=self.comm_malicious_fixed_agent_id,
        )

    def _update_obs_stats(self, view):
        if not self.obs_noise_normalize:
            return
        arr = np.asarray(view, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.size == 0:
            return
        count = int(arr.shape[0])
        batch_mean = arr.mean(axis=0)
        batch_M2 = ((arr - batch_mean) ** 2).sum(axis=0)
        if self._obs_count <= 0 or self._obs_mean is None or self._obs_M2 is None:
            self._obs_mean = batch_mean
            self._obs_M2 = batch_M2
            self._obs_count = count
            return
        delta = batch_mean - self._obs_mean
        total = self._obs_count + count
        self._obs_mean = self._obs_mean + delta * (count / total)
        self._obs_M2 = self._obs_M2 + batch_M2 + (delta ** 2) * (self._obs_count * count / total)
        self._obs_count = total

    def _obs_std(self, obs_dim):
        if self._obs_count <= 1 or self._obs_mean is None or self._obs_M2 is None:
            return np.ones((obs_dim,), dtype=np.float32)
        var = self._obs_M2 / float(max(1, self._obs_count))
        std = np.sqrt(var)
        eps = float(max(0.0, self.obs_noise_norm_eps))
        if eps > 0.0:
            std = np.maximum(std, eps)
        return std.astype(np.float32)

    def _apply_obs_noise(self, obs):
        if obs is None:
            return obs
        coeff = self._noise_coeff()
        obs_noise_std = float(max(0.0, self.obs_noise_std) * coeff)
        packet_drop_prob = float(max(0.0, self.packet_drop_prob) * coeff)
        malicious_obs_noise_scale = float(max(0.0, self.malicious_obs_noise_scale) * coeff)
        if (
            obs_noise_std <= 0.0
            and packet_drop_prob <= 0.0
            and (self._malicious_mask is None or not self._malicious_mask.any() or malicious_obs_noise_scale <= 0.0)
        ):
            return obs

        obs_arr = np.asarray(obs, dtype=np.float32).copy()
        if obs_arr.ndim == 3 and obs_arr.shape[0] == 1:
            view = obs_arr[0]
        else:
            view = obs_arr

        std = None
        if self.obs_noise_normalize and (obs_noise_std > 0.0 or malicious_obs_noise_scale > 0.0):
            self._update_obs_stats(view)
            std = self._obs_std(view.shape[-1])
            if view.ndim == 2:
                std = std.reshape(1, -1)

        if obs_noise_std > 0.0:
            if std is None:
                view += self._noise_rng.normal(0.0, obs_noise_std, size=view.shape).astype(np.float32)
            else:
                view += self._noise_rng.normal(0.0, obs_noise_std, size=view.shape).astype(np.float32) * std
        if packet_drop_prob > 0.0 and view.ndim >= 2:
            drop_mask = self._noise_rng.rand(view.shape[0]) < packet_drop_prob
            view[drop_mask] = 0.0
        if (
            self._malicious_mask is not None
            and self._malicious_mask.any()
            and view.ndim >= 2
            and view.shape[0] >= self._malicious_mask.shape[0]
        ):
            base_std = obs_noise_std if obs_noise_std > 0.0 else 1.0
            scale = float(max(1e-6, base_std) * malicious_obs_noise_scale)
            mal = self._malicious_mask.astype(bool)
            noise = self._noise_rng.normal(0.0, scale, size=view[mal].shape).astype(np.float32)
            if std is not None:
                noise = noise * std
            if self.malicious_obs_mode == "add":
                view[mal] = view[mal] + noise
            else:
                view[mal] = noise

        if obs_arr.ndim == 3 and obs_arr.shape[0] == 1:
            obs_arr[0] = view
        else:
            obs_arr = view
        return obs_arr

    def _noise_info(self):
        info = {}
        if self._malicious_mask is not None:
            info["malicious_mask"] = self._malicious_mask.copy()
        if self._comm_malicious_mask is not None:
            info["comm_malicious_mask"] = self._comm_malicious_mask.copy()
        if self._last_avail_actions is not None:
            info["avail_actions"] = self._last_avail_actions
        return info

    def _inject_noise_info(self, info):
        if not isinstance(info, dict):
            info = {"env_info": info}
        info.update(self._noise_info())
        return info
