#!/usr/bin/env python
import sys
import os
try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None
import socket
import random
from contextlib import closing
try:
    import setproctitle  # type: ignore
except ImportError:  # pragma: no cover
    setproctitle = None
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

"""Train script for SMAC."""

def _env_float(name: str, default: float = 0.0) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value)


def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
        return True


def _pick_sc2_port_base(total_ports: int, *, step: int, host: str = "127.0.0.1") -> int:
    start = int(os.environ.get("ONPOLICY_SC2_PORT_START", "12000"))
    end = int(os.environ.get("ONPOLICY_SC2_PORT_END", "60000"))
    preferred = os.environ.get("ONPOLICY_SC2_BASE_PORT")

    total_ports = max(1, int(total_ports))
    step = max(1, int(step))

    max_base = end - step * (total_ports + 1)
    if max_base <= start:
        raise ValueError(f"Invalid SC2 port range: start={start}, end={end}, step={step}, total={total_ports}")

    if preferred is not None:
        base = int(preferred)
        ports = [base + i * step for i in range(total_ports)]
        if all(_is_port_free(p, host=host) for p in ports):
            return base
        raise RuntimeError(f"ONPOLICY_SC2_BASE_PORT={base} not available (ports {ports[:5]}...)")

    for _ in range(256):
        base = random.randrange(start, max_base)
        base -= base % step
        ports = [base + i * step for i in range(total_ports)]
        if all(_is_port_free(p, host=host) for p in ports):
            return base

    # Fall back to a deterministic base even if we can't pre-validate availability.
    return start


def _ensure_sc2_ports(all_args) -> None:
    if getattr(all_args, "_sc2_port_base", None) is not None:
        return

    step = int(os.environ.get("ONPOLICY_SC2_PORT_STEP", "10"))
    n_train = int(getattr(all_args, "n_rollout_threads", 1))
    n_eval = int(getattr(all_args, "n_eval_rollout_threads", 0)) if getattr(all_args, "use_eval", False) else 0
    base = _pick_sc2_port_base(n_train + n_eval, step=step)

    all_args._sc2_port_base = int(base)
    all_args._sc2_port_step = int(step)


class NoisyEnvWrapper:
    """Applies observation noise / packet drop / malicious observation perturbations.

    Controlled via env vars (inherited by subprocess vec envs):
      - ONPOLICY_OBS_NOISE_STD
      - ONPOLICY_PACKET_DROP_PROB
      - ONPOLICY_MALICIOUS_AGENT_PROB
      - ONPOLICY_MALICIOUS_OBS_NOISE_SCALE
      - ONPOLICY_MALICIOUS_OBS_MODE
      - ONPOLICY_NOISE_WARMUP_STEPS
      - ONPOLICY_REWARD_MULT

    Defaults are no-op (0/1), so the clean "official baseline" remains unchanged.
    """

    def __init__(
        self,
        env,
        *,
        obs_noise_std: float = 0.0,
        packet_drop_prob: float = 0.0,
        malicious_agent_prob: float = 0.0,
        malicious_obs_noise_scale: float = 3.0,
        malicious_obs_mode: str = "replace",
        noise_warmup_steps: int = 0,
        start_at_full_noise: bool = False,
        reward_mult: float = 1.0,
    ):
        self.env = env
        self.obs_noise_std = float(obs_noise_std)
        self.packet_drop_prob = float(packet_drop_prob)
        self.malicious_agent_prob = float(malicious_agent_prob)
        self.malicious_obs_noise_scale = float(malicious_obs_noise_scale)
        self.malicious_obs_mode = str(malicious_obs_mode).lower()
        self.noise_warmup_steps = int(max(0, noise_warmup_steps))
        self.reward_mult = float(reward_mult)

        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space

        self.n_agents = getattr(env, "n_agents", len(getattr(env, "action_space", [])))
        self._rng = np.random.RandomState()
        self._malicious_mask = None
        self._noise_step = int(self.noise_warmup_steps) if start_at_full_noise else 0

    def _noise_coeff(self) -> float:
        warmup = int(self.noise_warmup_steps)
        if warmup <= 0:
            return 1.0
        return float(min(1.0, self._noise_step / float(warmup)))

    def seed(self, seed):
        try:
            self._rng = np.random.RandomState(int(seed))
        except Exception:
            self._rng = np.random.RandomState()
        return self.env.seed(seed)

    def get_malicious_mask(self):
        if self._malicious_mask is None:
            return None
        return self._malicious_mask.copy()

    def reset(self):
        obs, share_obs, available_actions = self.env.reset()
        self._malicious_mask = None
        coeff = self._noise_coeff()
        eff_mal_prob = float(max(0.0, self.malicious_agent_prob) * coeff)
        if eff_mal_prob > 0.0 and self.n_agents > 0:
            self._malicious_mask = (self._rng.rand(self.n_agents) < eff_mal_prob)
        obs = self._apply_obs_noise(obs)
        return obs, share_obs, available_actions

    def step(self, actions):
        obs, share_obs, rewards, dones, infos, available_actions = self.env.step(actions)
        self._noise_step += 1
        obs = self._apply_obs_noise(obs)
        if self.reward_mult != 1.0:
            rewards = (np.asarray(rewards, dtype=np.float32) * self.reward_mult).tolist()
        return obs, share_obs, rewards, dones, infos, available_actions

    def close(self):
        return self.env.close()

    def render(self, *args, **kwargs):
        return getattr(self.env, "render")(*args, **kwargs)

    def _apply_obs_noise(self, obs):
        if self.obs_noise_std <= 0.0 and self.packet_drop_prob <= 0.0 and self.malicious_agent_prob <= 0.0:
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
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_noise_std > 0.0:
            obs_arr = obs_arr + self._rng.normal(0.0, obs_noise_std, size=obs_arr.shape).astype(np.float32)
        if packet_drop_prob > 0.0 and obs_arr.ndim >= 2:
            drop_mask = self._rng.rand(obs_arr.shape[0]) < packet_drop_prob
            obs_arr[drop_mask] = 0.0
        if (
            self._malicious_mask is not None
            and self._malicious_mask.any()
            and obs_arr.ndim >= 2
            and obs_arr.shape[0] >= self._malicious_mask.shape[0]
        ):
            base_std = obs_noise_std if obs_noise_std > 0.0 else 1.0
            std = float(max(1e-6, base_std) * malicious_obs_noise_scale)
            mal = self._malicious_mask.astype(bool)
            noise = self._rng.normal(0.0, std, size=obs_arr[mal].shape).astype(np.float32)
            if self.malicious_obs_mode == "add":
                obs_arr[mal] = obs_arr[mal] + noise
            else:
                obs_arr[mal] = noise
        return obs_arr

    def __getattr__(self, item):
        return getattr(self.env, item)


def maybe_wrap_noise(env, *, start_at_full_noise: bool = False):
    obs_noise_std = _env_float("ONPOLICY_OBS_NOISE_STD", 0.0)
    packet_drop_prob = _env_float("ONPOLICY_PACKET_DROP_PROB", 0.0)
    malicious_agent_prob = _env_float("ONPOLICY_MALICIOUS_AGENT_PROB", 0.0)
    malicious_obs_noise_scale = _env_float("ONPOLICY_MALICIOUS_OBS_NOISE_SCALE", 3.0)
    malicious_obs_mode = _env_str("ONPOLICY_MALICIOUS_OBS_MODE", "replace").lower()
    noise_warmup_steps = int(_env_float("ONPOLICY_NOISE_WARMUP_STEPS", 0.0))
    reward_mult = _env_float("ONPOLICY_REWARD_MULT", 1.0)
    if (
        obs_noise_std <= 0.0
        and packet_drop_prob <= 0.0
        and malicious_agent_prob <= 0.0
        and noise_warmup_steps <= 0
        and reward_mult == 1.0
    ):
        return env
    return NoisyEnvWrapper(
        env,
        obs_noise_std=obs_noise_std,
        packet_drop_prob=packet_drop_prob,
        malicious_agent_prob=malicious_agent_prob,
        malicious_obs_noise_scale=malicious_obs_noise_scale,
        malicious_obs_mode=malicious_obs_mode,
        noise_warmup_steps=noise_warmup_steps,
        start_at_full_noise=start_at_full_noise,
        reward_mult=reward_mult,
    )


def parse_smacv2_distribution(args):
    units = args.units.split('v')
    distribution_config = {
        "n_units": int(units[0]),
        "n_enemies": int(units[1]),
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "map_x": 32,
            "map_y": 32,
        }
    }
    if 'protoss' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            "unit_types": ["stalker", "zealot", "colossus"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        }
    elif 'zerg' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            "unit_types": ["zergling", "baneling", "hydralisk"],
            "weights": [0.45, 0.1, 0.45],
            "observe": True,
        } 
    elif 'terran' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        } 
    return distribution_config

def make_train_env(all_args):
    if all_args.env_name == "StarCraft2":
        _ensure_sc2_ports(all_args)

    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env
                env = StarCraft2Env(all_args)
                env._sc2_port = all_args._sc2_port_base + rank * all_args._sc2_port_step
                env = maybe_wrap_noise(env)
            elif all_args.env_name == "StarCraft2v2":
                from onpolicy.envs.starcraft2.SMACv2_modified import SMACv2
                env = SMACv2(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            elif all_args.env_name == "SMAC":
                from onpolicy.envs.starcraft2.SMAC import SMAC
                env = SMAC(map_name=all_args.map_name)
            elif all_args.env_name == "SMACv2":
                from onpolicy.envs.starcraft2.SMACv2 import SMACv2
                env = SMACv2(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    if all_args.env_name == "StarCraft2":
        _ensure_sc2_ports(all_args)

    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env
                env = StarCraft2Env(all_args)
                offset = int(getattr(all_args, "n_rollout_threads", 1))
                env._sc2_port = all_args._sc2_port_base + (offset + rank) * all_args._sc2_port_step
                env = maybe_wrap_noise(env, start_at_full_noise=True)
            elif all_args.env_name == "StarCraft2v2":
                from onpolicy.envs.starcraft2.SMACv2_modified import SMACv2
                env = SMACv2(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            elif all_args.env_name == "SMAC":
                from onpolicy.envs.starcraft2.SMAC import SMAC
                env = SMAC(map_name=all_args.map_name)
            elif all_args.env_name == "SMACv2":
                from onpolicy.envs.starcraft2.SMACv2 import SMACv2
                env = SMACv2(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m',
                        help="Which smac map to run on")
    parser.add_argument('--units', type=str, default='10v10') # for smac v2
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name in ("rmappo", "rvita"):
        print(f"u are choosing to use {all_args.algorithm_name}, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mat" or all_args.algorithm_name == "mat_dec":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    elif all_args.algorithm_name == "happo"  or all_args.algorithm_name == "hatrpo":
        # can or cannot use recurrent network?
        print("using", all_args.algorithm_name, 'without recurrent network')
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir_override = os.environ.get("ONPOLICY_RUN_DIR")
    if run_dir_override:
        run_dir = Path(run_dir_override)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        results_root = os.environ.get("ONPOLICY_RESULTS_DIR")
        if results_root:
            run_dir = Path(results_root) / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
        else:
            run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    if all_args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is required when `--use_wandb` is enabled. Please `pip install wandb`.")
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) + "_" + 
                              str(all_args.units) +
                              "_seed" + str(all_args.seed),
                        #  group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
        all_args = wandb.config # for wandb sweep
    else:
        if not run_dir_override:
            if not run_dir.exists():
                curr_run = 'run1'
            else:
                exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                                 str(folder.name).startswith('run')]
                if len(exst_run_nums) == 0:
                    curr_run = 'run1'
                else:
                    curr_run = 'run%i' % (max(exst_run_nums) + 1)
            run_dir = run_dir / curr_run
            if not run_dir.exists():
                os.makedirs(str(run_dir))

    if setproctitle is not None:
        setproctitle.setproctitle(
            str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
                all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    if all_args.env_name == "SMAC":
        from smac.env.starcraft2.maps import get_map_params
        num_agents = get_map_params(all_args.map_name)["n_agents"]
    elif all_args.env_name == 'StarCraft2':
        from onpolicy.envs.starcraft2.smac_maps import get_map_params
        num_agents = get_map_params(all_args.map_name)["n_agents"]
    elif all_args.env_name == "SMACv2" or all_args.env_name == 'StarCraft2v2':
        from smacv2.env.starcraft2.maps import get_map_params
        num_agents = parse_smacv2_distribution(all_args)['n_units']

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    if all_args.algorithm_name == "rvita" and not all_args.share_policy:
        raise ValueError("rvita requires `share_policy=True` (shared runner) in this repository.")

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.smac_runner import SMACRunner as Runner
    else:
        from onpolicy.runner.separated.smac_runner import SMACRunner as Runner

    if all_args.algorithm_name == "happo" or all_args.algorithm_name == "hatrpo":
        from onpolicy.runner.separated.smac_runner import SMACRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

    if hasattr(runner, "close_json"):
        runner.close_json()


if __name__ == "__main__":
    main(sys.argv[1:])
