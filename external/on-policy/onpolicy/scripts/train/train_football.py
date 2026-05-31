#!/usr/bin/env python
# python standard libraries
import os
from pathlib import Path
import sys
import socket

# third-party packages
import numpy as np
import setproctitle
import torch
try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

# code repository sub-packages
from onpolicy.config import get_config
from onpolicy.envs.football.Football_Env import FootballEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None else str(value)


class NoisyFootballEnvWrapper:
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
        obs = self.env.reset()
        self._malicious_mask = None
        coeff = self._noise_coeff()
        eff_mal_prob = float(max(0.0, self.malicious_agent_prob) * coeff)
        if eff_mal_prob > 0.0 and self.n_agents > 0:
            self._malicious_mask = self._rng.rand(self.n_agents) < eff_mal_prob
        return self._apply_obs_noise(obs)

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        self._noise_step += 1
        obs = self._apply_obs_noise(obs)
        if self.reward_mult != 1.0:
            rewards = np.asarray(rewards, dtype=np.float32) * self.reward_mult
        return obs, rewards, dones, infos

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
    return NoisyFootballEnvWrapper(
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


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Football":
                env = FootballEnv(all_args)
                env = maybe_wrap_noise(env)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Football":
                env = FootballEnv(all_args)
                env = maybe_wrap_noise(env, start_at_full_noise=True)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str,
                        default="academy_3_vs_1_with_keeper",
                        help="which scenario to run on.")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="number of controlled players.")
    parser.add_argument("--representation", type=str, default="simple115v2",
                        choices=["simple115v2", "extracted", "pixels_gray", "pixels"],
                        help="representation used to build the observation.")
    parser.add_argument("--rewards", type=str, default="scoring",
                        help="comma separated list of rewards to be added.")
    parser.add_argument("--smm_width", type=int, default=96,
                        help="width of super minimap.")
    parser.add_argument("--smm_height", type=int, default=72,
                        help="height of super minimap.")
    parser.add_argument("--remove_redundancy", action="store_true",
                        default=False,
                        help="by default False. If True, remove redundancy features")
    parser.add_argument("--zero_feature", action="store_true",
                        default=False,
                        help="by default False. If True, replace -1 by 0")
    parser.add_argument("--eval_deterministic", action="store_false",
                        default=True,
                        help="by default True. If False, sample action according to probability")
    parser.add_argument("--share_reward", action='store_false',
                        default=True,
                        help="by default true. If false, use different reward for each agent.")

    parser.add_argument("--save_videos", action="store_true", default=False,
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--video_dir", type=str, default="",
                        help="directory to save videos.")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name in ("rmappo", "rvita"):
        print(f"u are choosing to use {all_args.algorithm_name}, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False. Note that GRF is a fully observed game, so ippo is rmappo.")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

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
            run_dir = Path(results_root) / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
        else:
            run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    if all_args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is required when `--use_wandb` is enabled. Please `pip install wandb`.")
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name="-".join([
                            all_args.algorithm_name,
                            all_args.experiment_name,
                            "seed" + str(all_args.seed)
                         ]),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir_override:
            if not run_dir.exists():
                curr_run = 'run1'
            else:
                exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
                if len(exst_run_nums) == 0:
                    curr_run = 'run1'
                else:
                    curr_run = 'run%i' % (max(exst_run_nums) + 1)
            run_dir = run_dir / curr_run
            if not run_dir.exists():
                os.makedirs(str(run_dir))

    setproctitle.setproctitle("-".join([
        all_args.env_name,
        all_args.scenario_name,
        all_args.algorithm_name,
        all_args.experiment_name
    ]) + "@" + all_args.user_name)

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    if all_args.algorithm_name == "rvita" and not all_args.share_policy:
        raise ValueError("rvita requires `share_policy=True` (shared runner) in this repository.")

    if all_args.share_policy:
        from onpolicy.runner.shared.football_runner import FootballRunner as Runner
    else:
        from onpolicy.runner.separated.football_runner import FootballRunner as Runner

    runner = Runner(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        export_fn = getattr(runner.writter, "export_scalars_to_json", None)
        if callable(export_fn):
            export_fn(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

    if hasattr(runner, "close_json"):
        runner.close_json()


if __name__ == "__main__":
    main(sys.argv[1:])
