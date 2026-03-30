from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _external_onpolicy_root() -> Path:
    return _repo_root() / "external" / "on-policy"


def _as_int(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected int for '{name}', got: {value!r}") from exc


def _as_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected float for '{name}', got: {value!r}") from exc


def _flag_store_false(args: List[str], flag: str, desired: bool) -> None:
    """on-policy uses many `action='store_false', default=True` flags.

    If desired is False, we need to pass `--<flag>` to flip it to False.
    """
    if not desired:
        args.append(f"--{flag}")


def _flag_store_true(args: List[str], flag: str, desired: bool) -> None:
    if desired:
        args.append(f"--{flag}")


def build_onpolicy_smac_args(cfg: Dict[str, Any], *, config_path: Path) -> List[str]:
    env_cfg = cfg.get("env") or {}
    train_cfg = cfg.get("train") or {}
    model_cfg = cfg.get("model") or {}
    onpolicy_cfg = cfg.get("onpolicy") or {}

    map_name = env_cfg.get("map_name")
    if not map_name:
        raise ValueError("Missing `env.map_name` in config.")

    seed = _as_int(cfg.get("seed", 42), name="seed")

    num_envs = _as_int(env_cfg.get("num_envs", 1), name="env.num_envs")
    episode_length = _as_int(train_cfg.get("episode_length"), name="train.episode_length")

    updates = _as_int(train_cfg.get("updates"), name="train.updates")
    num_env_steps = _as_int(
        train_cfg.get("num_env_steps", updates * episode_length * num_envs),
        name="train.num_env_steps",
    )

    experiment_name = str(cfg.get("experiment_name") or cfg.get("experiment") or config_path.stem)

    top_algo = str(cfg.get("algorithm") or "").lower()
    default_algorithm = "rvita" if top_algo in {"vita", "rvita"} else "rmappo"
    algorithm_name = str(onpolicy_cfg.get("algorithm_name") or default_algorithm)
    env_name = str(onpolicy_cfg.get("env_name") or "StarCraft2")

    args: List[str] = []
    args += ["--env_name", env_name]
    args += ["--map_name", str(map_name)]
    args += ["--algorithm_name", algorithm_name]
    args += ["--experiment_name", experiment_name]
    args += ["--seed", str(seed)]

    hidden_size = _as_int(model_cfg.get("hidden_dim", 64), name="model.hidden_dim")
    args += ["--hidden_size", str(hidden_size)]

    args += ["--n_training_threads", str(_as_int(train_cfg.get("n_training_threads", 1), name="train.n_training_threads"))]
    args += ["--n_rollout_threads", str(num_envs)]
    args += ["--n_eval_rollout_threads", str(_as_int(train_cfg.get("n_eval_rollout_threads", 1), name="train.n_eval_rollout_threads"))]

    args += ["--episode_length", str(episode_length)]
    args += ["--num_env_steps", str(num_env_steps)]

    lr = _as_float(train_cfg.get("lr", 5e-4), name="train.lr")
    critic_lr = _as_float(train_cfg.get("critic_lr", lr), name="train.critic_lr")
    args += ["--lr", str(lr)]
    args += ["--critic_lr", str(critic_lr)]
    args += ["--opti_eps", str(_as_float(train_cfg.get("opti_eps", 1e-5), name="train.opti_eps"))]
    args += ["--weight_decay", str(_as_float(train_cfg.get("weight_decay", 0.0), name="train.weight_decay"))]

    args += ["--gamma", str(_as_float(train_cfg.get("gamma", 0.99), name="train.gamma"))]
    args += ["--gae_lambda", str(_as_float(train_cfg.get("gae_lambda", 0.95), name="train.gae_lambda"))]
    args += ["--clip_param", str(_as_float(train_cfg.get("clip_param", 0.2), name="train.clip_param"))]

    ppo_epoch = _as_int(train_cfg.get("ppo_epoch", train_cfg.get("ppo_epochs", 15)), name="train.ppo_epochs")
    args += ["--ppo_epoch", str(ppo_epoch)]
    args += ["--num_mini_batch", str(_as_int(train_cfg.get("num_mini_batch", 1), name="train.num_mini_batch"))]
    args += ["--entropy_coef", str(_as_float(train_cfg.get("entropy_coef", 0.01), name="train.entropy_coef"))]
    args += ["--value_loss_coef", str(_as_float(train_cfg.get("value_loss_coef", 0.5), name="train.value_loss_coef"))]
    args += ["--max_grad_norm", str(_as_float(train_cfg.get("max_grad_norm", 10.0), name="train.max_grad_norm"))]
    args += ["--huber_delta", str(_as_float(train_cfg.get("huber_delta", 10.0), name="train.huber_delta"))]

    args += ["--data_chunk_length", str(_as_int(train_cfg.get("data_chunk_length", 10), name="train.data_chunk_length"))]

    if algorithm_name == "rvita":
        history_length = _as_int(model_cfg.get("history_length", 1), name="model.history_length")
        args += ["--stacked_frames", str(history_length)]
        _flag_store_true(args, "use_stacked_frames", history_length > 1)

        args += ["--vita_latent_dim", str(_as_int(model_cfg.get("latent_dim", 64), name="model.latent_dim"))]
        args += ["--vita_trust_gamma", str(_as_float(model_cfg.get("trust_gamma", 1.0), name="model.trust_gamma"))]
        args += ["--vita_kl_beta", str(_as_float(model_cfg.get("kl_beta", 1e-3), name="model.kl_beta"))]
        args += ["--vita_kl_free_bits", str(_as_float(model_cfg.get("kl_free_bits", 0.0), name="model.kl_free_bits"))]
        args += ["--vita_attn_bias_coef", str(_as_float(model_cfg.get("attn_bias_coef", 1.0), name="model.attn_bias_coef"))]
        args += ["--vita_trust_lambda", str(_as_float(model_cfg.get("trust_lambda", 0.1), name="model.trust_lambda"))]
        args += ["--vita_trust_malicious_weight", str(_as_float(model_cfg.get("trust_malicious_weight", 1.0), name="model.trust_malicious_weight"))]
        args += ["--vita_trust_gate_floor", str(_as_float(model_cfg.get("trust_gate_floor", 0.0), name="model.trust_gate_floor"))]
        args += ["--vita_comm_dropout", str(_as_float(model_cfg.get("comm_dropout", 0.1), name="model.comm_dropout"))]
        args += ["--vita_comm_sight_range", str(_as_float(model_cfg.get("comm_sight_range", 0.0), name="model.comm_sight_range"))]
        args += ["--vita_max_neighbors", str(_as_int(model_cfg.get("max_neighbors", 4), name="model.max_neighbors"))]
        if not bool(model_cfg.get("enable_trust", True)):
            _flag_store_true(args, "vita_disable_trust", True)
        if not bool(model_cfg.get("enable_kl", True)):
            _flag_store_true(args, "vita_disable_kl", True)
        if bool(model_cfg.get("attention_only", False)):
            _flag_store_true(args, "vita_attention_only", True)

        args += ["--vita_trust_warmup_updates", str(_as_int(train_cfg.get("trust_warmup_updates", 0), name="train.trust_warmup_updates"))]
        args += ["--vita_trust_delay_updates", str(_as_int(train_cfg.get("trust_delay_updates", 0), name="train.trust_delay_updates"))]
        args += ["--vita_kl_warmup_updates", str(_as_int(train_cfg.get("kl_warmup_updates", 0), name="train.kl_warmup_updates"))]
        args += ["--vita_kl_delay_updates", str(_as_int(train_cfg.get("kl_delay_updates", 0), name="train.kl_delay_updates"))]
        args += ["--vita_comm_delay_updates", str(_as_int(train_cfg.get("comm_delay_updates", 0), name="train.comm_delay_updates"))]
        args += ["--vita_comm_warmup_updates", str(_as_int(train_cfg.get("comm_warmup_updates", 0), name="train.comm_warmup_updates"))]
        args += ["--vita_comm_full_warmup_updates", str(_as_int(train_cfg.get("comm_full_warmup_updates", 0), name="train.comm_full_warmup_updates"))]

    args += ["--log_interval", str(_as_int(train_cfg.get("log_interval", 1), name="train.log_interval"))]
    args += ["--save_interval", str(_as_int(train_cfg.get("save_interval", 1), name="train.save_interval"))]

    _flag_store_false(args, "use_wandb", bool(onpolicy_cfg.get("use_wandb", False)))
    _flag_store_false(args, "cuda", bool(cfg.get("cuda", True)))

    _flag_store_true(args, "use_proper_time_limits", bool(train_cfg.get("use_proper_time_limits", False)))
    _flag_store_false(args, "use_clipped_value_loss", bool(train_cfg.get("use_clipped_value_loss", True)))
    _flag_store_false(args, "use_huber_loss", bool(train_cfg.get("use_huber_loss", True)))
    _flag_store_false(args, "use_value_active_masks", bool(train_cfg.get("use_value_active_masks", True)))
    _flag_store_false(args, "use_policy_active_masks", bool(train_cfg.get("use_policy_active_masks", True)))
    _flag_store_false(args, "use_max_grad_norm", bool(train_cfg.get("use_max_grad_norm", True)))
    _flag_store_false(args, "use_gae", bool(train_cfg.get("use_gae", True)))
    _flag_store_false(args, "use_valuenorm", bool(train_cfg.get("use_valuenorm", True)))

    eval_interval = _as_int(train_cfg.get("eval_interval_updates", 0), name="train.eval_interval_updates")
    eval_episodes = _as_int(train_cfg.get("eval_episodes", 0), name="train.eval_episodes")
    if eval_interval > 0 and eval_episodes > 0:
        _flag_store_true(args, "use_eval", True)
        args += ["--eval_interval", str(eval_interval)]
        args += ["--eval_episodes", str(eval_episodes)]

    return args


def build_onpolicy_football_args(cfg: Dict[str, Any], *, config_path: Path) -> List[str]:
    env_cfg = cfg.get("env") or {}
    train_cfg = cfg.get("train") or {}
    model_cfg = cfg.get("model") or {}
    onpolicy_cfg = cfg.get("onpolicy") or {}

    scenario_name = env_cfg.get("scenario_name")
    if not scenario_name:
        raise ValueError("Missing `env.scenario_name` in config.")

    seed = _as_int(cfg.get("seed", 42), name="seed")
    num_agents = _as_int(env_cfg.get("num_agents"), name="env.num_agents")
    num_envs = _as_int(env_cfg.get("num_envs", 1), name="env.num_envs")
    episode_length = _as_int(train_cfg.get("episode_length", 200), name="train.episode_length")
    updates = _as_int(train_cfg.get("updates"), name="train.updates")
    num_env_steps = _as_int(
        train_cfg.get("num_env_steps", updates * episode_length * num_envs),
        name="train.num_env_steps",
    )

    experiment_name = str(cfg.get("experiment_name") or cfg.get("experiment") or config_path.stem)

    top_algo = str(cfg.get("algorithm") or "").lower()
    default_algorithm = "rvita" if top_algo in {"vita", "rvita"} else "rmappo"
    algorithm_name = str(onpolicy_cfg.get("algorithm_name") or default_algorithm)
    env_name = str(onpolicy_cfg.get("env_name") or "Football")

    interval_base = episode_length * num_envs

    def _interval_in_steps(step_key: str, update_key: str, default_updates: int) -> int:
        if step_key in train_cfg:
            return _as_int(train_cfg.get(step_key), name=f"train.{step_key}")
        return _as_int(train_cfg.get(update_key, default_updates), name=f"train.{update_key}") * interval_base

    args: List[str] = []
    args += ["--env_name", env_name]
    args += ["--scenario_name", str(scenario_name)]
    args += ["--algorithm_name", algorithm_name]
    args += ["--experiment_name", experiment_name]
    args += ["--seed", str(seed)]
    args += ["--num_agents", str(num_agents)]

    args += ["--representation", str(env_cfg.get("representation", "simple115v2"))]
    args += ["--rewards", str(env_cfg.get("rewards", "scoring"))]
    args += ["--smm_width", str(_as_int(env_cfg.get("smm_width", 96), name="env.smm_width"))]
    args += ["--smm_height", str(_as_int(env_cfg.get("smm_height", 72), name="env.smm_height"))]
    _flag_store_true(args, "remove_redundancy", bool(env_cfg.get("remove_redundancy", False)))
    _flag_store_true(args, "zero_feature", bool(env_cfg.get("zero_feature", False)))
    _flag_store_false(args, "share_reward", bool(env_cfg.get("share_reward", True)))
    _flag_store_false(args, "eval_deterministic", bool(env_cfg.get("eval_deterministic", True)))

    hidden_size = _as_int(model_cfg.get("hidden_dim", 64), name="model.hidden_dim")
    args += ["--hidden_size", str(hidden_size)]

    args += ["--n_training_threads", str(_as_int(train_cfg.get("n_training_threads", 1), name="train.n_training_threads"))]
    args += ["--n_rollout_threads", str(num_envs)]
    args += ["--n_eval_rollout_threads", str(_as_int(train_cfg.get("n_eval_rollout_threads", 1), name="train.n_eval_rollout_threads"))]

    args += ["--episode_length", str(episode_length)]
    args += ["--num_env_steps", str(num_env_steps)]

    lr = _as_float(train_cfg.get("lr", 5e-4), name="train.lr")
    critic_lr = _as_float(train_cfg.get("critic_lr", lr), name="train.critic_lr")
    args += ["--lr", str(lr)]
    args += ["--critic_lr", str(critic_lr)]
    args += ["--opti_eps", str(_as_float(train_cfg.get("opti_eps", 1e-5), name="train.opti_eps"))]
    args += ["--weight_decay", str(_as_float(train_cfg.get("weight_decay", 0.0), name="train.weight_decay"))]
    args += ["--gamma", str(_as_float(train_cfg.get("gamma", 0.99), name="train.gamma"))]
    args += ["--gae_lambda", str(_as_float(train_cfg.get("gae_lambda", 0.95), name="train.gae_lambda"))]
    args += ["--clip_param", str(_as_float(train_cfg.get("clip_param", 0.2), name="train.clip_param"))]
    args += ["--ppo_epoch", str(_as_int(train_cfg.get("ppo_epoch", train_cfg.get("ppo_epochs", 15)), name="train.ppo_epochs"))]
    args += ["--num_mini_batch", str(_as_int(train_cfg.get("num_mini_batch", 1), name="train.num_mini_batch"))]
    args += ["--entropy_coef", str(_as_float(train_cfg.get("entropy_coef", 0.01), name="train.entropy_coef"))]
    args += ["--value_loss_coef", str(_as_float(train_cfg.get("value_loss_coef", 0.5), name="train.value_loss_coef"))]
    args += ["--max_grad_norm", str(_as_float(train_cfg.get("max_grad_norm", 10.0), name="train.max_grad_norm"))]
    args += ["--huber_delta", str(_as_float(train_cfg.get("huber_delta", 10.0), name="train.huber_delta"))]
    args += ["--data_chunk_length", str(_as_int(train_cfg.get("data_chunk_length", 10), name="train.data_chunk_length"))]

    if algorithm_name == "rvita":
        history_length = _as_int(model_cfg.get("history_length", 1), name="model.history_length")
        args += ["--stacked_frames", str(history_length)]
        _flag_store_true(args, "use_stacked_frames", history_length > 1)

        args += ["--vita_latent_dim", str(_as_int(model_cfg.get("latent_dim", 64), name="model.latent_dim"))]
        args += ["--vita_trust_gamma", str(_as_float(model_cfg.get("trust_gamma", 1.0), name="model.trust_gamma"))]
        args += ["--vita_kl_beta", str(_as_float(model_cfg.get("kl_beta", 1e-3), name="model.kl_beta"))]
        args += ["--vita_kl_free_bits", str(_as_float(model_cfg.get("kl_free_bits", 0.0), name="model.kl_free_bits"))]
        args += ["--vita_attn_bias_coef", str(_as_float(model_cfg.get("attn_bias_coef", 1.0), name="model.attn_bias_coef"))]
        args += ["--vita_trust_lambda", str(_as_float(model_cfg.get("trust_lambda", 0.1), name="model.trust_lambda"))]
        args += ["--vita_trust_malicious_weight", str(_as_float(model_cfg.get("trust_malicious_weight", 1.0), name="model.trust_malicious_weight"))]
        args += ["--vita_trust_gate_floor", str(_as_float(model_cfg.get("trust_gate_floor", 0.0), name="model.trust_gate_floor"))]
        args += ["--vita_comm_dropout", str(_as_float(model_cfg.get("comm_dropout", 0.1), name="model.comm_dropout"))]
        args += ["--vita_comm_sight_range", str(_as_float(model_cfg.get("comm_sight_range", 0.0), name="model.comm_sight_range"))]
        args += ["--vita_max_neighbors", str(_as_int(model_cfg.get("max_neighbors", max(1, num_agents - 1)), name="model.max_neighbors"))]
        if not bool(model_cfg.get("enable_trust", True)):
            _flag_store_true(args, "vita_disable_trust", True)
        if not bool(model_cfg.get("enable_kl", True)):
            _flag_store_true(args, "vita_disable_kl", True)

        args += ["--vita_trust_warmup_updates", str(_as_int(train_cfg.get("trust_warmup_updates", 0), name="train.trust_warmup_updates"))]
        args += ["--vita_trust_delay_updates", str(_as_int(train_cfg.get("trust_delay_updates", 0), name="train.trust_delay_updates"))]
        args += ["--vita_kl_warmup_updates", str(_as_int(train_cfg.get("kl_warmup_updates", 0), name="train.kl_warmup_updates"))]
        args += ["--vita_kl_delay_updates", str(_as_int(train_cfg.get("kl_delay_updates", 0), name="train.kl_delay_updates"))]
        args += ["--vita_comm_delay_updates", str(_as_int(train_cfg.get("comm_delay_updates", 0), name="train.comm_delay_updates"))]
        args += ["--vita_comm_warmup_updates", str(_as_int(train_cfg.get("comm_warmup_updates", 0), name="train.comm_warmup_updates"))]
        args += ["--vita_comm_full_warmup_updates", str(_as_int(train_cfg.get("comm_full_warmup_updates", 0), name="train.comm_full_warmup_updates"))]

    args += ["--log_interval", str(_interval_in_steps("log_interval_steps", "log_interval_updates", 1))]
    args += ["--save_interval", str(_interval_in_steps("save_interval_steps", "save_interval_updates", max(1, updates // 10 or 1)))]

    _flag_store_false(args, "use_wandb", bool(onpolicy_cfg.get("use_wandb", False)))
    _flag_store_false(args, "cuda", bool(cfg.get("cuda", True)))

    _flag_store_true(args, "use_proper_time_limits", bool(train_cfg.get("use_proper_time_limits", False)))
    _flag_store_true(args, "use_linear_lr_decay", bool(train_cfg.get("use_linear_lr_decay", False)))
    _flag_store_false(args, "use_clipped_value_loss", bool(train_cfg.get("use_clipped_value_loss", True)))
    _flag_store_false(args, "use_huber_loss", bool(train_cfg.get("use_huber_loss", True)))
    _flag_store_false(args, "use_value_active_masks", bool(train_cfg.get("use_value_active_masks", True)))
    _flag_store_false(args, "use_policy_active_masks", bool(train_cfg.get("use_policy_active_masks", True)))
    _flag_store_false(args, "use_max_grad_norm", bool(train_cfg.get("use_max_grad_norm", True)))
    _flag_store_false(args, "use_gae", bool(train_cfg.get("use_gae", True)))
    _flag_store_false(args, "use_valuenorm", bool(train_cfg.get("use_valuenorm", True)))
    _flag_store_false(args, "use_centralized_V", bool(onpolicy_cfg.get("use_centralized_V", True)))
    _flag_store_false(args, "share_policy", bool(onpolicy_cfg.get("share_policy", True)))

    eval_interval_steps = 0
    if "eval_interval_steps" in train_cfg:
        eval_interval_steps = _as_int(train_cfg.get("eval_interval_steps"), name="train.eval_interval_steps")
    elif "eval_interval_updates" in train_cfg:
        eval_interval_steps = _as_int(train_cfg.get("eval_interval_updates"), name="train.eval_interval_updates") * interval_base
    eval_episodes = _as_int(train_cfg.get("eval_episodes", 0), name="train.eval_episodes")
    if eval_interval_steps > 0 and eval_episodes > 0:
        _flag_store_true(args, "use_eval", True)
        args += ["--eval_interval", str(eval_interval_steps)]
        args += ["--eval_episodes", str(eval_episodes)]

    return args


def _write_run_snapshot(run_dir: Path, config_path: Path, args: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config_dst = run_dir / "config.yaml"
    if config_dst.exists():
        config_dst = run_dir / f"config_{timestamp}.yaml"
    try:
        shutil.copy2(config_path, config_dst)
    except FileNotFoundError:
        config_dst = None

    args_dst = run_dir / "args.json"
    if args_dst.exists():
        args_dst = run_dir / f"args_{timestamp}.json"

    args_payload = {"argv": args, "config_path": str(config_path)}
    try:
        from onpolicy.config import get_config
        parser = get_config()
        env_name = None
        if "--env_name" in args:
            idx = args.index("--env_name")
            if idx + 1 < len(args):
                env_name = args[idx + 1]
        if env_name == "Football":
            from onpolicy.scripts.train import train_football
            all_args = train_football.parse_args(args, parser)
        else:
            from onpolicy.scripts.train import train_smac
            all_args = train_smac.parse_args(args, parser)
        args_payload["parsed"] = vars(all_args)
    except Exception:
        pass

    with args_dst.open("w", encoding="utf-8") as f:
        json.dump(args_payload, f, ensure_ascii=False, indent=2)

    try:
        log_path = run_dir / "train.log"
        meta = {
            "time": time.time(),
            "phase": "meta",
            "config_path": str(config_path),
            "config_snapshot": str(config_dst) if config_dst is not None else None,
            "args_snapshot": str(args_dst),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    except Exception:
        pass


def run_onpolicy_smac(cfg: Dict[str, Any], *, config_path: Path, run_dir: Path) -> None:
    """Run the upstream on-policy SMAC trainer (external/on-policy).

    This is intended for paper baselines: algorithm/runtime is the official implementation,
    while we only inject output directory + JSON-lines logging for compatibility.
    """
    external_root = _external_onpolicy_root()
    sys.path.insert(0, str(external_root))

    os.environ["ONPOLICY_RUN_DIR"] = str(run_dir)
    os.environ.setdefault("ONPOLICY_JSON_LOG", "1")
    env_cfg = cfg.get("env") or {}
    os.environ["ONPOLICY_OBS_NOISE_STD"] = str(float(env_cfg.get("obs_noise_std", 0.0)))
    os.environ["ONPOLICY_PACKET_DROP_PROB"] = str(float(env_cfg.get("packet_drop_prob", 0.0)))
    os.environ["ONPOLICY_MALICIOUS_AGENT_PROB"] = str(float(env_cfg.get("malicious_agent_prob", 0.0)))
    os.environ["ONPOLICY_MALICIOUS_OBS_NOISE_SCALE"] = str(float(env_cfg.get("malicious_obs_noise_scale", 3.0)))
    os.environ["ONPOLICY_MALICIOUS_OBS_MODE"] = str(env_cfg.get("malicious_obs_mode", "replace"))
    os.environ["ONPOLICY_NOISE_WARMUP_STEPS"] = str(int(env_cfg.get("noise_warmup_steps", 0)))
    os.environ["ONPOLICY_COMM_NOISE_STD"] = str(float(env_cfg.get("comm_noise_std", 0.0)))
    os.environ["ONPOLICY_COMM_PACKET_DROP_PROB"] = str(float(env_cfg.get("comm_packet_drop_prob", 0.0)))
    os.environ["ONPOLICY_COMM_MALICIOUS_AGENT_PROB"] = str(float(env_cfg.get("comm_malicious_agent_prob", 0.0)))
    os.environ["ONPOLICY_COMM_MALICIOUS_NOISE_SCALE"] = str(float(env_cfg.get("comm_malicious_noise_scale", 3.0)))
    os.environ["ONPOLICY_COMM_MALICIOUS_MODE"] = str(env_cfg.get("comm_malicious_mode", "bernoulli"))
    os.environ["ONPOLICY_COMM_MALICIOUS_FIXED_AGENT_ID"] = str(int(env_cfg.get("comm_malicious_fixed_agent_id", 0)))
    os.environ["ONPOLICY_REWARD_MULT"] = str(float(env_cfg.get("reward_scale", 1.0)))

    args = build_onpolicy_smac_args(cfg, config_path=config_path)

    _write_run_snapshot(run_dir, config_path, args)

    from onpolicy.scripts.train import train_smac

    train_smac.main(args)


def run_onpolicy_football(cfg: Dict[str, Any], *, config_path: Path, run_dir: Path) -> None:
    external_root = _external_onpolicy_root()
    sys.path.insert(0, str(external_root))

    os.environ["ONPOLICY_RUN_DIR"] = str(run_dir)
    os.environ.setdefault("ONPOLICY_JSON_LOG", "1")

    args = build_onpolicy_football_args(cfg, config_path=config_path)

    _write_run_snapshot(run_dir, config_path, args)

    from onpolicy.scripts.train import train_football

    train_football.main(args)
