from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ONPOLICY_ROOT = REPO_ROOT / "external" / "on-policy"
if str(ONPOLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(ONPOLICY_ROOT))

from src.utils import load_config, set_seed  # noqa: E402
from src.algorithms.onpolicy_baseline import build_onpolicy_smac_args  # noqa: E402
from onpolicy.config import get_config  # noqa: E402
from onpolicy.scripts.train.train_smac import make_eval_env  # noqa: E402
from onpolicy.algorithms.r_vita.algorithm.rVITAPolicy import R_VITAPolicy  # noqa: E402
from onpolicy.runner.shared.smac_runner import SMACRunner  # noqa: E402


def _parse_onpolicy_args(cli_args: list[str]):
    parser = get_config()
    parser.add_argument("--map_name", type=str, default="3m")
    parser.add_argument("--units", type=str, default="10v10")
    parser.add_argument("--add_move_state", action="store_true", default=False)
    parser.add_argument("--add_local_obs", action="store_true", default=False)
    parser.add_argument("--add_distance_state", action="store_true", default=False)
    parser.add_argument("--add_enemy_action_state", action="store_true", default=False)
    parser.add_argument("--add_agent_id", action="store_true", default=False)
    parser.add_argument("--add_visible_state", action="store_true", default=False)
    parser.add_argument("--add_xy_state", action="store_true", default=False)
    parser.add_argument("--use_state_agent", action="store_false", default=True)
    parser.add_argument("--use_mustalive", action="store_false", default=True)
    parser.add_argument("--add_center_xy", action="store_false", default=True)
    return parser.parse_known_args(cli_args)[0]


def _setup_args(cfg: dict[str, Any], config_path: Path, model_dir: Path, eval_episodes: int):
    cli_args = build_onpolicy_smac_args(cfg, config_path=config_path)
    cli_args += [
        "--model_dir",
        str(model_dir),
        "--use_eval",
        "--eval_episodes",
        str(eval_episodes),
        "--n_eval_rollout_threads",
        "1",
        "--n_rollout_threads",
        "1",
    ]
    args = _parse_onpolicy_args(cli_args)
    args.use_wandb = False
    args.use_render = False
    args.n_eval_rollout_threads = 1
    args.n_rollout_threads = 1
    return args


def _unit_type_labels(env) -> list[str]:
    if getattr(env, "map_type", "") != "MMM":
        return [f"agent{i}" for i in range(env.n_agents)]
    labels = []
    for agent_id in range(env.n_agents):
        unit = env.get_unit_by_id(agent_id)
        if unit.unit_type == env.marauder_id:
            prefix = "Marauder"
        elif unit.unit_type == env.marine_id:
            prefix = "Marine"
        elif unit.unit_type == env.medivac_id:
            prefix = "Medivac"
        else:
            prefix = f"Type{env.get_unit_type_id(unit, True)}"
        labels.append(f"{agent_id}:{prefix}")
    return labels


def _save_heatmap(matrix: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val) and val > 0:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _extract_type_name(label: str) -> str:
    if ":" not in label:
        return label
    return label.split(":", 1)[1]


def _aggregate_by_type(matrix: np.ndarray, valid_count: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    type_names = [_extract_type_name(x) for x in labels]
    ordered_types: list[str] = []
    for name in type_names:
        if name not in ordered_types:
            ordered_types.append(name)

    n_types = len(ordered_types)
    sum_mat = np.zeros((n_types, n_types), dtype=np.float64)
    cnt_mat = np.zeros((n_types, n_types), dtype=np.float64)
    type_to_idx = {name: i for i, name in enumerate(ordered_types)}

    for i, src_type in enumerate(type_names):
        for j, dst_type in enumerate(type_names):
            ti = type_to_idx[src_type]
            tj = type_to_idx[dst_type]
            if valid_count[i, j] > 0:
                sum_mat[ti, tj] += float(matrix[i, j]) * float(valid_count[i, j])
                cnt_mat[ti, tj] += float(valid_count[i, j])

    mean_mat = np.divide(sum_mat, np.maximum(cnt_mat, 1.0), where=cnt_mat > 0)
    return mean_mat, ordered_types, cnt_mat


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VITA trust heatmap from a trained MMM2 checkpoint.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--model-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_args = _setup_args(cfg, config_path, model_dir, args.eval_episodes)
    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")
    envs = make_eval_env(all_args)

    if all_args.env_name == "StarCraft2":
        from onpolicy.envs.starcraft2.smac_maps import get_map_params
        num_agents = get_map_params(all_args.map_name)["n_agents"]
    else:
        raise ValueError(f"Unsupported env for export: {all_args.env_name}")

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": output_dir,
    }
    runner = SMACRunner(config)
    runner.model_dir = str(model_dir)
    runner.restore(model_dir)
    final_update = int((cfg.get("train") or {}).get("updates", 0))
    if hasattr(runner.trainer.policy, "update_schedules"):
        runner.trainer.policy.update_schedules(final_update)
    runner.trainer.policy.enable_act_debug = True
    runner.trainer.prep_rollout()

    obs, share_obs, available_actions = envs.reset()
    eval_rnn_states = np.zeros((1, num_agents, runner.recurrent_N, runner.hidden_size), dtype=np.float32)
    eval_masks = np.ones((1, num_agents, 1), dtype=np.float32)
    eval_positions = np.zeros((1, num_agents, 2), dtype=np.float32)
    eval_alive = np.ones((1, num_agents), dtype=np.float32)
    eval_prev_actions = None
    eval_comm_malicious_senders = runner._vita_sample_comm_malicious_senders(1, num_agents)

    trust_sum = np.zeros((num_agents, num_agents), dtype=np.float64)
    alloc_sum = np.zeros((num_agents, num_agents), dtype=np.float64)
    valid_count = np.zeros((num_agents, num_agents), dtype=np.float64)

    episode_count = 0
    while episode_count < int(args.eval_episodes):
        neighbor_obs, neighbor_masks, neighbor_idx, neighbor_malicious = runner._vita_compute_neighbors(
            obs, positions=eval_positions, alive=eval_alive, comm_malicious_senders=eval_comm_malicious_senders
        )
        neighbor_channel_masks, neighbor_channel_noise = runner._vita_sample_channel_effects(neighbor_masks, neighbor_malicious)
        act_dim = int(envs.action_space[0].n)
        if eval_prev_actions is not None:
            neighbor_prev_actions = runner._vita_gather_neighbor_actions(eval_prev_actions, neighbor_idx)
        else:
            neighbor_prev_actions = np.zeros(
                (neighbor_obs.shape[0], neighbor_obs.shape[1], neighbor_obs.shape[2], act_dim),
                dtype=np.float32,
            )
        runner.trainer.policy.set_step_context(
            neighbor_obs.reshape(-1, neighbor_obs.shape[2], neighbor_obs.shape[3]),
            neighbor_masks.reshape(-1, neighbor_masks.shape[2], 1),
            neighbor_prev_actions.reshape(-1, neighbor_prev_actions.shape[2], neighbor_prev_actions.shape[3]),
            neighbor_malicious.reshape(-1, neighbor_malicious.shape[2], 1),
            neighbor_channel_masks.reshape(-1, neighbor_channel_masks.shape[2], 1),
            neighbor_channel_noise.reshape(-1, neighbor_channel_noise.shape[2], neighbor_channel_noise.shape[3]),
        )
        eval_actions, eval_rnn_states = runner.trainer.policy.act(
            np.concatenate(obs),
            np.concatenate(eval_rnn_states),
            np.concatenate(eval_masks),
            np.concatenate(available_actions),
            deterministic=True,
        )
        debug = getattr(runner.trainer.policy, "last_act_debug", {}) or {}
        missing = [key for key in ("edge_reliability", "edge_allocation", "edge_valid_mask") if key not in debug]
        if missing:
            raise RuntimeError(
                "VITA edge debug was not exported by policy.act(); missing keys: "
                + ", ".join(missing)
            )
        edge_rel = np.asarray(debug.get("edge_reliability"), dtype=np.float32).reshape(1, num_agents, -1, 1)
        edge_alloc = np.asarray(debug.get("edge_allocation"), dtype=np.float32).reshape(1, num_agents, -1, 1)
        edge_valid = np.asarray(debug.get("edge_valid_mask"), dtype=np.float32).reshape(1, num_agents, -1, 1)

        for src in range(num_agents):
            for k in range(neighbor_idx.shape[2]):
                dst = int(neighbor_idx[0, src, k])
                if edge_valid[0, src, k, 0] > 0.5:
                    valid_count[src, dst] += 1.0
                    trust_sum[src, dst] += float(edge_rel[0, src, k, 0])
                    alloc_sum[src, dst] += float(edge_alloc[0, src, k, 0])

        eval_actions_np = eval_actions.detach().cpu().numpy()
        eval_actions = np.array(np.split(eval_actions_np, 1))
        eval_prev_actions = eval_actions
        obs, share_obs, rewards, dones, infos, available_actions = envs.step(eval_actions)
        dones_env = np.all(dones, axis=1)
        runner._vita_update_meta(infos, dones_env=dones_env, positions=eval_positions, alive=eval_alive)

        eval_rnn_states[dones_env == True] = 0.0
        eval_masks = np.ones((1, num_agents, 1), dtype=np.float32)
        eval_masks[dones_env == True] = 0.0
        if dones_env[0]:
            episode_count += 1
            sampled = runner._vita_sample_comm_malicious_senders(1, num_agents)
            eval_comm_malicious_senders = sampled if sampled is not None else np.zeros((1, num_agents), dtype=bool)
            eval_prev_actions = None

    env = envs.envs[0] if hasattr(envs, "envs") else None
    labels = _unit_type_labels(env) if env is not None else [f"agent{i}" for i in range(num_agents)]
    trust_mean = np.divide(trust_sum, np.maximum(valid_count, 1.0), where=valid_count > 0)
    alloc_mean = np.divide(alloc_sum, np.maximum(valid_count, 1.0), where=valid_count > 0)

    np.save(output_dir / "trust_matrix.npy", trust_mean)
    np.save(output_dir / "allocation_matrix.npy", alloc_mean)
    np.save(output_dir / "valid_count.npy", valid_count)
    (output_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "trust_matrix.json").write_text(json.dumps(trust_mean.tolist(), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "allocation_matrix.json").write_text(json.dumps(alloc_mean.tolist(), ensure_ascii=False, indent=2), encoding="utf-8")

    _save_heatmap(trust_mean, labels, output_dir / "trust_heatmap.png", "VITA Trust Matrix on MMM2")
    _save_heatmap(alloc_mean, labels, output_dir / "allocation_heatmap.png", "VITA Allocation Matrix on MMM2")

    trust_type_mean, type_labels, type_valid = _aggregate_by_type(trust_mean, valid_count, labels)
    alloc_type_mean, _, _ = _aggregate_by_type(alloc_mean, valid_count, labels)
    np.save(output_dir / "trust_type_matrix.npy", trust_type_mean)
    np.save(output_dir / "allocation_type_matrix.npy", alloc_type_mean)
    np.save(output_dir / "type_valid_count.npy", type_valid)
    (output_dir / "type_labels.json").write_text(json.dumps(type_labels, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "trust_type_matrix.json").write_text(json.dumps(trust_type_mean.tolist(), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "allocation_type_matrix.json").write_text(json.dumps(alloc_type_mean.tolist(), ensure_ascii=False, indent=2), encoding="utf-8")
    _save_heatmap(trust_type_mean, type_labels, output_dir / "trust_type_heatmap.png", "VITA Type-Level Trust Matrix on MMM2")
    _save_heatmap(alloc_type_mean, type_labels, output_dir / "allocation_type_heatmap.png", "VITA Type-Level Allocation Matrix on MMM2")

    envs.close()


if __name__ == "__main__":
    main()
