from __future__ import annotations

from itertools import chain
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import update_linear_schedule

try:
    from src.vita import VITAAgent, VITAAgentConfig
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "VITA requires the local `src.vita` package to be importable. "
        "Run training from the repository root so `src/` is on PYTHONPATH."
    ) from exc


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


def _comm_schedule_coeff(elapsed: int, stage_one: int, stage_two: int) -> float:
    if elapsed <= 0:
        return 0.0
    if stage_one > 0 and elapsed <= stage_one:
        return min(0.5, 0.5 * elapsed / float(stage_one))
    base = 0.0 if stage_one <= 0 else 0.5
    if stage_two <= 0:
        return 1.0 if elapsed > stage_one else base
    extra = max(0.0, min(elapsed - max(stage_one, 0), stage_two))
    return min(1.0, max(base, 0.5) + 0.5 * (extra / float(stage_two)))


class R_VITAPolicy:
    """VITA policy wrapper compatible with the on-policy runner/buffer.

    Notes:
    - Uses the official RMAPPo rollout/update pipeline (masks, bad_masks, ValueNorm, etc).
    - Additional neighbor tensors are provided by the runner/buffer for VITA.
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device: torch.device = torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self._use_policy_active_masks = bool(getattr(args, "use_policy_active_masks", True))

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        if act_space.__class__.__name__ != "Discrete":
            raise NotImplementedError("R_VITAPolicy currently supports Discrete action spaces only.")

        obs_total_dim = int(obs_space[0]) if isinstance(obs_space, list) else int(obs_space.shape[0])
        state_total_dim = int(cent_obs_space[0]) if isinstance(cent_obs_space, list) else int(cent_obs_space.shape[0])

        self.use_stacked_frames = bool(getattr(args, "use_stacked_frames", False))
        self.history_length = int(getattr(args, "stacked_frames", 1) if self.use_stacked_frames else 1)
        self.history_length = max(1, self.history_length)
        if obs_total_dim % self.history_length != 0:
            raise ValueError(f"obs_dim={obs_total_dim} must be divisible by history_length={self.history_length}")
        if state_total_dim % self.history_length != 0:
            raise ValueError(f"state_dim={state_total_dim} must be divisible by history_length={self.history_length}")

        self.obs_dim = obs_total_dim // self.history_length
        self.state_dim = state_total_dim // self.history_length
        self.action_dim = int(act_space.n)

        agent_cfg = VITAAgentConfig(
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=int(getattr(args, "hidden_size", 64)),
            latent_dim=int(getattr(args, "vita_latent_dim", 64)),
            trust_gamma=float(getattr(args, "vita_trust_gamma", 1.0)),
            kl_beta=float(getattr(args, "vita_kl_beta", 1e-3)),
            kl_free_bits=float(getattr(args, "vita_kl_free_bits", 0.0)),
            trust_lambda=float(getattr(args, "vita_trust_lambda", 0.1)),
            trust_malicious_weight=float(getattr(args, "vita_trust_malicious_weight", 1.0)),
            trust_margin_weight=float(getattr(args, "vita_trust_margin_weight", 0.5)),
            trust_margin=float(getattr(args, "vita_trust_margin", 0.1)),
            trust_action_loss_weight=float(getattr(args, "vita_trust_action_loss_weight", 1.0)),
            trust_reliability_mix=float(getattr(args, "vita_trust_reliability_mix", 0.5)),
            trust_utility_mix=float(getattr(args, "vita_trust_utility_mix", 0.6)),
            trust_counterfactual_mix=float(getattr(args, "vita_trust_counterfactual_mix", 0.5)),
            trust_counterfactual_margin=float(getattr(args, "vita_trust_counterfactual_margin", 0.02)),
            trust_counterfactual_weight=float(getattr(args, "vita_trust_counterfactual_weight", 1.0)),
            trust_consistency_mix=float(getattr(args, "vita_trust_consistency_mix", 0.0)),
            trust_consistency_weight=float(getattr(args, "vita_trust_consistency_weight", 0.0)),
            trust_consistency_margin=float(getattr(args, "vita_trust_consistency_margin", 0.05)),
            trust_consistency_noise_std=float(getattr(args, "vita_trust_consistency_noise_std", 0.0)),
            max_neighbors=int(getattr(args, "vita_max_neighbors", 4)),
            comm_dropout=float(getattr(args, "vita_comm_dropout", 0.1)),
            enable_trust=not bool(getattr(args, "vita_disable_trust", False)),
            enable_kl=not bool(getattr(args, "vita_disable_kl", False)),
            vib_deterministic=bool(getattr(args, "vita_vib_deterministic", False)),
            attention_only=bool(getattr(args, "vita_attention_only", False)),
            trust_hard_topk=bool(getattr(args, "vita_trust_hard_topk", False)),
            trust_topk_k=int(getattr(args, "vita_trust_topk_k", 0)),
            trust_gate_floor=float(getattr(args, "vita_trust_gate_floor", 0.0)),
            trust_malicious_gate_coef=float(getattr(args, "vita_trust_malicious_gate_coef", 1.0)),
            allocation_sharpness=float(getattr(args, "vita_allocation_sharpness", 1.0)),
            allocation_floor=float(getattr(args, "vita_allocation_floor", 0.0)),
            trust_pair_product=bool(getattr(args, "vita_trust_pair_product", False)),
            trust_gate_product=bool(getattr(args, "vita_trust_gate_product", False)),
            trust_decouple_allocation=bool(getattr(args, "vita_trust_decouple_allocation", False)),
            trust_use_utility_for_gate=not bool(getattr(args, "vita_trust_disable_utility_gate", False)),
            trust_gate_threshold=float(getattr(args, "vita_trust_gate_threshold", 0.0)),
            attn_bias_coef=float(getattr(args, "vita_attn_bias_coef", 1.0)),
        )

        self.agent = VITAAgent(agent_cfg).to(device)
        # Expose actor/critic attributes for compatibility with on-policy runners (save/restore, mode switches).
        # VITA uses a single module with disjoint actor/critic parameter subsets.
        self.actor = self.agent
        self.critic = self.agent

        comm_params = []
        if self.agent.vib_gat is not None:
            comm_params = list(self.agent.vib_gat.parameters())

        trust_params = list(self.agent.trust_predictor.parameters()) if self.agent.cfg.enable_trust else []

        actor_params = list(
            chain(
                self.agent.actor_encoder.parameters(),
                self.agent.comm_encoder.parameters(),
                trust_params,
                comm_params,
                self.agent.residual.parameters(),
                self.agent.neighbor_norm.parameters(),
                self.agent.policy_head.parameters(),
            )
        )
        critic_params = list(
            chain(
                self.agent.critic_encoder.parameters(),
                self.agent.critic_mlp.parameters(),
                self.agent.value_head.parameters(),
            )
        )

        self.actor_optimizer = torch.optim.Adam(
            actor_params, lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_params, lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay
        )

        self._actor_params = actor_params
        self._critic_params = critic_params

        self._neighbor_obs: Optional[np.ndarray] = None
        self._neighbor_masks: Optional[np.ndarray] = None
        self._neighbor_actions: Optional[np.ndarray] = None
        self._neighbor_malicious: Optional[np.ndarray] = None
        self._neighbor_channel_masks: Optional[np.ndarray] = None
        self._neighbor_channel_noise: Optional[np.ndarray] = None

        self._vita_comm_delay = int(getattr(args, "vita_comm_delay_updates", 0))
        self._vita_comm_warmup = int(getattr(args, "vita_comm_warmup_updates", 0))
        self._vita_comm_full_warmup = int(getattr(args, "vita_comm_full_warmup_updates", 0))
        legacy_trust_delay = int(getattr(args, "vita_trust_delay_updates", 0))
        legacy_trust_warmup = int(getattr(args, "vita_trust_warmup_updates", 0))
        self._vita_trust_gate_delay = _resolve_schedule(
            getattr(args, "vita_trust_gate_delay_updates", -1),
            legacy_trust_delay,
        )
        self._vita_trust_gate_warmup = _resolve_schedule(
            getattr(args, "vita_trust_gate_warmup_updates", -1),
            legacy_trust_warmup,
        )

        self.update_schedules(0)

    def actor_parameters(self):
        return self._actor_params

    def critic_parameters(self):
        return self._critic_params

    def lr_decay(self, episode: int, episodes: int) -> None:
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def update_schedules(self, update: int) -> None:
        update = int(update)

        comm_elapsed = update - self._vita_comm_delay
        comm_coeff = _comm_schedule_coeff(comm_elapsed, self._vita_comm_warmup, self._vita_comm_full_warmup)
        self.agent.set_comm_strength(comm_coeff)
        self.agent.set_comm_enabled(comm_coeff > 0.0)

        trust_elapsed = update - self._vita_trust_gate_delay
        trust_gate = _schedule_coeff(trust_elapsed, self._vita_trust_gate_warmup)
        self.agent.set_trust_strength(trust_gate)

    def set_step_context(
        self,
        neighbor_obs: np.ndarray,
        neighbor_masks: np.ndarray,
        neighbor_actions: Optional[np.ndarray] = None,
        neighbor_malicious: Optional[np.ndarray] = None,
        neighbor_channel_masks: Optional[np.ndarray] = None,
        neighbor_channel_noise: Optional[np.ndarray] = None,
    ) -> None:
        self._neighbor_obs = neighbor_obs
        self._neighbor_masks = neighbor_masks
        self._neighbor_actions = neighbor_actions
        self._neighbor_malicious = neighbor_malicious
        self._neighbor_channel_masks = neighbor_channel_masks
        self._neighbor_channel_noise = neighbor_channel_noise

    def _reshape_obs_seq(self, obs_flat: torch.Tensor) -> torch.Tensor:
        if self.history_length == 1:
            return obs_flat.unsqueeze(1)
        frames = obs_flat.view(obs_flat.size(0), self.history_length, self.obs_dim)
        past = frames[:, :-1].flip(dims=[1])
        current = frames[:, -1:].contiguous()
        return torch.cat([past, current], dim=1)

    def _reshape_neighbor_seq(self, neighbor_obs_flat: torch.Tensor) -> torch.Tensor:
        if self.history_length == 1:
            return neighbor_obs_flat.unsqueeze(2)
        frames = neighbor_obs_flat.view(neighbor_obs_flat.size(0), neighbor_obs_flat.size(1), self.history_length, self.obs_dim)
        past = frames[:, :, :-1].flip(dims=[2])
        current = frames[:, :, -1:].contiguous()
        return torch.cat([past, current], dim=2)

    def _extract_current_state(self, cent_obs_flat: torch.Tensor) -> torch.Tensor:
        if self.history_length == 1:
            return cent_obs_flat
        frames = cent_obs_flat.view(cent_obs_flat.size(0), self.history_length, self.state_dim)
        return frames[:, -1].contiguous()

    def get_actions(
        self,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic: bool = False,
    ):
        if self._neighbor_obs is None or self._neighbor_masks is None:
            raise RuntimeError("VITA step context (neighbor_obs/masks) was not set before get_actions().")

        cent_obs = check(cent_obs).to(device=self.device, dtype=torch.float32)
        obs = check(obs).to(device=self.device, dtype=torch.float32)
        rnn_states_actor = check(rnn_states_actor).to(device=self.device, dtype=torch.float32)
        rnn_states_critic = check(rnn_states_critic).to(device=self.device, dtype=torch.float32)
        masks = check(masks).to(device=self.device, dtype=torch.float32)
        if available_actions is not None:
            available_actions = check(available_actions).to(device=self.device, dtype=torch.float32)

        neighbor_obs = check(self._neighbor_obs).to(device=self.device, dtype=torch.float32)
        neighbor_masks = check(self._neighbor_masks).to(device=self.device, dtype=torch.float32)
        neighbor_actions = None
        if self._neighbor_actions is not None:
            neighbor_actions = check(self._neighbor_actions).to(device=self.device, dtype=torch.float32)
        neighbor_malicious = None
        if self._neighbor_malicious is not None:
            neighbor_malicious = check(self._neighbor_malicious).to(device=self.device, dtype=torch.float32)
        neighbor_channel_masks = None
        if self._neighbor_channel_masks is not None:
            neighbor_channel_masks = check(self._neighbor_channel_masks).to(device=self.device, dtype=torch.float32)
        neighbor_channel_noise = None
        if self._neighbor_channel_noise is not None:
            neighbor_channel_noise = check(self._neighbor_channel_noise).to(device=self.device, dtype=torch.float32)

        if rnn_states_actor.dim() != 3 or rnn_states_actor.size(1) != 1:
            raise ValueError("VITA expects rnn_states_actor to have shape [B, 1, H].")
        if rnn_states_critic.dim() != 3 or rnn_states_critic.size(1) != 1:
            raise ValueError("VITA expects rnn_states_critic to have shape [B, 1, H].")

        obs_seq = self._reshape_obs_seq(obs)
        state = self._extract_current_state(cent_obs)
        neighbor_seq = self._reshape_neighbor_seq(neighbor_obs)

        out = self.agent.act(
            obs_seq,
            state,
            neighbor_seq,
            neighbor_masks,
            neighbor_channel_masks,
            neighbor_channel_noise,
            neighbor_actions,
            neighbor_malicious,
            rnn_states_actor[:, 0],
            rnn_states_critic[:, 0],
            masks,
            available_actions,
            deterministic=deterministic,
        )

        values = out["values"]
        actions = out["actions"]
        action_log_probs = out["log_probs"]
        next_rnn_actor = out["next_actor_state"].unsqueeze(1)
        next_rnn_critic = out["next_critic_state"].unsqueeze(1)
        return values, actions, action_log_probs, next_rnn_actor, next_rnn_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        cent_obs = check(cent_obs).to(device=self.device, dtype=torch.float32)
        rnn_states_critic = check(rnn_states_critic).to(device=self.device, dtype=torch.float32)
        masks = check(masks).to(device=self.device, dtype=torch.float32)

        state = self._extract_current_state(cent_obs)
        values, _ = self.agent.get_values(state, rnn_states_critic[:, 0], masks)
        return values

    def evaluate_actions_vita(
        self,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        neighbor_obs,
        neighbor_actions,
        neighbor_masks,
        neighbor_channel_masks,
        neighbor_channel_noise,
        neighbor_malicious,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        cent_obs = check(cent_obs).to(device=self.device, dtype=torch.float32)
        obs = check(obs).to(device=self.device, dtype=torch.float32)
        rnn_states_actor = check(rnn_states_actor).to(device=self.device, dtype=torch.float32)
        rnn_states_critic = check(rnn_states_critic).to(device=self.device, dtype=torch.float32)
        neighbor_obs = check(neighbor_obs).to(device=self.device, dtype=torch.float32)
        neighbor_actions = check(neighbor_actions).to(device=self.device, dtype=torch.float32)
        neighbor_masks = check(neighbor_masks).to(device=self.device, dtype=torch.float32)
        neighbor_channel_masks = None if neighbor_channel_masks is None else check(neighbor_channel_masks).to(device=self.device, dtype=torch.float32)
        neighbor_channel_noise = None if neighbor_channel_noise is None else check(neighbor_channel_noise).to(device=self.device, dtype=torch.float32)
        neighbor_malicious = None if neighbor_malicious is None else check(neighbor_malicious).to(device=self.device, dtype=torch.float32)
        action = check(action).to(device=self.device)
        masks = check(masks).to(device=self.device, dtype=torch.float32)
        if available_actions is not None:
            available_actions = check(available_actions).to(device=self.device, dtype=torch.float32)
        if active_masks is not None:
            active_masks = check(active_masks).to(device=self.device, dtype=torch.float32)

        if rnn_states_actor.dim() != 3 or rnn_states_actor.size(1) != 1:
            raise ValueError("VITA expects rnn_states_actor to have shape [B, 1, H].")
        if rnn_states_critic.dim() != 3 or rnn_states_critic.size(1) != 1:
            raise ValueError("VITA expects rnn_states_critic to have shape [B, 1, H].")

        if obs.size(0) == rnn_states_actor.size(0):
            obs_seq = self._reshape_obs_seq(obs)
            state = self._extract_current_state(cent_obs)
            neighbor_seq = self._reshape_neighbor_seq(neighbor_obs)

            eval_out = self.agent.evaluate_actions(
                obs_seq,
                state,
                neighbor_seq,
                neighbor_masks,
                neighbor_channel_masks,
                neighbor_channel_noise,
                neighbor_actions,
                neighbor_malicious,
                action,
                rnn_states_actor[:, 0],
                rnn_states_critic[:, 0],
                masks,
                available_actions,
            )
            log_probs = eval_out["log_probs"]
            entropy = eval_out["entropy"]
            values = eval_out["values"]
            kl_loss = eval_out["kl_loss"]
            trust_loss = eval_out["trust_loss"]
            debug = {
                "kl_raw": float(eval_out["kl_raw"].item()),
                "trust_score_mean": float(eval_out["trust_score_mean"].item()),
                "trust_score_p10": float(eval_out["trust_score_p10"].item()),
                "trust_score_p50": float(eval_out["trust_score_p50"].item()),
                "trust_score_p90": float(eval_out["trust_score_p90"].item()),
                "trust_gate_ratio": float(eval_out["trust_gate_ratio"].item()),
                "comm_valid_neighbors": float(eval_out["comm_valid_neighbors"].item()),
                "comm_kept_neighbors": float(eval_out["comm_kept_neighbors"].item()),
                "comm_strength": float(eval_out["comm_strength"].item()),
                "trust_strength": float(eval_out["trust_strength"].item()),
                "comm_enabled": float(eval_out["comm_enabled"].item()),
                "comm_malicious_ratio": float(eval_out["comm_malicious_ratio"].item()),
                "trust_malicious_gap": float(eval_out["trust_malicious_gap"].item()),
                "consistency_score_mean": float(eval_out["consistency_score_mean"].item()),
                "consistency_score_mal": float(eval_out["consistency_score_mal"].item()),
                "consistency_score_clean": float(eval_out["consistency_score_clean"].item()),
                "consistency_malicious_gap": float(eval_out["consistency_malicious_gap"].item()),
                "residual_gate_mean": float(eval_out["residual_gate_mean"].item()),
                "residual_gate_max": float(eval_out["residual_gate_max"].item()),
                "residual_comm_ratio": float(eval_out["residual_comm_ratio"].item()),
            }

            if self._use_policy_active_masks and active_masks is not None:
                denom = active_masks.sum().clamp_min(1.0)
                dist_entropy = (entropy * active_masks).sum() / denom
            else:
                dist_entropy = entropy.mean()
            return values, log_probs, dist_entropy, kl_loss, trust_loss, debug

        # Sequence mode (T*N, ...) with initial hidden states (N, ...)
        N = rnn_states_actor.size(0)
        if obs.size(0) % N != 0:
            raise ValueError(f"obs batch {obs.size(0)} not divisible by N={N} for recurrent evaluation.")
        T = int(obs.size(0) // N)

        obs = obs.view(T, N, -1)
        cent_obs = cent_obs.view(T, N, -1)
        action = action.view(T, N, -1)
        masks = masks.view(T, N, -1)
        if available_actions is not None:
            available_actions = available_actions.view(T, N, -1)
        if active_masks is not None:
            active_masks = active_masks.view(T, N, -1)
        neighbor_obs = neighbor_obs.view(T, N, neighbor_obs.size(1), -1)
        neighbor_actions = neighbor_actions.view(T, N, neighbor_actions.size(1), -1)
        neighbor_masks = neighbor_masks.view(T, N, neighbor_masks.size(1), -1)
        if neighbor_channel_masks is not None:
            neighbor_channel_masks = neighbor_channel_masks.view(T, N, neighbor_channel_masks.size(1), -1)
        if neighbor_channel_noise is not None:
            neighbor_channel_noise = neighbor_channel_noise.view(T, N, neighbor_channel_noise.size(1), neighbor_channel_noise.size(2))
        if neighbor_malicious is not None:
            neighbor_malicious = neighbor_malicious.view(T, N, neighbor_malicious.size(1), -1)

        log_probs_list = []
        entropy_list = []
        values_list = []
        kl_list = []
        trust_list = []
        kl_raw_list = []
        trust_mean_list = []
        trust_p10_list = []
        trust_p50_list = []
        trust_p90_list = []
        trust_gate_ratio_list = []
        comm_valid_neighbors_list = []
        comm_kept_neighbors_list = []
        comm_malicious_ratio_list = []
        trust_malicious_gap_list = []
        consistency_score_mean_list = []
        consistency_score_mal_list = []
        consistency_score_clean_list = []
        consistency_malicious_gap_list = []
        residual_gate_mean_list = []
        residual_gate_max_list = []
        residual_comm_ratio_list = []
        comm_strength = None
        comm_enabled = None
        trust_strength = None

        rnn_actor = rnn_states_actor[:, 0]
        rnn_critic = rnn_states_critic[:, 0]

        for t in range(T):
            obs_seq = self._reshape_obs_seq(obs[t])
            state = self._extract_current_state(cent_obs[t])
            neigh_seq = self._reshape_neighbor_seq(neighbor_obs[t])

            eval_out = self.agent.evaluate_actions(
                obs_seq,
                state,
                neigh_seq,
                neighbor_masks[t],
                None if neighbor_channel_masks is None else neighbor_channel_masks[t],
                None if neighbor_channel_noise is None else neighbor_channel_noise[t],
                neighbor_actions[t],
                None if neighbor_malicious is None else neighbor_malicious[t],
                action[t],
                rnn_actor,
                rnn_critic,
                masks[t],
                available_actions[t] if available_actions is not None else None,
            )
            log_probs_list.append(eval_out["log_probs"])
            entropy_list.append(eval_out["entropy"])
            values_list.append(eval_out["values"])
            kl_list.append(eval_out["kl_loss"])
            trust_list.append(eval_out["trust_loss"])
            kl_raw_list.append(eval_out["kl_raw"])
            trust_mean_list.append(eval_out["trust_score_mean"])
            trust_p10_list.append(eval_out["trust_score_p10"])
            trust_p50_list.append(eval_out["trust_score_p50"])
            trust_p90_list.append(eval_out["trust_score_p90"])
            trust_gate_ratio_list.append(eval_out["trust_gate_ratio"])
            comm_valid_neighbors_list.append(eval_out["comm_valid_neighbors"])
            comm_kept_neighbors_list.append(eval_out["comm_kept_neighbors"])
            comm_malicious_ratio_list.append(eval_out["comm_malicious_ratio"])
            trust_malicious_gap_list.append(eval_out["trust_malicious_gap"])
            consistency_score_mean_list.append(eval_out["consistency_score_mean"])
            consistency_score_mal_list.append(eval_out["consistency_score_mal"])
            consistency_score_clean_list.append(eval_out["consistency_score_clean"])
            consistency_malicious_gap_list.append(eval_out["consistency_malicious_gap"])
            residual_gate_mean_list.append(eval_out["residual_gate_mean"])
            residual_gate_max_list.append(eval_out["residual_gate_max"])
            residual_comm_ratio_list.append(eval_out["residual_comm_ratio"])
            if comm_strength is None:
                comm_strength = eval_out["comm_strength"]
                comm_enabled = eval_out["comm_enabled"]
                trust_strength = eval_out["trust_strength"]
            rnn_actor = eval_out["next_actor_state"]
            rnn_critic = eval_out["next_critic_state"]

        log_probs = torch.stack(log_probs_list, dim=0).reshape(T * N, -1)
        entropy = torch.stack(entropy_list, dim=0).reshape(T * N, -1)
        values = torch.stack(values_list, dim=0).reshape(T * N, -1)
        kl_loss = torch.stack(kl_list, dim=0).mean()
        trust_loss = torch.stack(trust_list, dim=0).mean()
        kl_raw = torch.stack(kl_raw_list, dim=0).mean()
        trust_score_mean = torch.stack(trust_mean_list, dim=0).mean()
        trust_score_p10 = torch.stack(trust_p10_list, dim=0).mean()
        trust_score_p50 = torch.stack(trust_p50_list, dim=0).mean()
        trust_score_p90 = torch.stack(trust_p90_list, dim=0).mean()
        trust_gate_ratio = torch.stack(trust_gate_ratio_list, dim=0).mean()
        comm_valid_neighbors = torch.stack(comm_valid_neighbors_list, dim=0).mean() if comm_valid_neighbors_list else torch.zeros(1, device=self.device)
        comm_kept_neighbors = torch.stack(comm_kept_neighbors_list, dim=0).mean() if comm_kept_neighbors_list else torch.zeros(1, device=self.device)
        comm_malicious_ratio = torch.stack(comm_malicious_ratio_list, dim=0).mean() if comm_malicious_ratio_list else torch.zeros(1, device=self.device)
        trust_malicious_gap = torch.stack(trust_malicious_gap_list, dim=0).mean() if trust_malicious_gap_list else torch.zeros(1, device=self.device)
        consistency_score_mean = torch.stack(consistency_score_mean_list, dim=0).mean() if consistency_score_mean_list else torch.zeros(1, device=self.device)
        consistency_score_mal = torch.stack(consistency_score_mal_list, dim=0).mean() if consistency_score_mal_list else torch.zeros(1, device=self.device)
        consistency_score_clean = torch.stack(consistency_score_clean_list, dim=0).mean() if consistency_score_clean_list else torch.zeros(1, device=self.device)
        consistency_malicious_gap = torch.stack(consistency_malicious_gap_list, dim=0).mean() if consistency_malicious_gap_list else torch.zeros(1, device=self.device)
        residual_gate_mean = torch.stack(residual_gate_mean_list, dim=0).mean() if residual_gate_mean_list else torch.zeros(1, device=self.device)
        residual_gate_max = torch.stack(residual_gate_max_list, dim=0).mean() if residual_gate_max_list else torch.zeros(1, device=self.device)
        residual_comm_ratio = torch.stack(residual_comm_ratio_list, dim=0).mean() if residual_comm_ratio_list else torch.zeros(1, device=self.device)
        debug = {
            "kl_raw": float(kl_raw.item()),
            "trust_score_mean": float(trust_score_mean.item()),
            "trust_score_p10": float(trust_score_p10.item()),
            "trust_score_p50": float(trust_score_p50.item()),
            "trust_score_p90": float(trust_score_p90.item()),
            "trust_gate_ratio": float(trust_gate_ratio.item()),
            "comm_valid_neighbors": float(comm_valid_neighbors.item()),
            "comm_kept_neighbors": float(comm_kept_neighbors.item()),
            "comm_malicious_ratio": float(comm_malicious_ratio.item()),
            "trust_malicious_gap": float(trust_malicious_gap.item()),
            "consistency_score_mean": float(consistency_score_mean.item()),
            "consistency_score_mal": float(consistency_score_mal.item()),
            "consistency_score_clean": float(consistency_score_clean.item()),
            "consistency_malicious_gap": float(consistency_malicious_gap.item()),
            "comm_strength": float(comm_strength.item()) if comm_strength is not None else 0.0,
            "trust_strength": float(trust_strength.item()) if trust_strength is not None else 0.0,
            "comm_enabled": float(comm_enabled.item()) if comm_enabled is not None else 0.0,
            "residual_gate_mean": float(residual_gate_mean.item()),
            "residual_gate_max": float(residual_gate_max.item()),
            "residual_comm_ratio": float(residual_comm_ratio.item()),
        }

        if self._use_policy_active_masks and active_masks is not None:
            denom = active_masks.reshape(T * N, -1).sum().clamp_min(1.0)
            dist_entropy = (entropy * active_masks.reshape(T * N, -1)).sum() / denom
        else:
            dist_entropy = entropy.mean()
        return values, log_probs, dist_entropy, kl_loss, trust_loss, debug

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic: bool = False):
        if self._neighbor_obs is None or self._neighbor_masks is None:
            raise RuntimeError("VITA step context (neighbor_obs/masks) was not set before act().")

        obs = check(obs).to(device=self.device, dtype=torch.float32)
        rnn_states_actor = check(rnn_states_actor).to(device=self.device, dtype=torch.float32)
        masks = check(masks).to(device=self.device, dtype=torch.float32)
        if available_actions is not None:
            available_actions = check(available_actions).to(device=self.device, dtype=torch.float32)
        neighbor_obs = check(self._neighbor_obs).to(device=self.device, dtype=torch.float32)
        neighbor_masks = check(self._neighbor_masks).to(device=self.device, dtype=torch.float32)
        neighbor_actions = None
        if self._neighbor_actions is not None:
            neighbor_actions = check(self._neighbor_actions).to(device=self.device, dtype=torch.float32)
        neighbor_malicious = None
        if self._neighbor_malicious is not None:
            neighbor_malicious = check(self._neighbor_malicious).to(device=self.device, dtype=torch.float32)
        neighbor_channel_masks = None
        if self._neighbor_channel_masks is not None:
            neighbor_channel_masks = check(self._neighbor_channel_masks).to(device=self.device, dtype=torch.float32)
        neighbor_channel_noise = None
        if self._neighbor_channel_noise is not None:
            neighbor_channel_noise = check(self._neighbor_channel_noise).to(device=self.device, dtype=torch.float32)

        obs_seq = self._reshape_obs_seq(obs)
        neighbor_seq = self._reshape_neighbor_seq(neighbor_obs)
        dummy_state = torch.zeros(obs.size(0), self.state_dim, device=self.device, dtype=torch.float32)
        dummy_critic = torch.zeros_like(rnn_states_actor)

        out = self.agent.act(
            obs_seq,
            dummy_state,
            neighbor_seq,
            neighbor_masks,
            neighbor_channel_masks,
            neighbor_channel_noise,
            neighbor_actions,
            neighbor_malicious,
            rnn_states_actor[:, 0],
            dummy_critic[:, 0],
            masks,
            available_actions,
            deterministic=deterministic,
        )
        return out["actions"], out["next_actor_state"].unsqueeze(1)
