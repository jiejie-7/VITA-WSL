import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.utils.distributions import FixedCategorical
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import get_shape_from_obs_space


class TarMACActor(nn.Module):
    """MAPPO actor using the MAGIC TarMAC communication/update equations.

    This ports the TarCommNetMLP communication block into the on-policy actor:
    encoder -> state2query/key/value -> soft attention -> C_modules -> recurrent
    hidden update. The MAPPO runner stores one recurrent state, so the LSTM cell
    state is reconstructed from the stored hidden state at sequence boundaries.
    """

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(TarMACActor, self).__init__()
        if action_space.__class__.__name__ != "Discrete":
            raise NotImplementedError("TarMACActor currently supports Discrete action spaces only.")

        self.hidden_size = int(args.hidden_size)
        self.action_dim = int(action_space.n)
        self.comm_passes = max(1, int(getattr(args, "tarmac_comm_passes", 2)))
        # MAGIC TarMAC hard-codes query/key size to 16.
        self.attn_dim = max(1, int(getattr(args, "tarmac_attn_dim", 16)))
        self.use_recurrent = bool(args.use_recurrent_policy or args.use_naive_recurrent_policy)
        self._use_policy_active_masks = bool(args.use_policy_active_masks)
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        obs_dim = int(obs_shape[0])

        # Match MAGIC TarCommNetMLP: plain Linear encoder and tanh only for non-recurrent mode.
        self.encoder = nn.Linear(obs_dim, self.hidden_size)
        if self.use_recurrent:
            self.f_module = nn.LSTMCell(self.hidden_size, self.hidden_size)
        else:
            self.f_modules = nn.ModuleList(
                [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.comm_passes)]
            )

        self.C_modules = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.comm_passes)]
        )
        self.tanh = nn.Tanh()

        self.state2query = nn.Linear(self.hidden_size, self.attn_dim)
        self.state2key = nn.Linear(self.hidden_size, self.attn_dim)
        self.state2value = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)

        self.to(device)

    def _encode_obs(self, obs):
        x = self.encoder(obs)
        if not self.use_recurrent:
            x = self.tanh(x)
        return x

    def _prepare_neighbor_comm(
        self,
        neighbor_obs,
        neighbor_masks,
        neighbor_channel_masks=None,
        neighbor_channel_noise=None,
    ):
        if neighbor_obs is None or neighbor_masks is None:
            raise RuntimeError("TarMAC requires neighbor_obs and neighbor_masks.")

        neighbor_obs = check(neighbor_obs).to(**self.tpdv)
        neighbor_masks = check(neighbor_masks).to(**self.tpdv)
        if neighbor_masks.dim() == 2:
            neighbor_masks = neighbor_masks.unsqueeze(-1)

        batch_size, max_neighbors = neighbor_obs.shape[:2]
        flat_neighbor_obs = neighbor_obs.reshape(batch_size * max_neighbors, -1)
        comm = self._encode_obs(flat_neighbor_obs).view(batch_size, max_neighbors, self.hidden_size)

        effective_masks = neighbor_masks
        if neighbor_channel_masks is not None:
            effective_masks = effective_masks * check(neighbor_channel_masks).to(**self.tpdv)

        if neighbor_channel_noise is not None:
            channel_noise = check(neighbor_channel_noise).to(**self.tpdv)
            if channel_noise.size(-1) < self.hidden_size:
                pad = torch.zeros(
                    *channel_noise.shape[:-1],
                    self.hidden_size - channel_noise.size(-1),
                    dtype=channel_noise.dtype,
                    device=channel_noise.device,
                )
                channel_noise = torch.cat([channel_noise, pad], dim=-1)
            elif channel_noise.size(-1) > self.hidden_size:
                channel_noise = channel_noise[..., : self.hidden_size]
            # MAGIC injects communication noise into the communicated hidden vector
            # before query/key/value projection.
            comm = comm + channel_noise

        comm = comm * effective_masks
        return comm, neighbor_masks, effective_masks

    def _magic_tarmac_update(
        self,
        encoded_obs,
        hidden_state,
        cell_state,
        neighbor_obs,
        neighbor_masks,
        neighbor_channel_masks=None,
        neighbor_channel_noise=None,
    ):
        neighbor_comm, env_masks, effective_masks = self._prepare_neighbor_comm(
            neighbor_obs,
            neighbor_masks,
            neighbor_channel_masks,
            neighbor_channel_noise,
        )

        valid = effective_masks.squeeze(-1) > 0.5
        has_neighbor = valid.any(dim=-1, keepdim=True)
        attn_entropy = encoded_obs.new_zeros(encoded_obs.size(0))

        for pass_i in range(self.comm_passes):
            # MAGIC uses current hidden state to query senders, and sender hidden
            # vectors to produce keys/values.
            query = self.state2query(hidden_state).unsqueeze(1)
            key = self.state2key(neighbor_comm)
            value = self.state2value(neighbor_comm)

            scores = torch.matmul(query, key.transpose(-2, -1)).squeeze(1) / math.sqrt(float(self.hidden_size))
            scores = scores.masked_fill(~valid, -1e9)

            attn = F.softmax(scores, dim=-1)
            attn = torch.where(has_neighbor, attn, torch.zeros_like(attn))
            attn = attn * valid.to(attn.dtype)

            comm = torch.matmul(attn.unsqueeze(1), value).squeeze(1)
            c = self.C_modules[pass_i](comm)

            if self.use_recurrent:
                hidden_state, cell_state = self.f_module(encoded_obs + c, (hidden_state, cell_state))
            else:
                hidden_state = self.tanh(encoded_obs + self.f_modules[pass_i](hidden_state) + c)

            attn_entropy = -(attn * torch.log(attn.clamp_min(1e-8))).sum(dim=-1)

        debug = {
            "comm_env_neighbors": env_masks.squeeze(-1).sum(dim=-1).mean(),
            "comm_effective_neighbors": effective_masks.squeeze(-1).sum(dim=-1).mean(),
            "comm_attention_entropy": attn_entropy[has_neighbor.squeeze(-1)].mean()
            if has_neighbor.any()
            else encoded_obs.new_tensor(0.0),
        }
        return hidden_state, cell_state, debug

    def _single_step_features(
        self,
        obs,
        rnn_states,
        masks,
        neighbor_obs,
        neighbor_masks,
        neighbor_channel_masks=None,
        neighbor_channel_noise=None,
    ):
        encoded_obs = self._encode_obs(obs)
        if self.use_recurrent:
            if rnn_states.dim() == 3:
                hidden_state = rnn_states[:, 0]
            else:
                hidden_state = rnn_states
            hidden_state = hidden_state * masks
            # The on-policy buffer stores one actor recurrent state. MAGIC's
            # LSTMCell also has a cell state; using the stored hidden as the
            # boundary cell keeps the update deterministic without changing the
            # shared buffer layout.
            cell_state = hidden_state
        else:
            hidden_state = encoded_obs
            cell_state = encoded_obs

        features, cell_state, debug = self._magic_tarmac_update(
            encoded_obs,
            hidden_state,
            cell_state,
            neighbor_obs,
            neighbor_masks,
            neighbor_channel_masks,
            neighbor_channel_noise,
        )
        next_rnn_states = features.unsqueeze(1) if self.use_recurrent else rnn_states
        return features, next_rnn_states, debug

    def _sequence_features(
        self,
        obs,
        rnn_states,
        masks,
        neighbor_obs,
        neighbor_masks,
        neighbor_channel_masks=None,
        neighbor_channel_noise=None,
    ):
        n_batch = int(rnn_states.size(0))
        if obs.size(0) % n_batch != 0:
            raise ValueError(f"TarMAC recurrent batch {obs.size(0)} is not divisible by N={n_batch}.")
        t_steps = int(obs.size(0) // n_batch)

        obs = obs.view(t_steps, n_batch, -1)
        masks = masks.view(t_steps, n_batch, -1)
        neighbor_obs = neighbor_obs.view(t_steps, n_batch, neighbor_obs.size(1), -1)
        neighbor_masks = neighbor_masks.view(t_steps, n_batch, neighbor_masks.size(1), -1)
        if neighbor_channel_masks is not None:
            neighbor_channel_masks = neighbor_channel_masks.view(
                t_steps, n_batch, neighbor_channel_masks.size(1), -1
            )
        if neighbor_channel_noise is not None:
            neighbor_channel_noise = neighbor_channel_noise.view(
                t_steps,
                n_batch,
                neighbor_channel_noise.size(1),
                neighbor_channel_noise.size(2),
            )

        hidden_state = rnn_states[:, 0] if rnn_states.dim() == 3 else rnn_states
        cell_state = hidden_state
        features = []
        debug_accum = {
            "comm_env_neighbors": [],
            "comm_effective_neighbors": [],
            "comm_attention_entropy": [],
        }

        for t in range(t_steps):
            hidden_state = hidden_state * masks[t]
            cell_state = cell_state * masks[t]
            encoded_obs = self._encode_obs(obs[t])
            hidden_state, cell_state, debug = self._magic_tarmac_update(
                encoded_obs,
                hidden_state,
                cell_state,
                neighbor_obs[t],
                neighbor_masks[t],
                None if neighbor_channel_masks is None else neighbor_channel_masks[t],
                None if neighbor_channel_noise is None else neighbor_channel_noise[t],
            )
            features.append(hidden_state)
            for key in debug_accum:
                debug_accum[key].append(debug[key])

        out_features = torch.stack(features, dim=0).reshape(t_steps * n_batch, -1)
        next_rnn_states = hidden_state.unsqueeze(1)
        debug = {key: torch.stack(values).mean() for key, values in debug_accum.items()}
        return out_features, next_rnn_states, debug

    def _features(
        self,
        obs,
        rnn_states,
        masks,
        neighbor_obs,
        neighbor_masks,
        neighbor_channel_masks=None,
        neighbor_channel_noise=None,
    ):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        neighbor_obs = check(neighbor_obs).to(**self.tpdv)
        neighbor_masks = check(neighbor_masks).to(**self.tpdv)
        if neighbor_channel_masks is not None:
            neighbor_channel_masks = check(neighbor_channel_masks).to(**self.tpdv)
        if neighbor_channel_noise is not None:
            neighbor_channel_noise = check(neighbor_channel_noise).to(**self.tpdv)

        if (not self.use_recurrent) or obs.size(0) == rnn_states.size(0):
            return self._single_step_features(
                obs,
                rnn_states,
                masks,
                neighbor_obs,
                neighbor_masks,
                neighbor_channel_masks,
                neighbor_channel_noise,
            )
        return self._sequence_features(
            obs,
            rnn_states,
            masks,
            neighbor_obs,
            neighbor_masks,
            neighbor_channel_masks,
            neighbor_channel_noise,
        )

    def _distribution(self, features, available_actions=None):
        logits = self.action_head(features)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            logits = logits.masked_fill(available_actions <= 0.0, -1e10)
        return FixedCategorical(logits=logits)

    def forward(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
        neighbor_obs=None,
        neighbor_masks=None,
        neighbor_channel_masks=None,
        neighbor_channel_noise=None,
    ):
        features, next_rnn_states, debug = self._features(
            obs,
            rnn_states,
            masks,
            neighbor_obs,
            neighbor_masks,
            neighbor_channel_masks,
            neighbor_channel_noise,
        )
        dist = self._distribution(features, available_actions)
        actions = dist.mode() if deterministic else dist.sample()
        action_log_probs = dist.log_probs(actions)
        return actions, action_log_probs, next_rnn_states, debug

    def evaluate_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        neighbor_obs=None,
        neighbor_masks=None,
        neighbor_channel_masks=None,
        neighbor_channel_noise=None,
    ):
        features, _next_rnn_states, debug = self._features(
            obs,
            rnn_states,
            masks,
            neighbor_obs,
            neighbor_masks,
            neighbor_channel_masks,
            neighbor_channel_noise,
        )
        action = check(action).to(device=features.device)
        dist = self._distribution(features, available_actions)
        action_log_probs = dist.log_probs(action)
        entropy = dist.entropy()

        if active_masks is not None and self._use_policy_active_masks:
            active_masks = check(active_masks).to(**self.tpdv).squeeze(-1)
            dist_entropy = (entropy * active_masks).sum() / active_masks.sum().clamp_min(1.0)
        else:
            dist_entropy = entropy.mean()
        return action_log_probs, dist_entropy, debug
