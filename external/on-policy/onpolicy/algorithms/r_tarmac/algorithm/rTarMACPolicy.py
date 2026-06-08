import torch

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Critic
from onpolicy.algorithms.r_tarmac.algorithm.tarmac_actor import TarMACActor
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import update_linear_schedule


class R_TarMACPolicy:
    """MAPPO-compatible policy wrapper for TarMAC-style communication."""

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        if act_space.__class__.__name__ != "Discrete":
            raise NotImplementedError("R_TarMACPolicy currently supports Discrete action spaces only.")

        self.actor = TarMACActor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay
        )

        self._neighbor_obs = None
        self._neighbor_masks = None
        self._neighbor_channel_masks = None
        self._neighbor_channel_noise = None

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def set_step_context(
        self,
        neighbor_obs,
        neighbor_masks,
        neighbor_actions=None,
        neighbor_malicious=None,
        neighbor_channel_masks=None,
        neighbor_channel_noise=None,
    ):
        self._neighbor_obs = neighbor_obs
        self._neighbor_masks = neighbor_masks
        self._neighbor_channel_masks = neighbor_channel_masks
        self._neighbor_channel_noise = neighbor_channel_noise

    def get_actions(
        self,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False,
    ):
        if self._neighbor_obs is None or self._neighbor_masks is None:
            raise RuntimeError("TarMAC step context (neighbor_obs/masks) was not set before get_actions().")

        actions, action_log_probs, rnn_states_actor, _ = self.actor(
            obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
            neighbor_obs=self._neighbor_obs,
            neighbor_masks=self._neighbor_masks,
            neighbor_channel_masks=self._neighbor_channel_masks,
            neighbor_channel_noise=self._neighbor_channel_noise,
        )
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions_tarmac(
        self,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        neighbor_obs,
        neighbor_masks,
        neighbor_channel_masks,
        neighbor_channel_noise,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        action_log_probs, dist_entropy, debug_tensors = self.actor.evaluate_actions(
            obs,
            rnn_states_actor,
            action,
            masks,
            available_actions,
            active_masks,
            neighbor_obs=neighbor_obs,
            neighbor_masks=neighbor_masks,
            neighbor_channel_masks=neighbor_channel_masks,
            neighbor_channel_noise=neighbor_channel_noise,
        )
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        debug = {
            key: float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
            for key, value in debug_tensors.items()
        }
        return values, action_log_probs, dist_entropy, debug

    def evaluate_actions(
        self,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        if self._neighbor_obs is None or self._neighbor_masks is None:
            raise RuntimeError("TarMAC step context (neighbor_obs/masks) was not set before evaluate_actions().")
        neighbor_obs = check(self._neighbor_obs).to(device=self.device, dtype=torch.float32)
        neighbor_masks = check(self._neighbor_masks).to(device=self.device, dtype=torch.float32)
        neighbor_channel_masks = None
        if self._neighbor_channel_masks is not None:
            neighbor_channel_masks = check(self._neighbor_channel_masks).to(device=self.device, dtype=torch.float32)
        neighbor_channel_noise = None
        if self._neighbor_channel_noise is not None:
            neighbor_channel_noise = check(self._neighbor_channel_noise).to(device=self.device, dtype=torch.float32)
        values, action_log_probs, dist_entropy, _ = self.evaluate_actions_tarmac(
            cent_obs,
            obs,
            rnn_states_actor,
            rnn_states_critic,
            neighbor_obs,
            neighbor_masks,
            neighbor_channel_masks,
            neighbor_channel_noise,
            action,
            masks,
            available_actions,
            active_masks,
        )
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        if self._neighbor_obs is None or self._neighbor_masks is None:
            raise RuntimeError("TarMAC step context (neighbor_obs/masks) was not set before act().")

        actions, _, rnn_states_actor, _ = self.actor(
            obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
            neighbor_obs=self._neighbor_obs,
            neighbor_masks=self._neighbor_masks,
            neighbor_channel_masks=self._neighbor_channel_masks,
            neighbor_channel_noise=self._neighbor_channel_noise,
        )
        return actions, rnn_states_actor
