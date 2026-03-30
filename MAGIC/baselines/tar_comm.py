# Slightly adapted from a version from Dr. Abhishek Das

import math

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from models import MLP
from action_utils import select_action, translate_action

class TarCommNetMLP(nn.Module):
    """
    MLP based CommNet. Uses communication vector to communicate info
    between agents
    """
    def __init__(self, args, num_inputs):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            MLP {object} -- Self
            args {Namespace} -- Parse args namespace
            num_inputs {number} -- Environment observation dimension for agents
        """

        super(TarCommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent

        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o)
                                        for o in args.naction_heads])
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2
        self.comm_noise_std = float(getattr(args, 'comm_noise_std', 0.0))
        self.comm_packet_drop_prob = float(getattr(args, 'comm_packet_drop_prob', 0.0))
        self.comm_malicious_agent_prob = float(getattr(args, 'comm_malicious_agent_prob', 0.0))
        self.comm_malicious_noise_scale = float(getattr(args, 'comm_malicious_noise_scale', 0.0))
        self.comm_malicious_mode = str(getattr(args, 'comm_malicious_mode', 'bernoulli')).strip().lower()
        self.comm_malicious_fixed_agent_id = int(getattr(args, 'comm_malicious_fixed_agent_id', 0))
        seed = getattr(args, 'seed', None)
        if seed is not None:
            try:
                seed = int(seed)
            except Exception:
                seed = None
        self._comm_rng = np.random.RandomState(seed) if seed is not None and seed >= 0 else np.random.RandomState()

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)


        # Since linear layers in PyTorch now accept * as any number of dimensions
        # between last and first dim, num_agents dimension will be covered.
        # The network below is function r in the paper for encoding
        # initial environment stage
        self.encoder = nn.Linear(num_inputs, args.hid_size)

        # if self.args.env_name == 'starcraft':
        #     self.state_encoder = nn.Linear(num_inputs, num_inputs)
        #     self.encoder = nn.Linear(num_inputs * 2, args.hid_size)
        if args.recurrent:
            self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

        if args.recurrent:
            self.init_hidden(args.batch_size)
            self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)

        else:
            if args.share_weights:
                self.f_module = nn.Linear(args.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                                for _ in range(self.comm_passes)])
        # else:
            # raise RuntimeError("Unsupported RNN type.")

        # Our main function for converting current hidden state to next state
        # self.f = nn.Linear(args.hid_size, args.hid_size)
        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                            for _ in range(self.comm_passes)])
        # self.C = nn.Linear(args.hid_size, args.hid_size)

        # initialise weights as 0
        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

        # print(self.C)
        # self.C.weight.data.zero_()
        # Init weights for linear layers
        # self.apply(self.init_weights)

        self.value_head = nn.Linear(self.hid_size, 1)

        ######################################################
        # [TarMAC changeset] Attentional communication modules
        ######################################################

        self.state2query = nn.Linear(args.hid_size, 16)
        self.state2key = nn.Linear(args.hid_size, 16)
        self.state2value = nn.Linear(args.hid_size, args.hid_size)


    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n
        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1).clone()

        return num_agents_alive, agent_mask

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            x, extras = x
            x = self.encoder(x)

            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
            # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)
        else:
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state


    def forward(self, x, info={}):
        # TODO: Update dimensions
        """Forward function for CommNet class, expects state, previous hidden
        and communication tensor.
        B: Batch Size: Normally 1 in case of episode
        N: number of agents

        Arguments:
            x {tensor} -- State of the agents (N x num_inputs)
            prev_hidden_state {tensor} -- Previous hidden state for the networks in
            case of multiple passes (1 x N x hid_size)
            comm_in {tensor} -- Communication tensor for the network. (1 x N x N x hid_size)

        Returns:
            tuple -- Contains
                next_hidden {tensor}: Next hidden state for network
                comm_out {tensor}: Next communication tensor
                action_data: Data needed for taking next action (Discrete values in
                case of discrete, mean and std in case of continuous)
                v: value head
        """

        # if self.args.env_name == 'starcraft':
        #     maxi = x.max(dim=-2)[0]
        #     x = self.state_encoder(x)
        #     x = x.sum(dim=-2)
        #     x = torch.cat([x, maxi], dim=-1)
        #     x = self.tanh(x)

        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        agent_mask_alive = agent_mask.clone()

        # Hard Attention - action whether an agent communicates or not
        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action'])
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
            agent_mask *= comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)

        for i in range(self.comm_passes):
            # Choose current or prev depending on recurrent
            comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state

            if self.args.comm_mask_zero:
                comm_mask = torch.zeros_like(comm)
                comm = comm * comm_mask

            sender_mask = agent_mask[:, :1, :, :].squeeze(1)
            comm = self._apply_comm_noise(comm, sender_mask, info)
            #########################################################
            # [TarMAC changeset] Don't expand same comm vector to all
            #########################################################
            # Get the next communication vector based on next hidden state
            # comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)
            #########################################################

            ########################################################
            # [TarMAC changeset] Removing self-communication masking
            ########################################################
            # # Create mask for masking self communication
            # mask = self.comm_mask.view(1, n, n)
            # mask = mask.expand(comm.shape[0], n, n)
            # mask = mask.unsqueeze(-1)
            # mask = mask.expand_as(comm)
            # comm = comm * mask
            ########################################################

            ############################################################
            # [TarMAC changeset] Replacing averaging with soft-attention
            ############################################################
            # if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' \
            #     and num_agents_alive > 1:
            #     comm = comm / (num_agents_alive - 1)
            ############################################################

            # if info['comm_action'].sum() != 0:
            #     import pdb; pdb.set_trace()

            #########################################################
            # [TarMAC changeset] Attentional communication b/w agents
            #########################################################
            # compute q, k, c
            query = self.state2query(comm)
            key = self.state2key(comm)
            value = self.state2value(comm)

            # scores
            scores = torch.matmul(query, key.transpose(
                -2, -1)) / math.sqrt(self.hid_size)
            # scores = scores.masked_fill(comm_action_mask.squeeze(-1) == 0, -1e9)
            # Use agent_mask instead of comm_action_mask to make this work in tj env
            scores = scores.masked_fill(agent_mask.squeeze(-1) == 0, -1e9)
            drop_mask = self._sample_comm_drop(scores, agent_mask, agent_mask_transpose)
            if drop_mask is not None:
                scores = scores.masked_fill(drop_mask, -1e9)

            # softmax + weighted sum
            attn = F.softmax(scores, dim=-1)
            # if the scores are all -1e9 for all agents, the attns should be all 0 (fixed from the original version)
            attn = attn * agent_mask.squeeze(-1) # cannot use inplace operation *=
            if drop_mask is not None:
                attn = attn.masked_fill(drop_mask, 0.0)
            comm = torch.matmul(attn, value)
            
            ####################################################
            # [TarMAC changeset] Incorporated this masking above
            ####################################################
            # # Mask comm_in
            # # Mask communcation from dead agents
            # comm = comm * agent_mask
            # # Mask communication to dead agents
            # comm = comm * agent_mask_transpose
            ###########################################################

            ###########################################################
            # [TarMAC changeset] Replaced this averaging with attention
            ###########################################################
            # # Combine all of C_j for an ith agent which essentially are h_j
            # comm_sum = comm.sum(dim=1)
            ###########################################################
            # for tj: dead agents do not receive messages
            # for tj: alive agents with no comm actions can receive messages (align with tarmac+ic3net in pp)
            comm *= agent_mask_alive.squeeze(-1)[:, 0].unsqueeze(-1).expand(batch_size, n, self.hid_size)
            c = self.C_modules[i](comm)

            if self.args.recurrent:
                # skip connection - combine comm. matrix and encoded input for all agents
                inp = x + c

                inp = inp.view(batch_size * n, self.hid_size)

                output = self.f_module(inp, (hidden_state, cell_state))

                hidden_state = output[0]
                cell_state = output[1]

            else: # MLP|RNN
                # Get next hidden state from f node
                # and Add skip connection from start and sum them
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)

        # v = torch.stack([self.value_head(hidden_state[:, i, :]) for i in range(n)])
        # v = v.view(hidden_state.size(0), n, -1)
        value_head = self.value_head(hidden_state)
        h = hidden_state.view(batch_size, n, self.hid_size)

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actions
            avail_actions = info.get('avail_actions') if isinstance(info, dict) else None
            action = []
            for head in self.heads:
                logits = head(h)
                logits = self._apply_avail_action_mask(logits, avail_actions)
                action.append(F.log_softmax(logits, dim=-1))

        if self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone())
        else:
            return action, value_head


    def _apply_comm_noise(self, comm, sender_mask, info):
        if (
            self.comm_noise_std <= 0.0
            and self.comm_malicious_noise_scale <= 0.0
            and self.comm_malicious_agent_prob <= 0.0
        ):
            return comm
        if self.comm_noise_std > 0.0:
            comm = comm + torch.randn_like(comm) * float(self.comm_noise_std)
        mal_mask = self._comm_malicious_mask(info, comm.size(0), comm.size(1), comm.device, comm.dtype, sender_mask)
        if mal_mask is not None:
            base_std = self.comm_noise_std if self.comm_noise_std > 0.0 else 1.0
            std = float(max(1e-6, base_std) * max(0.0, self.comm_malicious_noise_scale))
            if std > 0.0:
                comm = comm + torch.randn_like(comm) * std * mal_mask
        if sender_mask is not None:
            comm = comm * sender_mask
        return comm

    def _comm_malicious_mask(self, info, batch_size, n, device, dtype, sender_mask=None):
        mask = None
        if isinstance(info, dict) and 'comm_malicious_mask' in info:
            mask = info.get('comm_malicious_mask')
        elif isinstance(info, dict) and 'malicious_mask' in info:
            mask = info.get('malicious_mask')

        if mask is None:
            prob = float(self.comm_malicious_agent_prob)
            if prob <= 0.0:
                return None
            prob = float(min(1.0, max(0.0, prob)))
            mode = str(self.comm_malicious_mode or 'bernoulli').strip().lower()
            if mode in {'fixed', 'fixed_agent', 'fixed_sender'}:
                agent_id = int(self.comm_malicious_fixed_agent_id)
                if agent_id < 0 or agent_id >= n:
                    agent_id = 0
                mask = np.zeros((batch_size, n), dtype=bool)
                if prob >= 1.0:
                    mask[:, agent_id] = True
                else:
                    active = self._comm_rng.rand(batch_size) < prob
                    if active.any():
                        mask[active, agent_id] = True
            elif mode in {'one', 'one_per_episode', 'episode', 'sample_one', 'random_one'}:
                mask = np.zeros((batch_size, n), dtype=bool)
                if prob >= 1.0:
                    active = np.ones((batch_size,), dtype=bool)
                else:
                    active = self._comm_rng.rand(batch_size) < prob
                if active.any():
                    env_sel = np.where(active)[0]
                    agent_sel = self._comm_rng.randint(0, n, size=env_sel.size)
                    mask[env_sel, agent_sel] = True
            else:
                mask = (self._comm_rng.rand(batch_size, n) < prob)

        if torch.is_tensor(mask):
            if mask.dim() == 1:
                mask = mask.view(1, n).expand(batch_size, -1)
            elif mask.dim() > 2:
                mask = mask.view(batch_size, n)
            if mask.size(0) != batch_size or mask.size(1) != n:
                return None
            mask = (mask.to(device=device, dtype=dtype) > 0.5).to(dtype).view(batch_size, n, 1)
        else:
            mask = np.asarray(mask)
            if mask.ndim == 1:
                mask = np.broadcast_to(mask.reshape(1, n), (batch_size, n))
            elif mask.ndim > 2:
                mask = mask.reshape(batch_size, n)
            if mask.shape != (batch_size, n):
                return None
            mask = torch.from_numpy(mask.astype(np.float64)).to(device=device, dtype=dtype).view(batch_size, n, 1)

        if sender_mask is not None:
            mask = mask * (sender_mask > 0.5).to(dtype)
        return mask

    def _sample_comm_drop(self, scores, agent_mask, agent_mask_transpose):
        if self.comm_packet_drop_prob <= 0.0:
            return None
        valid = torch.ones_like(scores, dtype=torch.bool)
        if agent_mask is not None:
            valid = valid & (agent_mask.squeeze(-1) > 0.5)
        if agent_mask_transpose is not None:
            valid = valid & (agent_mask_transpose.squeeze(-1) > 0.5)
        drop = self._comm_rng.rand(*scores.shape) < float(self.comm_packet_drop_prob)
        for b in range(scores.shape[0]):
            np.fill_diagonal(drop[b], False)
        drop = drop & valid.detach().cpu().numpy()
        if not drop.any():
            return None
        return torch.from_numpy(drop).to(device=scores.device, dtype=torch.bool)

    def _apply_avail_action_mask(self, logits, avail_actions):
        if avail_actions is None:
            return logits

        mask = torch.as_tensor(avail_actions, dtype=logits.dtype, device=logits.device)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3:
            return logits

        if mask.size(0) == 1 and logits.size(0) > 1:
            mask = mask.expand(logits.size(0), -1, -1)

        if mask.size(0) != logits.size(0) or mask.size(1) != logits.size(1) or mask.size(2) != logits.size(2):
            return logits

        valid = mask > 0.5
        all_invalid = ~valid.any(dim=-1, keepdim=True)
        if all_invalid.any():
            fallback = torch.zeros_like(valid)
            fallback_idx = torch.argmax(logits, dim=-1, keepdim=True)
            fallback.scatter_(-1, fallback_idx, True)
            valid = torch.where(all_invalid, fallback, valid)

        return logits.masked_fill(~valid, -1e10)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

