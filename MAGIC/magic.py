import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from action_utils import select_action, translate_action
from gnn_layers import GraphAttention

class MAGIC(nn.Module):
    """
    The communication protocol of Multi-Agent Graph AttentIon Communication (MAGIC)
    """
    def __init__(self, args):
        super(MAGIC, self).__init__()
        """
        Initialization method for the MAGIC communication protocol (2 rounds of communication)

        Arguements:
            args (Namespace): Parse arguments
        """

        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        
        dropout = 0
        negative_slope = 0.2

        # initialize sub-processors
        self.sub_processor1 = GraphAttention(args.hid_size, args.gat_hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads, self_loop_type=args.self_loop_type1, average=False, normalize=args.first_gat_normalize)
        self.sub_processor2 = GraphAttention(args.gat_hid_size*args.gat_num_heads, args.hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads_out, self_loop_type=args.self_loop_type2, average=True, normalize=args.second_gat_normalize)
        # initialize the gat encoder for the Scheduler
        if args.use_gat_encoder:
            self.gat_encoder = GraphAttention(args.hid_size, args.gat_encoder_out_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.ge_num_heads, self_loop_type=1, average=True, normalize=args.gat_encoder_normalize)

        self.obs_encoder = nn.Linear(args.obs_size, args.hid_size)

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
        self._comm_malicious_senders = None

        self.init_hidden(args.batch_size)
        self.lstm_cell= nn.LSTMCell(args.hid_size, args.hid_size)

        # initialize mlp layers for the sub-schedulers
        if not args.first_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))
                
        if args.learn_second_graph and not args.second_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))

        if args.message_encoder:
            self.message_encoder = nn.Linear(args.hid_size, args.hid_size)
        if args.message_decoder:
            self.message_decoder = nn.Linear(args.hid_size, args.hid_size)

        # initialize weights as 0
        if args.comm_init == 'zeros':
            if args.message_encoder:
                self.message_encoder.weight.data.zero_()
            if args.message_decoder:
                self.message_decoder.weight.data.zero_()
            if not args.first_graph_complete:
                self.sub_scheduler_mlp1.apply(self.init_linear)
            if args.learn_second_graph and not args.second_graph_complete:
                self.sub_scheduler_mlp2.apply(self.init_linear)
                   
        # initialize the action head (in practice, one action head is used)
        self.action_heads = nn.ModuleList([nn.Linear(2*args.hid_size, o)
                                        for o in args.naction_heads])
        # initialize the value head
        self.value_head = nn.Linear(2 * self.hid_size, 1)


    def forward(self, x, info={}):
        """
        Forward function of MAGIC (two rounds of communication)

        Arguments:
            x (list): a list for the input of the communication protocol [observations, (previous hidden states, previous cell states)]
            observations (tensor): the observations for all agents [1 (batch_size) * n * obs_size]
            previous hidden/cell states (tensor): the hidden/cell states from the previous time steps [n * hid_size]

        Returns:
            action_out (list): a list of tensors of size [1 (batch_size) * n * num_actions] that represent output policy distributions
            value_head (tensor): estimated values [n * 1]
            next hidden/cell states (tensor): next hidden/cell states [n * hid_size]
        """

        # n: number of agents

        obs, extras = x

        # encoded_obs: [1 (batch_size) * n * hid_size]
        encoded_obs = self.obs_encoder(obs)
        hidden_state, cell_state = extras

        batch_size = encoded_obs.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        # if self.args.comm_mask_zero == True, block the communiction (can also comment out the protocol to make training faster)
        if self.args.comm_mask_zero:
            agent_mask *= torch.zeros(n, 1)

        hidden_state, cell_state = self.lstm_cell(encoded_obs.squeeze(), (hidden_state, cell_state))

        # comm: [n * hid_size]
        comm = hidden_state
        if self.args.message_encoder:
            comm = self.message_encoder(comm)
            
        # mask communcation from dead agents (only effective in Traffic Junction)
        comm = comm * agent_mask
        comm = self._apply_comm_noise(comm, agent_mask, info)
        comm_ori = comm.clone()

        # sub-scheduler 1
        # if args.first_graph_complete == True, sub-scheduler 1 will be disabled
        if not self.args.first_graph_complete:
            if self.args.use_gat_encoder:
                adj_complete = self.get_complete_graph(agent_mask)
                encoded_state1 = self.gat_encoder(comm, adj_complete)
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, encoded_state1, agent_mask, self.args.directed)
            else:
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, comm, agent_mask, self.args.directed)
        else:
            adj1 = self.get_complete_graph(agent_mask)

        adj1 = self._apply_comm_drop(adj1, agent_mask)

        # sub-processor 1
        comm = F.elu(self.sub_processor1(comm, adj1))
        
        # sub-scheduler 2
        if self.args.learn_second_graph and not self.args.second_graph_complete:
            if self.args.use_gat_encoder:
                if self.args.first_graph_complete:
                    adj_complete = self.get_complete_graph(agent_mask)
                    encoded_state2 = self.gat_encoder(comm_ori, adj_complete)
                else:
                    encoded_state2 = encoded_state1
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, encoded_state2, agent_mask, self.args.directed)
            else:
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, comm_ori, agent_mask, self.args.directed)
        elif not self.args.learn_second_graph and not self.args.second_graph_complete:
            adj2 = adj1
        else:
            adj2 = self.get_complete_graph(agent_mask)

        adj2 = self._apply_comm_drop(adj2, agent_mask)

        # sub-processor 2
        comm = self.sub_processor2(comm, adj2)
        
        # mask communication to dead agents (only effective in Traffic Junction)
        comm = comm * agent_mask
        
        if self.args.message_decoder:
            comm = self.message_decoder(comm)

        value_head = self.value_head(torch.cat((hidden_state, comm), dim=-1))
        h = hidden_state.view(batch_size, n, self.hid_size)
        c = comm.view(batch_size, n, self.hid_size)

        joint = torch.cat((h, c), dim=-1)
        avail_actions = info.get('avail_actions') if isinstance(info, dict) else None
        action_out = []
        for action_head in self.action_heads:
            logits = action_head(joint)
            logits = self._apply_avail_action_mask(logits, avail_actions)
            action_out.append(F.log_softmax(logits, dim=-1))

        return action_out, value_head, (hidden_state.clone(), cell_state.clone())

    def _apply_comm_noise(self, comm, agent_mask, info):
        if (
            self.comm_noise_std <= 0.0
            and self.comm_malicious_noise_scale <= 0.0
            and self.comm_malicious_agent_prob <= 0.0
        ):
            return comm
        if self.comm_noise_std > 0.0:
            comm = comm + torch.randn_like(comm) * float(self.comm_noise_std)
        mal_mask = self._comm_malicious_mask(info, comm.size(0), comm.device, comm.dtype, agent_mask)
        if mal_mask is not None:
            base_std = self.comm_noise_std if self.comm_noise_std > 0.0 else 1.0
            std = float(max(1e-6, base_std) * max(0.0, self.comm_malicious_noise_scale))
            if std > 0.0:
                comm = comm + torch.randn_like(comm) * std * mal_mask
        if agent_mask is not None:
            comm = comm * agent_mask
        return comm

    def _comm_malicious_mask(self, info, n, device, dtype, agent_mask=None):
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
                mask = np.zeros((n,), dtype=bool)
                if prob >= 1.0:
                    mask[agent_id] = True
                else:
                    if self._comm_rng.rand() < prob:
                        mask[agent_id] = True
            elif mode in {'one', 'one_per_episode', 'episode', 'sample_one', 'random_one'}:
                mask = np.zeros((n,), dtype=bool)
                if prob >= 1.0 or self._comm_rng.rand() < prob:
                    agent_id = int(self._comm_rng.randint(0, n))
                    mask[agent_id] = True
            else:
                mask = (self._comm_rng.rand(n) < prob)

        if torch.is_tensor(mask):
            if mask.dim() > 1:
                mask = mask.view(-1)
            if mask.numel() != n:
                return None
            mask = mask.to(device=device, dtype=dtype)
            mask = (mask > 0.5).view(n, 1)
        else:
            mask = np.asarray(mask)
            if mask.ndim > 1:
                mask = mask.reshape(-1)
            if mask.size != n:
                return None
            mask = torch.from_numpy(mask.astype(np.float64)).to(device=device, dtype=dtype).view(n, 1)

        if agent_mask is not None:
            mask = mask * (agent_mask > 0.5).to(dtype)
        return mask

    def _apply_comm_drop(self, adj, agent_mask):
        if self.comm_packet_drop_prob <= 0.0:
            return adj
        n = adj.size(0)
        drop = (self._comm_rng.rand(n, n) < float(self.comm_packet_drop_prob))
        np.fill_diagonal(drop, False)
        if agent_mask is not None:
            alive = (agent_mask.view(-1).detach().cpu().numpy() > 0.5)
            drop = drop & (alive[:, None] & alive[None, :])
        keep = torch.from_numpy((~drop).astype(np.float64)).to(device=adj.device, dtype=adj.dtype)
        return adj * keep

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

    def get_agent_mask(self, batch_size, info):
        """
        Function to generate agent mask to mask out inactive agents (only effective in Traffic Junction)

        Returns:
            num_agents_alive (int): number of active agents
            agent_mask (tensor): [n, 1]
        """

        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(n, 1).clone()

        return num_agents_alive, agent_mask

    def init_linear(self, m):
        """
        Function to initialize the parameters in nn.Linear as o 
        """
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)
        
    def init_hidden(self, batch_size):
        """
        Function to initialize the hidden states and cell states
        """
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))
    
    
    def sub_scheduler(self, sub_scheduler_mlp, hidden_state, agent_mask, directed=True):
        """
        Function to perform a sub-scheduler

        Arguments: 
            sub_scheduler_mlp (nn.Sequential): the MLP layers in a sub-scheduler
            hidden_state (tensor): the encoded messages input to the sub-scheduler [n * hid_size]
            agent_mask (tensor): [n * 1]
            directed (bool): decide if generate directed graphs

        Return:
            adj (tensor): a adjacency matrix which is the communication graph [n * n]  
        """

        # hidden_state: [n * hid_size]
        n = self.args.nagents
        hid_size = hidden_state.size(-1)
        # hard_attn_input: [n * n * (2*hid_size)]
        hard_attn_input = torch.cat([hidden_state.repeat(1, n).view(n * n, -1), hidden_state.repeat(n, 1)], dim=1).view(n, -1, 2 * hid_size)
        # hard_attn_output: [n * n * 2]
        if directed:
            hard_attn_output = F.gumbel_softmax(sub_scheduler_mlp(hard_attn_input), hard=True)
        else:
            hard_attn_output = F.gumbel_softmax(0.5*sub_scheduler_mlp(hard_attn_input)+0.5*sub_scheduler_mlp(hard_attn_input.permute(1,0,2)), hard=True)
        # hard_attn_output: [n * n * 1]
        hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
        # agent_mask and agent_mask_transpose: [n * n]
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        # adj: [n * n]
        adj = hard_attn_output.squeeze() * agent_mask * agent_mask_transpose
        
        return adj
    
    def get_complete_graph(self, agent_mask):
        """
        Function to generate a complete graph, and mask it with agent_mask
        """
        n = self.args.nagents
        adj = torch.ones(n, n)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        adj = adj * agent_mask * agent_mask_transpose
        
        return adj
