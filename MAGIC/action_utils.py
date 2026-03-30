import numpy as np
import torch
from torch.autograd import Variable


def parse_action_args(args):
    if args.num_actions[0] > 0:
        # environment takes discrete action
        args.continuous = False
        # assert args.dim_actions == 1
        # support multi action
        args.naction_heads = [int(args.num_actions[i]) for i in range(args.dim_actions)]
    else:
        # environment takes continuous action
        actions_heads = args.nactions.split(':')
        if len(actions_heads) == 1 and int(actions_heads[0]) == 1:
            args.continuous = True
        elif len(actions_heads) == 1 and int(actions_heads[0]) > 1:
            args.continuous = False
            args.naction_heads = [int(actions_heads[0]) for _ in range(args.dim_actions)]
        elif len(actions_heads) > 1:
            args.continuous = False
            args.naction_heads = [int(i) for i in actions_heads]
        else:
            raise RuntimeError("--nactions wrong format!")


def select_action(args, action_out, deterministic=False):
    if args.continuous:
        action_mean, _, action_std = action_out
        if deterministic:
            action = action_mean
        else:
            action = torch.normal(action_mean, action_std)
        return action.detach()

    log_p_a = action_out
    p_a = [[z.exp() for z in x] for x in log_p_a]
    sampled = []
    for p in p_a:
        per_head = []
        for probs in p:
            probs_sum = probs.sum()
            invalid_probs = (
                (not torch.isfinite(probs_sum))
                or probs_sum.item() <= 0.0
                or (not torch.isfinite(probs).all())
            )
            if deterministic or invalid_probs:
                act = torch.argmax(probs, dim=-1, keepdim=True).detach()
            else:
                act = torch.multinomial(probs, 1).detach()
            per_head.append(act)
        sampled.append(torch.stack(per_head))
    return torch.stack(sampled)


def translate_action(args, env, action):
    if args.num_actions[0] > 0:
        # environment takes discrete action
        action = [x.squeeze().data.numpy() for x in action]
        actual = action
        return action, actual
    else:
        if args.continuous:
            action = action.data[0].numpy()
            cp_action = action.copy()
            # clip and scale action to correct range
            for i in range(len(action)):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                cp_action[i] = cp_action[i] * args.action_scale
                cp_action[i] = max(-1.0, min(cp_action[i], 1.0))
                cp_action[i] = 0.5 * (cp_action[i] + 1.0) * (high - low) + low
            return action, cp_action
        else:
            actual = np.zeros(len(action))
            for i in range(len(action)):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                actual[i] = action[i].data.squeeze()[0] * (high - low) / (args.naction_heads[i] - 1) + low
            action = [x.squeeze().data[0] for x in action]
            return action, actual
