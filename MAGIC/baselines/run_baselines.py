import sys
import copy
import json
import yaml
import time
import signal
import argparse
import os
import random
import shutil
import socket
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
import visdom
from comm import CommNetMLP
from ga_comm import GACommNetMLP
from tar_comm import TarCommNetMLP
from trainer import Trainer

sys.path.append("..")
import data
from utils import *
from action_utils import parse_action_args
from multi_processing import MultiProcessTrainer

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch RL trainer')

# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=16,
                    help='How many processes to run')

# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--qk_hid_size', default=16, type=int,
                    help='key and query size for soft attention')
parser.add_argument('--value_hid_size', default=32, type=int,
                    help='value size for soft attention')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')

# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed')
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')

# environment
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')

# smac (StarCraft II)
parser.add_argument('--map_name', type=str, default='3m',
                    help='smac map name')
parser.add_argument('--use_obs_instead_of_state', action='store_true', default=False)
parser.add_argument('--add_move_state', action='store_true', default=False)
parser.add_argument('--add_local_obs', action='store_true', default=False)
parser.add_argument('--add_distance_state', action='store_true', default=False)
parser.add_argument('--add_enemy_action_state', action='store_true', default=False)
parser.add_argument('--add_agent_id', action='store_true', default=False)
parser.add_argument('--add_visible_state', action='store_true', default=False)
parser.add_argument('--add_xy_state', action='store_true', default=False)
parser.add_argument('--use_state_agent', action='store_false', default=True)
parser.add_argument('--use_mustalive', action='store_false', default=True)
parser.add_argument('--add_center_xy', action='store_false', default=True)
parser.add_argument('--use_stacked_frames', action='store_true', default=False)
parser.add_argument('--stacked_frames', type=int, default=1)

# noise (shared with MAGIC/VITA)
parser.add_argument('--obs_noise_std', default=0.0, type=float,
                    help='std of Gaussian observation noise')
parser.add_argument('--packet_drop_prob', default=0.0, type=float,
                    help='probability of dropping an agent observation')
parser.add_argument('--malicious_agent_prob', default=0.0, type=float,
                    help='probability of selecting a malicious agent')
parser.add_argument('--malicious_obs_noise_scale', default=0.0, type=float,
                    help='scale of malicious observation noise')
parser.add_argument('--malicious_obs_mode', default='add', type=str,
                    help='malicious obs noise mode: add or replace')
parser.add_argument('--noise_warmup_steps', default=0, type=int,
                    help='warmup steps for observation noise')
parser.add_argument('--obs_noise_normalize', action='store_true', default=False,
                    help='normalize obs before applying noise (use running std)')
parser.add_argument('--obs_noise_norm_eps', default=1e-6, type=float,
                    help='epsilon for obs noise normalization std')
parser.add_argument('--comm_noise_std', default=0.0, type=float,
                    help='std of Gaussian communication noise')
parser.add_argument('--comm_packet_drop_prob', default=0.0, type=float,
                    help='probability of dropping a communication edge/message')
parser.add_argument('--comm_malicious_agent_prob', default=0.0, type=float,
                    help='probability of selecting a malicious communication sender')
parser.add_argument('--comm_malicious_noise_scale', default=0.0, type=float,
                    help='scale of malicious communication noise')
parser.add_argument('--comm_malicious_mode', default='bernoulli', type=str,
                    help='malicious communication mode: bernoulli, fixed, or one')
parser.add_argument('--comm_malicious_fixed_agent_id', default=0, type=int,
                    help='fixed malicious sender id when comm_malicious_mode=fixed')

# other
parser.add_argument('--config', default='', type=str,
                    help='path to a yaml config file')
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--plot_port', default='8097', type=str,
                    help='plot port')
parser.add_argument('--run_dir', default='', type=str,
                    help='base directory for run outputs')
parser.add_argument('--save', action="store_true", default=False,
                    help='save the model after training')
parser.add_argument('--save_every', default=0, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')
parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")
parser.add_argument('--eval_interval', type=int, default=0,
                    help='evaluate every N training batches (0 disables)')
parser.add_argument('--eval_episodes', type=int, default=32,
                    help='number of evaluation episodes')

# CommNet specific args
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=False,
                    help="enable ic3net model")
parser.add_argument('--tarcomm', action='store_true', default=False,
                    help="enable tarmac model (with commnet or ic3net)")
parser.add_argument('--gacomm', action='store_true', default=False,
                    help="enable gacomm model")
parser.add_argument('--nagents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=1,
                    help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="Whether communication should be there")
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval. Default 10000 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true',
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')


def _flatten_config(cfg):
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")
    flat = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if sub_key in flat:
                    raise ValueError(f"duplicate config key: {sub_key}")
                flat[sub_key] = sub_val
        else:
            if key in flat:
                raise ValueError(f"duplicate config key: {key}")
            flat[key] = val
    return flat


def _apply_config(parser):
    args, _ = parser.parse_known_args()
    if not args.config:
        return
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    flat = _flatten_config(cfg)
    known = {}
    unknown = []
    for key, val in flat.items():
        if any(action.dest == key for action in parser._actions):
            known[key] = val
        else:
            unknown.append(key)
    if unknown:
        print("[WARN] Unknown config keys ignored:", ", ".join(sorted(unknown)))
    if known:
        parser.set_defaults(**known)


def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
        return True


def _pick_sc2_port_base(total_ports: int, *, step: int, host: str = "127.0.0.1") -> int:
    start = int(os.environ.get("MAGIC_SC2_PORT_START", "12000"))
    end = int(os.environ.get("MAGIC_SC2_PORT_END", "60000"))
    preferred = os.environ.get("MAGIC_SC2_BASE_PORT")

    total_ports = max(1, int(total_ports))
    step = max(1, int(step))

    max_base = end - step * (total_ports + 1)
    if max_base <= start:
        raise ValueError(
            f"Invalid SC2 port range: start={start}, end={end}, step={step}, total={total_ports}"
        )

    if preferred is not None:
        base = int(preferred)
        ports = [base + i * step for i in range(total_ports)]
        if all(_is_port_free(p, host=host) for p in ports):
            return base
        raise RuntimeError(
            f"MAGIC_SC2_BASE_PORT={base} not available (ports {ports[:5]}...)"
        )

    for _ in range(256):
        base = random.randrange(start, max_base)
        base -= base % step
        ports = [base + i * step for i in range(total_ports)]
        if all(_is_port_free(p, host=host) for p in ports):
            return base

    return start


def _ensure_sc2_ports(all_args) -> None:
    if getattr(all_args, "_sc2_port_base", None) is not None:
        return
    step = int(os.environ.get("MAGIC_SC2_PORT_STEP", "10"))
    n_train = int(getattr(all_args, "nprocesses", 1))
    n_eval = 1 if (getattr(all_args, "eval_interval", 0) > 0 and getattr(all_args, "eval_episodes", 0) > 0) else 0
    base = _pick_sc2_port_base(n_train + n_eval, step=step)
    all_args._sc2_port_base = int(base)
    all_args._sc2_port_step = int(step)


def _make_env_args(base_args, rank: int = 0, *, eval_env: bool = False):
    local_args = copy.copy(base_args)
    if eval_env:
        local_args.start_at_full_noise = True
    if base_args.env_name in ("smac", "starcraft2"):
        if base_args.nprocesses > 1 or (base_args.eval_interval > 0 and base_args.eval_episodes > 0):
            _ensure_sc2_ports(base_args)
            offset = base_args.nprocesses if eval_env else 0
            local_args._sc2_port = base_args._sc2_port_base + (offset + int(rank)) * base_args._sc2_port_step
    return local_args


init_args_for_env(parser)
_apply_config(parser)
args = parser.parse_args()

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0

    if args.env_name == "traffic_junction":
        args.comm_action_one = True

if args.gacomm:
    args.commnet = 1
    args.mean_ratio = 0
    if args.env_name == "traffic_junction":
        args.comm_action_one = True

# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

if args.env_name == 'grf':
    render = args.render
    args.render = False

init_env_args = _make_env_args(args, rank=0, eval_env=False)
env = data.init(init_env_args.env_name, init_env_args, False)

if args.env_name in ('smac', 'starcraft2'):
    if hasattr(env, 'n_agents'):
        args.nagents = int(env.n_agents)
        args.nfriendly = args.nagents
    if hasattr(env, 'episode_limit'):
        episode_limit = int(getattr(env, 'episode_limit', 0))
        if episode_limit > 0 and (args.max_steps <= 0 or args.max_steps == 20):
            args.max_steps = episode_limit

num_inputs = env.observation_dim
args.num_actions = env.num_actions

if hasattr(env, 'close'):
    try:
        env.close()
    except Exception:
        pass

# Multi-action
if not isinstance(args.num_actions, (list, tuple)):
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'

parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0, 10000)
torch.manual_seed(args.seed)

print(args)

if args.gacomm:
    policy_net = GACommNetMLP(args, num_inputs)
elif args.commnet:
    if args.tarcomm:
        policy_net = TarCommNetMLP(args, num_inputs)
    else:
        policy_net = CommNetMLP(args, num_inputs)
elif args.random:
    policy_net = Random(args, num_inputs)
elif args.recurrent:
    policy_net = RNN(args, num_inputs)
else:
    policy_net = MLP(args, num_inputs)

if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

if args.env_name == 'grf':
    args.render = render


def _make_trainer(rank=0):
    local_args = _make_env_args(args, rank=rank, eval_env=False)
    return Trainer(local_args, policy_net, data.init(local_args.env_name, local_args))


if args.nprocesses > 1:
    trainer = MultiProcessTrainer(args, _make_trainer)
else:
    trainer = _make_trainer(0)


disp_trainer = None
if args.display:
    disp_args = _make_env_args(args, rank=0, eval_env=False)
    disp_trainer = Trainer(disp_args, policy_net, data.init(disp_args.env_name, disp_args, False))
    disp_trainer.display = True


def disp():
    if disp_trainer is None:
        return
    _ = disp_trainer.get_episode(0)


eval_trainer = None
if args.eval_interval > 0 and args.eval_episodes > 0:
    eval_args = _make_env_args(args, rank=0, eval_env=True)
    eval_trainer = Trainer(eval_args, policy_net, data.init(eval_args.env_name, eval_args, False))
    eval_trainer.display = False


log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
log['density1'] = LogField(list(), True, 'epoch', 'num_steps')
log['density2'] = LogField(list(), True, 'epoch', 'num_steps')

if args.plot:
    vis = visdom.Visdom(env=args.plot_env, port=args.plot_port)


def _algo_dir_name():
    if args.tarcomm and args.commnet:
        return 'TarMAC'
    if args.gacomm:
        return 'GAComm'
    if args.ic3net:
        return 'IC3Net'
    if args.commnet:
        return 'CommNet'
    return 'Baseline'


repo_root = Path(__file__).resolve().parents[2]
default_root = repo_root / 'runs'
base_root = Path(args.run_dir) if args.run_dir else default_root
algo_name = _algo_dir_name()
algo_root = base_root if base_root.name.lower() == algo_name.lower() else (base_root / algo_name)

if args.env_name in ('smac', 'starcraft2'):
    model_dir = algo_root / 'smac' / args.map_name
elif args.env_name == 'grf':
    model_dir = algo_root / args.env_name / args.scenario
else:
    model_dir = algo_root / args.env_name

if not model_dir.exists():
    curr_run = 'run1'
else:
    exst_run_nums = [
        int(str(folder.name).split('run')[1])
        for folder in model_dir.iterdir()
        if str(folder.name).startswith('run')
    ]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)
run_dir = model_dir / curr_run
os.makedirs(run_dir, exist_ok=True)


def _save_run_metadata():
    args_dict = dict(vars(args))
    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=2)
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            shutil.copy2(cfg_path, run_dir / "config.yaml")


_save_run_metadata()


def _to_scalar(val):
    if isinstance(val, (list, tuple)):
        try:
            return float(np.mean(val))
        except Exception:
            return 0.0
    if isinstance(val, np.ndarray):
        try:
            return float(np.mean(val))
        except Exception:
            return 0.0
    try:
        return float(val)
    except Exception:
        return 0.0


def _append_log(entry):
    os.makedirs(run_dir, exist_ok=True)
    with (run_dir / "train.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_eval(update_idx, total_env_steps):
    if eval_trainer is None:
        return None
    episodes = int(max(1, args.eval_episodes))
    policy_net.eval()
    won_sum = 0.0
    reward_sum = 0.0
    score_reward_sum = 0.0
    with torch.no_grad():
        for _ in range(episodes):
            _, stat = eval_trainer.get_episode(update_idx, deterministic=True)
            won_sum += float(stat.get("won", stat.get("success", 0.0)))
            reward_sum += _to_scalar(stat.get("reward", 0.0))
            score_reward_sum += _to_scalar(stat.get("score_reward", stat.get("success", 0.0)))

    eval_win_rate = won_sum / float(episodes)
    eval_reward = reward_sum / float(episodes)
    eval_score_reward = score_reward_sum / float(episodes)

    if args.env_name == 'grf':
        print("Eval@{} Success: {:.4f} ({}/{})".format(update_idx, eval_win_rate, int(won_sum), episodes))
        print("Eval@{} ScoreReward: {:.4f}".format(update_idx, eval_score_reward))
    else:
        print("Eval@{} WinRate: {:.4f} ({}/{})".format(update_idx, eval_win_rate, int(won_sum), episodes))
    _append_log({
        "time": time.time(),
        "phase": "eval",
        "update": update_idx,
        "total_env_steps": total_env_steps,
        "eval_win_rate": eval_win_rate,
        "eval_episode_reward": eval_reward,
        "eval_score_reward": eval_score_reward,
        "step": update_idx,
    })
    if args.save:
        with (run_dir / "eval.log").open("a", encoding="utf-8") as f:
            metric = eval_win_rate
            f.write("{}\t{:.6f}\t{}\n".format(update_idx, metric, episodes))
    policy_net.train()
    return eval_win_rate


def run(num_epochs):
    num_episodes = 0
    total_updates = 0
    total_env_steps = 0

    if args.save:
        os.makedirs(run_dir, exist_ok=True)

    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        for n in range(args.epoch_size):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True
            s = trainer.train_batch(ep)
            total_updates += 1

            batch_steps = int(s.get("num_steps", 0))
            total_env_steps += batch_steps
            batch_episodes = int(s.get("num_episodes", 0))
            reward_mean = _to_scalar(s.get("reward", 0.0))
            episode_reward = reward_mean / float(max(1, batch_episodes))
            score_reward = _to_scalar(s.get("score_reward", s.get("success", 0.0))) / float(max(1, batch_episodes))
            won_count = _to_scalar(s.get("won", s.get("success", 0.0)))
            won_flag_rate = won_count / float(max(1, batch_episodes))

            battles_won = _to_scalar(s.get("battles_won", 0.0))
            battles_game = _to_scalar(s.get("battles_game", 0.0))
            if battles_game > 0:
                incre_win_rate = battles_won / battles_game
            else:
                incre_win_rate = won_flag_rate

            num_steps = float(max(1, batch_steps))
            value_loss = _to_scalar(s.get("value_loss", 0.0)) / num_steps
            policy_loss = _to_scalar(s.get("action_loss", 0.0)) / num_steps
            dist_entropy = _to_scalar(s.get("entropy", 0.0)) / num_steps
            density1 = _to_scalar(s.get("density1", 0.0)) / num_steps
            density2 = _to_scalar(s.get("density2", 0.0)) / num_steps
            action_fix_count = _to_scalar(s.get("action_fix_count", 0.0))
            action_fix_rate = action_fix_count / float(max(1.0, batch_steps * max(1, args.nfriendly)))

            _append_log({
                "time": time.time(),
                "phase": "train",
                "update": total_updates,
                "total_env_steps": total_env_steps,
                "episode_reward": episode_reward,
                "score_reward": score_reward,
                "value_loss": value_loss,
                "policy_loss": policy_loss,
                "dist_entropy": dist_entropy,
                "entropy": dist_entropy,
                "incre_win_rate": incre_win_rate,
                "won_flag_rate": won_flag_rate,
                "action_fix_count": action_fix_count,
                "action_fix_rate": action_fix_rate,
                "density1": density1,
                "density2": density2,
                "step": total_updates,
            })

            if eval_trainer is not None and args.eval_interval > 0 and total_updates % args.eval_interval == 0:
                run_eval(total_updates, total_env_steps)

            print('batch: ', n)
            merge_stat(s, stat)
            trainer.display = False

        epoch_time = time.time() - epoch_begin_time
        epoch = len(log['epoch'].data) + 1
        num_episodes += stat['num_episodes']
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        np.set_printoptions(precision=2)

        print('Epoch {}'.format(epoch))
        print('Episode: {}'.format(num_episodes))
        print('Reward: {}'.format(stat['reward']))
        print('Time: {:.2f}s'.format(epoch_time))
        if args.env_name == 'grf' and 'score_reward' in stat.keys() and stat['num_episodes'] > 0:
            print('Score-Reward: {:.4f}'.format(_to_scalar(stat['score_reward']) / float(max(1, stat['num_episodes']))))

        if 'enemy_reward' in stat.keys():
            print('Enemy-Reward: {}'.format(stat['enemy_reward']))
        if 'add_rate' in stat.keys():
            print('Add-Rate: {:.2f}'.format(stat['add_rate']))
        if 'success' in stat.keys():
            print('Success: {:.4f}'.format(stat['success']))
        if 'steps_taken' in stat.keys():
            print('Steps-Taken: {:.2f}'.format(stat['steps_taken']))
        if 'comm_action' in stat.keys():
            print('Comm-Action: {}'.format(stat['comm_action']))
        if 'enemy_comm' in stat.keys():
            print('Enemy-Comm: {}'.format(stat['enemy_comm']))
        if 'density1' in stat.keys():
            print('density1: {:.4f}'.format(stat['density1']))
        if 'density2' in stat.keys():
            print('density2: {:.4f}'.format(stat['density2']))

        if args.plot:
            for k, v in log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(
                        np.asarray(v.data),
                        np.asarray(log[v.x_axis].data[-len(v.data):]),
                        win=k,
                        opts=dict(xlabel=v.x_axis, ylabel=k),
                    )

        if args.save_every and ep and args.save and (ep + 1) % args.save_every == 0:
            save(final=False, episode=ep + 1)

        if args.save:
            save(final=True)


def save(final, episode=0):
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    if final:
        torch.save(d, run_dir / 'model.pt')
    else:
        torch.save(d, run_dir / ('model_ep%i.pt' % (episode)))


def load(path):
    d = torch.load(path)
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])


def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Exiting gracefully.')
    if args.display:
        env.end_display()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if args.load != '':
    load(args.load)

run(args.num_epochs)
if args.display:
    env.end_display()

if args.save:
    save(final=True)

if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    os._exit(0)
