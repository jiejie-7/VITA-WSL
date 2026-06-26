import argparse


def get_config():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch 
        --cuda
            by default True, will use GPU to train; or else will use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.
    
    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.
    
    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer. 
    
    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies. 
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards. 
        --use_valuenorm
            by default True, use running mean and std to normalize rewards. 
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs. 
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
    
    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)
    
    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss 
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.  
        --huber_delta <float>
            coefficient of huber loss.  
    
    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)
    
    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
    
    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    
    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='mappo', choices=["rmappo", "mappo", "happo", "hatrpo", "mat", "mat_dec", "rvita", "rtarmac"])

    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=32,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='marl', help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_false', default=True, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='StarCraft2', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", action='store_false',
                        default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false',
                        default=True, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # trpo parameters
    parser.add_argument("--kl_threshold", type=float, 
                        default=0.01, help='the threshold of kl-divergence (default: 0.01)')
    parser.add_argument("--ls_step", type=int, 
                        default=10, help='number of line search (default: 10)')
    parser.add_argument("--accept_ratio", type=float, 
                        default=0.5, help='accept ratio of loss improve (default: 0.5)')

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # VITA parameters (custom algorithm built on the official runner/buffer).
    parser.add_argument("--vita_latent_dim", type=int, default=64)
    parser.add_argument("--vita_trust_gamma", type=float, default=1.0,
                        help="sharpness for trust predictor (higher = lower trust on disagreement)")
    parser.add_argument("--vita_kl_beta", type=float, default=1e-3)
    parser.add_argument("--vita_kl_free_bits", type=float, default=0.0,
                        help="minimum KL (free-bits) used in VIB loss; 0 disables")
    parser.add_argument("--vita_vib_consistency_weight", type=float, default=0.0,
                        help="weight for sender-side dual-view VIB latent consistency regularization")
    parser.add_argument("--vita_vib_consistency_noise_std", type=float, default=0.0,
                        help="observation perturbation std used to build two training-only VIB consistency views")
    parser.add_argument("--vita_trust_lambda", type=float, default=0.1)
    parser.add_argument("--vita_trust_malicious_weight", type=float, default=1.0,
                        help="aux weight for malicious-trust loss (higher = stronger separation)")
    parser.add_argument("--vita_trust_margin_weight", type=float, default=0.5,
                        help="aux weight for separating clean/malicious trust logits")
    parser.add_argument("--vita_trust_margin", type=float, default=0.1,
                        help="margin used by VITA trust separation losses")
    parser.add_argument("--vita_trust_action_loss_weight", type=float, default=1.0,
                        help="weight on the action-prediction CE inside VITA trust loss")
    parser.add_argument("--vita_trust_reliability_mix", type=float, default=0.5,
                        help="mix confidence trust with malicious-gate reliability")
    parser.add_argument("--vita_trust_utility_mix", type=float, default=0.6,
                        help="mix auxiliary reliability with learned utility score")
    parser.add_argument("--vita_trust_counterfactual_mix", type=float, default=0.5,
                        help="mix supervised malicious labels with counterfactual utility targets")
    parser.add_argument("--vita_trust_counterfactual_margin", type=float, default=0.02,
                        help="margin for counterfactual utility ranking loss")
    parser.add_argument("--vita_trust_counterfactual_weight", type=float, default=1.0,
                        help="weight for counterfactual utility supervision")
    parser.add_argument("--vita_trust_consistency_mix", type=float, default=0.0,
                        help="mix receiver-side consistency with the reliability gate")
    parser.add_argument("--vita_trust_consistency_weight", type=float, default=0.0,
                        help="supervision weight for separating clean/malicious consistency scores")
    parser.add_argument("--vita_trust_consistency_margin", type=float, default=0.05,
                        help="margin used for clean-vs-malicious consistency separation")
    parser.add_argument("--vita_trust_consistency_noise_std", type=float, default=0.0,
                        help="deterministic perturbation scale used to probe receiver-side trust consistency")
    parser.add_argument("--vita_comm_dropout", type=float, default=0.1)
    parser.add_argument("--vita_attn_bias_coef", type=float, default=1.0)
    parser.add_argument("--vita_trust_gate_floor", type=float, default=0.0,
                        help="Minimum soft gate value applied to trust scores (0 disables).")
    parser.add_argument("--vita_trust_malicious_gate_coef", type=float, default=1.0,
                        help="strength of malicious probability in trust gate")
    parser.add_argument("--vita_allocation_sharpness", type=float, default=1.0,
                        help="sharpness exponent for VITA utility allocation")
    parser.add_argument("--vita_allocation_floor", type=float, default=0.0,
                        help="minimum allocation score before trust blending")
    parser.add_argument("--vita_trust_pair_product", action="store_true", default=False,
                        help="Use sender*receiver interaction instead of absolute feature difference in VITA trust.")
    parser.add_argument("--vita_trust_gate_product", action="store_true", default=False,
                        help="Use reliability-weighted utility product for VITA trust allocation.")
    parser.add_argument("--vita_trust_decouple_allocation", action="store_true", default=False,
                        help="Use trust only for gating and utility only for allocation in VITA.")
    parser.add_argument("--vita_trust_disable_utility_gate", action="store_true", default=False,
                        help="Do not mix utility into the VITA trust gate; use it only for allocation.")
    parser.add_argument("--vita_trust_gate_threshold", type=float, default=0.0,
                        help="Soft threshold applied to VITA reliability before trust gating.")
    parser.add_argument("--vita_comm_sight_range", type=float, default=0.0,
                        help="If >0, override per-agent sight range for communication.")
    parser.add_argument("--vita_max_neighbors", type=int, default=4)
    parser.add_argument("--vita_disable_trust", action="store_true", default=False)
    parser.add_argument("--vita_disable_kl", action="store_true", default=False)
    parser.add_argument("--vita_bypass_vib", action="store_true", default=False,
                        help="Completely bypass VIB message encoding/decoding and use hidden features as messages.")
    parser.add_argument("--vita_attention_only", action="store_true", default=False,
                        help="Bypass the variational bottleneck and use direct attention over neighbor features.")
    parser.add_argument("--vita_vib_deterministic", action="store_true", default=False,
                        help="Use deterministic sender messages (mu only) instead of sampling in the VIB encoder.")
    parser.add_argument("--vita_trust_hard_topk", action="store_true", default=False,
                        help="Use hard top-k neighbor allocation during trust-gated communication.")
    parser.add_argument("--vita_trust_topk_k", type=int, default=0,
                        help="number of neighbors retained by VITA hard top-k trust allocation")
    parser.add_argument("--vita_enable_belief_router", action="store_true", default=False,
                        help="Enable prediction-error-guided routing over self/prior/trusted social features.")
    parser.add_argument("--vita_belief_router_tau", type=float, default=3.0,
                        help="Sharpness mapping self prediction error to self confidence.")
    parser.add_argument("--vita_belief_router_strength", type=float, default=1.0,
                        help="Blend strength for the belief router over the default residual fusion.")
    parser.add_argument("--vita_belief_router_self_floor", type=float, default=0.1,
                        help="Minimum self-observation score in belief routing.")
    parser.add_argument("--vita_belief_router_prior_weight", type=float, default=0.5,
                        help="Relative weight for temporal prior fallback under self/comm uncertainty.")
    parser.add_argument("--vita_belief_router_social_weight", type=float, default=1.0,
                        help="Relative weight for trust-filtered social information under self uncertainty.")
    parser.add_argument("--vita_belief_router_comm_quantile", type=float, default=0.1,
                        help="Lower reliability quantile used for conservative social confidence.")
    parser.add_argument("--vita_belief_router_social_cap", type=float, default=0.6,
                        help="Maximum route weight assigned to trust-filtered social information.")
    parser.add_argument("--vita_belief_prior_loss_weight", type=float, default=0.0,
                        help="Weight for confidence-weighted temporal prior prediction loss.")
    parser.add_argument("--vita_belief_prior_loss_min_conf", type=float, default=0.05,
                        help="Minimum confidence weight for the temporal prior prediction loss.")

    # VITA schedule (in 'updates', i.e., PPO updates/episodes in on-policy terms).
    parser.add_argument("--vita_trust_warmup_updates", type=int, default=0)
    parser.add_argument("--vita_trust_delay_updates", type=int, default=0)
    parser.add_argument("--vita_trust_loss_warmup_updates", type=int, default=-1,
                        help="Warmup for trust supervision loss; negative means fall back to vita_trust_warmup_updates.")
    parser.add_argument("--vita_trust_loss_delay_updates", type=int, default=-1,
                        help="Delay for trust supervision loss; negative means fall back to vita_trust_delay_updates.")
    parser.add_argument("--vita_trust_gate_warmup_updates", type=int, default=-1,
                        help="Warmup for trust gating strength; negative means fall back to vita_trust_warmup_updates.")
    parser.add_argument("--vita_trust_gate_delay_updates", type=int, default=-1,
                        help="Delay for trust gating strength; negative means fall back to vita_trust_delay_updates.")
    parser.add_argument("--vita_kl_warmup_updates", type=int, default=0)
    parser.add_argument("--vita_kl_delay_updates", type=int, default=0)
    parser.add_argument("--vita_comm_delay_updates", type=int, default=0)
    parser.add_argument("--vita_comm_warmup_updates", type=int, default=0)
    parser.add_argument("--vita_comm_full_warmup_updates", type=int, default=0)

    # TarMAC baseline parameters (MAPPO backbone, VITA-aligned neighbor/noise tensors).
    parser.add_argument("--tarmac_comm_passes", type=int, default=2)
    parser.add_argument("--tarmac_attn_dim", type=int, default=16)
    parser.add_argument("--tarmac_comm_dropout", type=float, default=0.0)

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")
    
    # add for transformer
    parser.add_argument("--encode_state", action='store_true', default=False)
    parser.add_argument("--n_block", type=int, default=1)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--dec_actor", action='store_true', default=False)
    parser.add_argument("--share_actor", action='store_true', default=False)

    # add for online multi-task
    parser.add_argument("--train_maps", type=str, nargs='+', default=None)
    parser.add_argument("--eval_maps", type=str, nargs='+', default=None)
    
    return parser
