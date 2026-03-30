import gym
import gfootball.env as grf_env
import numpy as np

class GRFWrapperEnv(gym.Env):
    
    def __init__(self,):
        self.env = None
        self.num_controlled_lagents = 0
        self.num_controlled_ragents = 0
        self.num_controlled_agents = 0
        self.num_lagents = 0
        self.num_ragents = 0
        self.action_space = None
        self.observation_space = None
        
    def init_args(self, parser):
        env = parser.add_argument_group('GRF')
        env.add_argument('--scenario', type=str, default='academy_3_vs_1_with_keeper',
                         help="Scenario of the game")        
        env.add_argument('--num_controlled_lagents', type=int, default=3,
                         help="Number of controlled agents on the left side")
        env.add_argument('--num_controlled_ragents', type=int, default=0,
                         help="Number of controlled agents on the right side")  
        env.add_argument('--reward_type', type=str, default='scoring',
                         help="Reward type for training")
        env.add_argument('--grf_representation', type=str, default='multiagent',
                         help="Preferred GRF observation representation")
        env.add_argument('--grf_fallback_representation', type=str, default='simple115v2',
                         help="Fallback GRF observation representation when preferred one is unsupported")
        env.add_argument('--render', action="store_true", default=False,
                         help="Render training or testing process")
        
    def multi_agent_init(self, args):
        env_kwargs = dict(
            env_name=args.scenario,
            stacked=False,
            rewards=args.reward_type,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=args.render,
            dump_frequency=0,
            logdir='/tmp/test',
            extra_players=None,
            number_of_left_players_agent_controls=args.num_controlled_lagents,
            number_of_right_players_agent_controls=args.num_controlled_ragents,
            channel_dimensions=(3, 3),
        )
        preferred_rep = getattr(args, 'grf_representation', 'multiagent')
        fallback_rep = getattr(args, 'grf_fallback_representation', 'simple115v2')
        try:
            self.env = grf_env.create_environment(representation=preferred_rep, **env_kwargs)
            self.representation = preferred_rep
        except ValueError as exc:
            if preferred_rep == fallback_rep:
                raise
            if 'Unsupported representation' not in str(exc):
                raise
            print(f"[GRF] representation '{preferred_rep}' unsupported, fallback to '{fallback_rep}'")
            self.env = grf_env.create_environment(representation=fallback_rep, **env_kwargs)
            self.representation = fallback_rep
        self.num_controlled_lagents = args.num_controlled_lagents
        self.num_controlled_ragents = args.num_controlled_ragents
        self.num_controlled_agents = args.num_controlled_lagents + args.num_controlled_ragents
        self.num_lagents = args.num_controlled_lagents
        self.num_ragents = args.num_controlled_ragents
        if self.num_controlled_agents > 1:
            action_space = gym.spaces.Discrete(int(self.env.action_space.nvec[0]))
        else:
            action_space = self.env.action_space
        if self.num_controlled_agents > 1:
            observation_space = gym.spaces.Box(
                low=self.env.observation_space.low[0],
                high=self.env.observation_space.high[0],
                dtype=self.env.observation_space.dtype)
        else:
            observation_space = gym.spaces.Box(
                low=self.env.observation_space.low,
                high=self.env.observation_space.high,
                dtype=self.env.observation_space.dtype)
            
        # check spaces
        self.action_space = action_space
        self.observation_space = observation_space 
        return
   
    # check epoch arg
    def reset(self):
        self.stat = {'score_reward': 0.0, 'success': 0.0}
        obs = self.env.reset()
        if self.num_controlled_agents == 1:
            obs = obs.reshape(1, -1)
        return obs
    
    def step(self, actions):
        o, r, d, i = self.env.step(actions)
        if self.num_controlled_agents == 1:
            o = o.reshape(1, -1)
            r = r.reshape(1, -1)
        next_obs = o
        rewards = r
        dones = d
        infos = i
        score_reward = 0.0
        if isinstance(infos, dict):
            score_reward = float(np.asarray(infos.get('score_reward', 0.0), dtype=np.float32).mean())
        self.stat['score_reward'] = float(self.stat.get('score_reward', 0.0)) + score_reward
        if score_reward > 0.0:
            self.stat['success'] = 1.0
        
        return next_obs, rewards, dones, infos
        
    def seed(self):
        return
    
    def render(self):
        self.env.render()
        
    def exit_render(self):
        self.env.disable_render()
        