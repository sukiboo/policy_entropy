
import gym
import numpy as np
from environments.personalization.bandit_environment import HyperpersonalizationEnv


class PersonalizationEnv(gym.Env):
    '''create a contextual bandit based on Hyperpersonalization environment'''

    def __init__(self, num_eval=10000, seed=0):
        super().__init__()
        self.env_name = 'personalization'
        self.num_eval = num_eval
        self.random_seed = seed
        self.fix_random_seed()
        self.setup()

    def fix_random_seed(self):
        '''fix random seed for reproducibility'''
        self.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def setup(self):
        '''setup the environment'''
        params_env = {'num_a': 100, 'dim_s': 100, 'dim_a': 100, 'dim_feature': 5,
                      's_low': -1, 's_high': 1, 'a_low': -1, 'a_high': 1,
                      'r_arch': [], 'seed': self.random_seed}
        self.bandit_env = HyperpersonalizationEnv(params_env)
        self.setup_action_space()
        self.setup_state_space()
        self.S_eval = self.observe(self.num_eval)

    def setup_action_space(self):
        '''generate action space'''
        self.action_space = gym.spaces.Discrete(self.bandit_env.num_a)

    def setup_state_space(self):
        '''generate state space'''
        self.state_dim = self.bandit_env.dim_s
        self.state_low, self.state_high = self.bandit_env.s_low, self.bandit_env.s_high
        self.observation_space = gym.spaces.Box(low=self.state_low, high=self.state_high,
                                                shape=(self.state_dim,), dtype=np.float32)

    def compute_reward(self, S, a_ind=None):
        '''compute the reward value for a given state and an action index'''
        a = self.bandit_env.A[a_ind] if a_ind is not None else self.bandit_env.A
        reward = self.bandit_env.r(S,a)
        return reward

    def evaluate_predictions(self, A_ind):
        '''evaluate given actions on the evaluation set'''
        reward = self.compute_reward(self.S_eval)[range(self.num_eval),A_ind].mean()
        return reward

    def observe(self, num=1):
        '''sample observed states'''
        state = self.bandit_env.get_state(num)
        return state

    def reset(self):
        '''observe a new state'''
        self.state = self.observe()
        return self.state

    def step(self, a):
        '''given an observed state take an action and receive reward'''
        reward = self.compute_reward(self.state, a).item()
        done = True
        info = {}
        return self.state, reward, done, info

