
import gym
import numpy as np
import environments.recogym as recogym


class RecogymEnv(gym.Env):
    '''create a contextual bandit based on RecoGym environment'''

    def __init__(self, num_eval=10000, seed=0):
        super().__init__()
        self.env_name = 'recogym'
        self.num_eval = num_eval
        self.random_seed = seed
        self.fix_random_seed()
        self.setup()

    def fix_random_seed(self):
        '''fix random seed for reproducibility'''
        self.seed(self.random_seed)
        np.random.seed(self.random_seed)
        recogym.env_1_args['num_products'] = 50
        recogym.env_1_args['random_seed'] = self.random_seed

    def setup(self):
        '''setup the environment'''
        self.recogym_env = gym.make('reco-gym-v1')
        self.recogym_env.init_gym(recogym.env_1_args)
        self.setup_action_space()
        self.setup_state_space()
        self.S_eval = self.observe(self.num_eval)

    def setup_action_space(self):
        '''generate action space'''
        self.action_space = self.recogym_env.action_space

    def setup_state_space(self):
        '''generate state space'''
        self.state_dim = self.action_space.n
        self.state_low, self.state_high = -10., 10.
        self.observation_space = gym.spaces.Box(low=self.state_low, high=self.state_high,
                                                shape=(self.state_dim,), dtype=np.float32)

    def compute_reward(self, S, a):
        '''compute the reward value for a given state and an action index'''
        reward = np.zeros(len(S))
        for i in range(len(S)):
            self.reset()
            done = False
            while not done:
                _, r, done, info = self.recogym_env.step(np.argsort(S[i])[a])
                reward[i] += r
        return reward

    def evaluate_predictions(self, A_ind):
        '''evaluate given actions on the evaluation set'''
        reward = 0.
        for i in range(self.num_eval):
            reward += self.compute_reward(np.expand_dims(self.S_eval[i], axis=0), A_ind[i]).item()
        return reward / self.num_eval

    def observe(self, num=1):
        '''sample observed states'''
        state = np.clip(np.random.randn(num,self.state_dim), self.state_low, self.state_high)
        return state

    def reset(self):
        '''observe a new state'''
        self.recogym_env.reset_random_seed()
        self.recogym_env.reset()
        self.recogym_env.step(None)
        self.state = self.observe()
        return self.state

    def step(self, a):
        '''given an observed state take an action and receive reward'''
        reward = self.compute_reward(self.state, a).item()
        done = True
        info = {}
        return self.state, reward, done, info

