
import gym
import numpy as np
import tensorflow as tf


class ClassificationEnv(gym.Env):
    '''set up classification task as a contextual bandit environment'''

    def __init__(self, env_name, seed=0):
        super().__init__()
        self.env_name = env_name
        self.random_seed = seed
        self.fix_random_seed()
        self.setup()

    def fix_random_seed(self):
        '''fix random seed for reproducibility'''
        self.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

    def setup(self):
        '''setup the environment'''
        # mnist
        if self.env_name == 'mnist':
            self.labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            (x_tr, y_tr), (x_ts, y_ts) = tf.keras.datasets.mnist.load_data()
            self.x_tr = np.array([x/255 for x in x_tr]).reshape(-1,28,28,1)
            self.x_ts = np.array([x/255 for x in x_ts]).reshape(-1,28,28,1)

        # cifar10
        elif self.env_name == 'cifar10':
            self.labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            (x_tr, y_tr), (x_ts, y_ts) = tf.keras.datasets.cifar10.load_data()
            self.x_tr = np.array([x/255 for x in x_tr]).reshape(-1,32,32,3)
            self.x_ts = np.array([x/255 for x in x_ts]).reshape(-1,32,32,3)

        else:
            raise SystemExit(f'\ndataset {self.env_name} is not implemented...')

        self.y_tr = np.array(y_tr).flatten()
        self.y_ts = np.array(y_ts).flatten()
        self.setup_state_space()
        self.setup_action_space()
        self.setup_reward_space()

    def setup_state_space(self):
        '''generate state space'''
        self.num_states = self.y_tr.size
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=self.x_tr[0].shape, dtype=np.float32)

    def setup_action_space(self):
        '''generate action space'''
        self.num_classes = np.unique(self.y_tr).size
        self.action_space = gym.spaces.Discrete(self.num_classes)

    def setup_reward_space(self):
        '''generate reward matrix'''
        # maximum reward 1, average reward 0
        self.R = (self.num_classes * np.eye(self.num_classes) - 1) / (self.num_classes - 1)

    def compute_reward(self, s_ind, a_ind):
        '''compute the normalized reward value for a given state and an action index'''
        r = self.R[self.y_tr[s_ind], a_ind]
        return r

    def evaluate_predictions(self, A_ind):
        '''compute the normalized reward value on the test set'''
        r = self.R[self.y_ts, A_ind].mean()
        return r

    def observe(self):
        '''sample observed states'''
        self.state_index = np.random.randint(self.num_states)
        self.state = self.x_tr[self.state_index]
        return self.state.astype(np.float32)

    def reset(self):
        '''observe a new state'''
        state = self.observe()
        return state

    def step(self, action_index):
        '''given an observed state take an action and receive reward'''
        reward = self.compute_reward(self.state_index, action_index)
        done = True
        info = {}
        return self.state_index, reward, done, info

