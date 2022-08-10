
import gym
import numpy as np
import pandas as pd


class SpotifyEnv(gym.Env):
    '''generate a contextual bandit environment based on Spotify data'''

    def __init__(self, num_eval=10000, seed=0):
        super().__init__()
        self.env_name = 'spotify'
        self.info_train = pd.read_csv('./environments/spotify/spotify_genres.csv', index_col=0)
        self.info_test = pd.read_csv('./environments/spotify/spotify_actions.csv', index_col=0)
        self.num_eval = num_eval
        self.random_seed = seed
        self.fix_random_seed()
        self.setup()

    def fix_random_seed(self):
        '''fix random seed for reproducibility'''
        self.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.rng_s = np.random.default_rng(seed=self.random_seed)

    def setup(self):
        '''setup the environment'''
        self.genres = self.load_spotify_data_train()
        self.tracks = self.load_spotify_data_test()
        self.setup_state_space()
        self.setup_action_space()
        self.setup_reward_space()
        self.S_eval = self.observe(self.num_eval)

    def load_spotify_data_train(self):
        '''compute average feature vector for each data file in info_train'''
        genres = {}
        for genre in self.info_train['genre']:
            features = pd.read_csv(f'./environments/spotify/data/{genre}.csv', index_col=0)
            genres[genre] = dict(features.select_dtypes(include=np.number).mean())
        genres = pd.DataFrame(genres).transpose().sort_index()
        return genres

    def load_spotify_data_test(self):
        '''load test tracks from each data file in info_test'''
        tracks = []
        for playlist in self.info_test['genre']:
            tracks.append(pd.read_csv(f'./environments/spotify/data/{playlist}.csv', index_col=0))
        tracks = pd.concat(tracks, ignore_index=True)
        return tracks

    def setup_state_space(self):
        '''generate state space'''
        self.state_dim = len(self.genres)
        self.state_low, self.state_high = 0., 1.
        self.sparsity_low, self.sparsity_high = .05, .25
        self.observation_space = gym.spaces.Box(low=self.state_low, high=self.state_high,
                                                shape=(self.state_dim,), dtype=np.float32)

    def setup_action_space(self):
        '''generate action space'''
        self.action_space = gym.spaces.Discrete(len(self.tracks))
        self.labels = list(range(self.action_space.n))

    def setup_reward_space(self):
        '''generate reward matrix'''
        genre_vals = np.array(self.genres.values.tolist())
        track_vals = np.array(self.tracks.drop('id', axis=1).values.tolist())
        self.R = np.matmul(genre_vals - genre_vals.mean(axis=0), (track_vals - track_vals.mean(axis=0)).T)
        self.neg, self.pos = -.1, .1

    def compute_reward(self, s, a=None):
        '''compute the normalized reward value for a given state and an action index'''
        r = np.matmul(s, self.R) if a is None else np.matmul(s, self.R[:,a])
        r = (1*(r > self.pos) - 1*(r < self.neg)).astype(float)
        return r

    def evaluate_predictions(self, A_ind):
        '''evaluate given actions on the evaluation set'''
        r = self.compute_reward(self.S_eval)[range(self.num_eval),A_ind].mean()
        return r

    def observe(self, num=1):
        '''sample observed states'''
        sparsity = self.sparsity_low + (self.sparsity_high - self.sparsity_low) * np.random.rand(num)
        num_prefs = (sparsity * self.state_dim).astype(int)
        inds = [np.random.choice(np.arange(self.state_dim), replace=False, size=n) for n in num_prefs]
        prefs = np.zeros((num,self.state_dim))
        for i in range(num):
            prefs[i][inds[i]] = 1
        return prefs

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

