
import numpy as np
import torch
import stable_baselines3 as sb3

np.set_printoptions(precision=3, suppress=True)


def setup_agents(env, params_exp, params_agents, seed=0):
    '''setup agents to solve the given environment'''
    agents = {}
    for name, algorithm in params_agents.items():
        if algorithm['alg_type'] in ['A2C', 'DQN', 'PPO']:
            agents[name] = AgentRL(env, params_exp, params_agents[name], seed, name)
        else:
            raise SystemExit(f'algorithm type {algorithm["alg_type"]} is not implemented...')
    return agents


class AgentRL:
    '''RL agents for various tasks'''

    def __init__(self, env, params_exp, params_agent, seed, name='agent'):
        self.__dict__.update(params_exp)
        self.__dict__.update(params_agent)
        self.env = env
        self.name = name
        if 'seed' not in params_agent:
            self.seed = seed
        self.fix_random_seed()
        self.setup_policy()
        self.get_eval_data()

    def fix_random_seed(self):
        '''set random seed for reproducibility'''
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def setup_policy(self, verbose=0):
        '''configure the policy, all updates are once per batch_size'''
        # Advantage Actor Critic
        if self.alg_type == 'A2C':
            self.policy = sb3.A2C('MlpPolicy', self.env, policy_kwargs={'net_arch': self.arch},
                                  n_steps=self.batch_size,
                                  seed=self.seed, verbose=verbose, **self.hyperparams)
        # Proximal Policy Optization
        elif self.alg_type == 'PPO':
            self.policy = sb3.PPO('MlpPolicy', self.env, policy_kwargs={'net_arch': self.arch},
                                  batch_size=self.batch_size, n_steps=self.batch_size, n_epochs=1,
                                  seed=self.seed, verbose=verbose, **self.hyperparams)
        # Deep Q Network
        elif self.alg_type == 'DQN':
            self.policy = sb3.DQN('MlpPolicy', self.env, policy_kwargs={'net_arch': self.arch},
                                  batch_size=self.batch_size, train_freq=self.batch_size,
                                  learning_starts=0, buffer_size=10000,
                                  seed=self.seed, verbose=verbose, **self.hyperparams)
        else:
            raise SystemExit(f'{self.alg_type}-agent is not implemented...')

    def get_eval_data(self):
        '''get evaluation set to test agents'''
        if self.env.env_name in ['mnist', 'cifar10', 'cifar100']:
            self.S_eval = torch.from_numpy(self.env.x_ts).float().reshape(len(self.env.x_ts),-1)
        elif self.env.env_name in ['spotify', 'recogym', 'personalization']:
            self.S_eval = torch.from_numpy(self.env.S_eval).float()
        else:
            raise SystemExit(f'environment {self.env.env_name} does not have evaluation set...')

    def train(self):
        '''train agent on the given environment'''
        self.fix_random_seed()
        self.env.fix_random_seed()
        self.eval, self.hist, self.actions, self.rewards = [], [], [], []
        self.eval_steps = np.arange(0, self.num_timesteps + 1, self.eval_interval).astype(int)
        callback = Callback(self.evaluate_performance, self.eval_steps)
        self.policy.learn(int(self.num_timesteps), callback=callback)
        self.actions, self.rewards = callback.data

    def evaluate_performance(self, step=0, deterministic=False):
        '''evaluate agent's performance and action distribution on the collected data'''
        # compute network output
        if self.alg_type == 'DQN':
            out = 10*self.policy.policy.q_net(self.S_eval)
        else:
            out = self.policy.policy.action_net(self.policy.policy.mlp_extractor(self.S_eval)[0])

        # compute predictions, probabilities, and rewards
        if deterministic:
            preds = out.detach().numpy().argmax(axis=1)
            probs = np.histogram(preds, bins=np.arange(self.env.num_classes + 1), density=True)[0]
        else:
            preds = torch.distributions.Categorical(logits=out).sample().detach().numpy()
            probs = torch.nn.Softmax(dim=-1)(out).detach().numpy().mean(axis=0)
        r = self.env.evaluate_predictions(preds)

        # record data
        self.eval.append(r)
        self.hist.append(list(probs))
        print(f'iteration {step:6d}:  r = {r: .3f},  h = {np.array(self.hist[-1][:10])}')


class Callback(sb3.common.callbacks.BaseCallback):
    '''callback that evaluates agent's policy on given timesteps'''

    def __init__(self, evaluate_performance, eval_steps):
        super().__init__()
        self.evaluate_performance = evaluate_performance
        self.eval_steps = eval_steps
        self.data = [[], []]

    def _on_training_start(self):
        t = self.num_timesteps
        if t in self.eval_steps:
            self.evaluate_performance(step=t)

    def _on_step(self):
        t = self.num_timesteps
        if t in self.eval_steps:
            self.evaluate_performance(step=t)
        if t <= self.eval_steps[-1]:
            try:
                self.data[0].append(self.locals['actions'].item())
                self.data[1].append(self.locals['rewards'].item())
            except:
                # compatability fix
                self.data[0].append(0)
                self.data[1].append(0)
        return True

