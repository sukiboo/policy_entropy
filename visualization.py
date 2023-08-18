import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dill as pickle
import yaml
import os

sbn_bold = ['#2288dd', '#dd8822', '#22dd88', '#8822dd', '#dd2288', '#88dd22']
sbn_mute = ['#66aadd', '#ddaa66', '#66ddaa', '#aa66dd', '#dd66aa', '#aadd66']
sbn_base = np.array([sbn_bold, sbn_mute]).flatten(order='C')
sbn_pair = np.array([sbn_bold, sbn_mute]).flatten(order='F')
sns.set_theme(style='darkgrid', palette=sbn_base, font='monospace', font_scale=1.5)


class DataVisualization:
    '''visualize agents' behavior throughout the training process'''

    def __init__(self, env, agents, configs):
        self.env_name = env.env_name
        self.agents = agents
        self.configs = configs
        if self.configs['exp_name'] is not None:
            self.exp_name = str(self.configs['exp_name'])
        else:
            self.exp_name = f'{self.env_name}_{self.configs["params_exp"]["arch"]}'
        os.makedirs('./images/', exist_ok=True)

    def serialize_data(self):
        '''save experiment data'''
        self.agents = {name:
                        {'eval': agent.eval, 'hist': agent.hist, 'eval_steps': agent.eval_steps,
                         'actions': agent.actions, 'rewards': agent.rewards}
                       for name, agent in self.agents.items()}
        os.makedirs('./data/', exist_ok=True)
        with open(f'./data/{self.exp_name}.pkl', 'wb') as save_file:
            pickle.dump(self.__dict__, save_file)
        yaml.dump(self.configs, open(f'./data/{self.exp_name}.yml', 'w'))

    def load_data(self, exp_name):
        '''load experiment data'''
        try:
            with open(f'./data/{exp_name}.pkl', 'rb') as save_file:
                self.__dict__.update(pickle.load(save_file))
        except:
            raise NameError(f'\ncannot load file {exp_name}...')

    def visualize_agents(self):
        '''plot various metrics'''
        self.plot_distributions(show=False)
        self.plot_distributions(sort=False, show=False)
        self.plot_entropy(show=False)
        self.plot_rewards(show=True)

    def plot_distributions(self, step=1, sort=True, show=True):
        '''plot agents' test histograms throughout the training process'''
        sns.set_palette('Paired')
        for name, agent in self.agents.items():
            fig, ax = plt.subplots(figsize=(8,4))
            df = pd.DataFrame(agent['hist'], index=agent['eval_steps'])[::step]
            if sort:
                df.values[:,::-1].sort(axis=1)
            df.plot.bar(stacked=True, width=1, ax=ax, linewidth=.1, legend=None)
            plt.xticks(np.linspace(0, len(df) - 1, 7), rotation=0)
            ax.set_ylim(0,1)
            ##ax.set_xlabel('number of agent-environment interactions')
            ##ax.set_ylabel(f'action distribution')
            plt.tight_layout()
            if sort:
                plt.savefig(f'./images/{self.exp_name}_dist_{name}.png', dpi=300, format='png')
            else:
                plt.savefig(f'./images/{self.exp_name}_dist_{name}_raw.png', dpi=300, format='png')
            if show:
                plt.show()
            else:
                plt.close()

    def plot_entropy(self, step=1, show=True):
        '''plot agents' entropy throughout the training process'''
        sns.set_palette(sbn_base)
        fig, ax = plt.subplots(figsize=(8,4))
        for name, agent in self.agents.items():
            agent_ent = [np.sum(-np.array(h) * np.log(h)) for h in agent['hist']]
            plt.plot(agent['eval_steps'][::step], agent_ent[::step], linewidth=4, label=name)
        ##ax.set_xlabel('number of agent-environment interactions')
        ##ax.set_ylabel('entropy')
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'./images/{self.exp_name}_entropy.pdf', format='pdf')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_rewards(self, step=1, show=True):
        '''plot evaluation rewards throughout the training process'''
        sns.set_palette(sbn_base)
        fig, ax = plt.subplots(figsize=(8,4))
        for name, agent in self.agents.items():
            plt.plot(agent['eval_steps'][::step], agent['eval'][::step], linewidth=4, label=name)
        ##ax.set_xlabel('number of agent-environment interactions')
        ##ax.set_ylabel('stochastic evaluation rewards')
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'./images/{self.exp_name}_reward.pdf', format='pdf')
        if show:
            plt.show()
        else:
            plt.close()

