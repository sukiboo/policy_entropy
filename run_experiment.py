import argparse
import yaml

from environment import create_environment
from agent import setup_agents
from visualization import DataVisualization


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', '-l', default=False)
    parser.add_argument('--config', '-c', default='mnist')
    args = parser.parse_args()

    # obtain experiment data
    if not args.load:

        # set up experiment
        configs = yaml.safe_load(open(f'./configs/{args.config}.yml'))
        print(f'\nRunning experiment {configs["exp_name"]}...')
        env = create_environment(configs['env_name'], configs['seed'])
        agents = setup_agents(env, configs['params_exp'], configs['params_agents'], configs['seed'])

        # train agents
        for name, agent in agents.items():
            print(f'\nTraining {name} agent on {env.env_name} environment:')
            agent.train()
        viz = DataVisualization(env, agents, configs)
        viz.serialize_data()

    else:

        # load experiment data
        print(f'\nLoading experiment {args.load}...')
        configs = yaml.safe_load(open(f'./data/{args.load}.yml'))
        env = create_environment(configs['env_name'], configs['seed'])
        agents = setup_agents(env, configs['params_exp'], configs['params_agents'], configs['seed'])
        viz = DataVisualization(env, agents, configs)
        viz.load_data(args.load)

    # visualize agents' behavior
    viz.visualize_agents()

