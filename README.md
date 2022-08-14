# Policy Entropy of RL Agents

This repository contains the source code for the numerical experiments presented in the paper "Analyzing Policy Entropy of Reinforcement Learning Agents for Personalization Tasks".

## How to Use

### Installation
Install the requirements via `pip install -r requirements.txt`.

### Run Experiments
Run the experiments via `python -m run_experiment -c config`, where `config` is a configuration file in `./configs/` directory.
The available values are `{config_mnist, config_cifar10, config_spotify, config_recogym, config_personalization}`, which could be specified to recreate each of the presented numerical experiments.
Optionally, a custom experiment can be set up by changing or adding new configuration file.

### Load Experiments
All previously performed experiments are stored in `./data/` directory and can be recreated by loading via `python -m run_experiment -l exp_name`, where `exp_name` is the name of the experiment as it is saved in `./data/`.


## File Structure
* `run_experiment.py` --- set up and run the experiment
* `agent.py` --- set up selected RL agents
* `environment.py` --- create the specified environment
* `environments/` --- data required to set up various environments
* `configs/` --- configuration files for the experiments
* `data/` --- store data from previously run experiments
* `images/` --- plots of various results from experiments
* `visualization.py` --- save/load the experiment data, plot the results

## License
This project is licensed under the MIT License.



