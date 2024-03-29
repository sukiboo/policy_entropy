# Policy Entropy of RL Agents

This repository contains the source code for the numerical experiments presented in the paper "[Analyzing Policy Entropy of Reinforcement Learning Agents for Personalization Tasks](https://arxiv.org/abs/2211.11869)".

## How to Use

### Installation
Install the requirements via `pip install -r requirements.txt`.

### Run Experiments
Run the experiments via `python -m run_experiment -c config`, where `config` is a configuration file in `./configs/` directory.
The available values are `{config_mnist, config_cifar10, config_spotify, config_recogym, config_personalization}`, which could be specified to recreate each of the presented numerical experiments.
Optionally, a custom experiment can be set up by changing or adding new configuration file.

### Load Experiments
All previously performed experiments are stored in `./data/` directory and can be recreated by loading via `python -m run_experiment -l exp_name`, where `exp_name` is the name of the experiment as it is saved in `./data/`.

### Results
![Reward values](https://github.com/sukiboo/policy_entropy/assets/38059493/9ced0db4-ca67-4319-b464-1ad04aac6a70)
![Entropy values](https://github.com/sukiboo/policy_entropy/assets/38059493/4a90feda-73dd-4809-89bc-772e9c4448de)



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



