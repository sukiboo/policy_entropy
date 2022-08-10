
from environments.classification_env import ClassificationEnv
from environments.spotify_env import SpotifyEnv
from environments.recogym_env import RecogymEnv
from environments.personalization_env import PersonalizationEnv


def create_environment(env_name, seed=0):
    '''create contextual bandit environment'''
    if env_name in ['mnist', 'cifar10', 'cifar100']:
        return ClassificationEnv(env_name, seed)
    elif env_name == 'spotify':
        return SpotifyEnv(seed=seed)
    elif env_name == 'recogym':
        return RecogymEnv(seed=seed)
    elif env_name == 'personalization':
        return PersonalizationEnv(seed=seed)
    else:
        raise SystemExit(f'\nenvironment {env_name} is not implemented...')


if __name__ == '__main__':
    env = create_environment(env_name='personalization', seed=0)

