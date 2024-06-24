from typing import Tuple
from typing import Any, Mapping
import dm_env
import dmlab2d
from gymnasium import spaces
from meltingpot import substrate
from meltingpot.utils.policies import policy
from ml_collections import config_dict
import numpy as np
from ray.rllib import algorithms
from ray.rllib.env import multi_agent_env
from ray.rllib.policy import sample_batch

from examples.gym import utils

PLAYER_STR_FORMAT = 'player_{index}'

class CustomRayEnv(multi_agent_env.MultiAgentEnv):
    """An adapter between the Melting Pot substrates and RLLib MultiAgentEnv."""

    # Metadata is required by the gym `Env` class that we are extending, to show
    # which modes the `render` method supports.
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, env: dmlab2d.Environment):
        """Initializes the instance.

        Args:
          env: dmlab2d environment to wrap. Will be closed when this wrapper closes.
        """
        self._env = env
        self._num_players = len(self._env.observation_spec())
        self._ordered_agent_ids = [
            PLAYER_STR_FORMAT.format(index=index)
            for index in range(self._num_players)
        ]
        # RLLib requires environments to have the following member variables:
        # observation_space, action_space, and _agent_ids
        self._agent_ids = set(self._ordered_agent_ids)
        # RLLib expects a dictionary of agent_id to observation or action,
        # Melting Pot uses a tuple, so we convert
        self.observation_space = self._convert_spaces_tuple_to_dict(
            utils.spec_to_space(self._env.observation_spec()),
            remove_world_observations=True)
        self.action_space = self._convert_spaces_tuple_to_dict(
            utils.spec_to_space(self._env.action_spec()))
        super().__init__()

    def _convert_spaces_tuple_to_dict(
        self,
        input_tuple: spaces.Tuple,
        remove_world_observations: bool = False) -> spaces.Dict:
        """Returns spaces tuple converted to a dictionary.

        Args:
          input_tuple: tuple to convert.
          remove_world_observations: If True will remove non-player observations.
        """
        return spaces.Dict({
            agent_id: (utils.remove_world_observations_from_space(input_tuple[i])['RGB']
                       if remove_world_observations else input_tuple[i])
            for i, agent_id in enumerate(self._ordered_agent_ids)
        })



    def reset(self, *args, **kwargs):
        """See base class."""
        timestep = self._env.reset()
        infos = {
            agent_id: {}
            for index, agent_id in enumerate(self._ordered_agent_ids)
        }
        return convert_timestep_to_observations(timestep), convert_timestep_to_infos(timestep)

    def step(self, action_dict):
        """See base class."""
        actions = [action_dict[agent_id] for agent_id in self._ordered_agent_ids]
        timestep = self._env.step(actions)
        rewards = {
            agent_id: timestep.reward[index]
            for index, agent_id in enumerate(self._ordered_agent_ids)
        }
        dones = {
            agent_id: timestep.last()
            for index, agent_id in enumerate(self._ordered_agent_ids)
        }
        dones['__all__'] = timestep.last()

        observations = convert_timestep_to_observations(timestep)
        infos = convert_timestep_to_infos(timestep)
        return observations, rewards, dones, dones, infos

    def close(self):
        """See base class."""
        self._env.close()

    def get_dmlab2d_env(self):
        """Returns the underlying DM Lab2D environment."""
        return self._env

    def render(self) -> np.ndarray:
        """Render the environment.

        This allows you to set `record_env` in your training config, to record
        videos of gameplay.

        Returns:
            np.ndarray: This returns a numpy.ndarray with shape (x, y, 3),
            representing RGB values for an x-by-y pixel image, suitable for turning
            into a video.
        """
        observation = self._env.observation()
        world_rgb = observation[0]['WORLD.RGB']

        # RGB mode is used for recording videos
        return world_rgb

def convert_timestep_to_observations(timestep: dm_env.TimeStep) -> Mapping[str, Any]:
    gym_observations = {}
    for index, observation in enumerate(timestep.observation):
        gym_observations[PLAYER_STR_FORMAT.format(index=index)] = observation['RGB']
    return gym_observations

def convert_timestep_to_infos(timestep: dm_env.TimeStep) -> Mapping[str, Any]:
    gym_infos = {}
    for index, observation in enumerate(timestep.observation):
        gym_infos[PLAYER_STR_FORMAT.format(index=index)] = {}
        if 'PLAYER_CLEANED' in observation:
            gym_infos[PLAYER_STR_FORMAT.format(index=index)]['clean_action'] = (observation['PLAYER_CLEANED'] > 0)
        if 'PLAYER_ATE_APPLE' in observation:
            gym_infos[PLAYER_STR_FORMAT.format(index=index)]['apple_collected'] = (observation['PLAYER_ATE_APPLE'] > 0)
        if 'PLAYER_ZAPPED' in observation:
            gym_infos[PLAYER_STR_FORMAT.format(index=index)]['tagged'] = (observation['PLAYER_ZAPPED'] > 0)
        if 'NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP' in observation:
            gym_infos[PLAYER_STR_FORMAT.format(index=index)]['number_of_tagged_agents'] = observation['NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP'].sum()
    return gym_infos

def custom_env_creator(env_config):
    """Outputs an environment for registering."""
    env_config = config_dict.ConfigDict(env_config)
    env = substrate.build(env_config['substrate'], roles=env_config['roles'])
    env = CustomRayEnv(env)
    return env
