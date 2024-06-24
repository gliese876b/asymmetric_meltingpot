import gymnasium as gym
from gymnasium import spaces
import ray
from typing import Dict, Tuple, Union
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env import BaseEnv
import numpy as np

class AsymmetricSocialOutcomeCallbacks(DefaultCallbacks):
    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )

        # Initialization is here since episode.get_agents() returns empty in on_episode_start()
        for agent_id in episode.get_agents():
            if agent_id not in episode.user_data.keys():
                episode.user_data[agent_id] = {}
                episode.user_data[agent_id]["rewards"] = []
                episode.user_data[agent_id]["ext_rewards"] = []
                episode.user_data[agent_id]["int_rewards"] = []
                episode.user_data[agent_id]["taggeds"] = []
                episode.user_data[agent_id]["tag_actions"] = []
                episode.user_data[agent_id]["clean_actions"] = []
                episode.user_data[agent_id]["collected_apples"] = []
                episode.user_data[agent_id]["cleared_waste_cells"] = []
                episode.user_data[agent_id]["tagged_agents"] = []
                episode.user_data[agent_id]["type"] = None

        if isinstance(episode, EpisodeV2):
            for agent_id in episode.get_agents():
                episode.user_data[agent_id]["rewards"].append(episode._agent_reward_history[agent_id][-1])
                if 'ext_reward' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["ext_rewards"].append(episode._last_infos[agent_id].get('ext_reward', 0))
                if 'int_reward' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["int_rewards"].append(episode._last_infos[agent_id].get('int_reward', 0))
                if 'tagged' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["taggeds"].append(int(episode._last_infos[agent_id].get('tagged', 0)))
                if 'tag_action' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["tag_actions"].append(int(episode._last_infos[agent_id].get('tag_action', 0)))
                if 'clean_action' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["clean_actions"].append(int(episode._last_infos[agent_id].get('clean_action', 0)))
                if 'apple_collected' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["collected_apples"].append(int(episode._last_infos[agent_id].get('apple_collected', 0)))
                if 'number_of_cleared_waste_cells' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["cleared_waste_cells"].append(int(episode._last_infos[agent_id].get('number_of_cleared_waste_cells', 0)))
                if 'number_of_tagged_agents' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["tagged_agents"].append(int(episode._last_infos[agent_id].get('number_of_tagged_agents', 0)))
                if 'type' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["type"] = episode._last_infos[agent_id].get('type', None)
        else:
            for agent_id in episode.get_agents():
                episode.user_data[agent_id]["rewards"].append(episode.last_reward_for(agent_id))
                if 'ext_reward' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["ext_rewards"].append(episode.last_info_for(agent_id).get('ext_reward', 0))
                if 'int_reward' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["int_rewards"].append(episode.last_info_for(agent_id).get('int_reward', 0))
                if 'tagged' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["taggeds"].append(int(episode.last_info_for(agent_id).get('tagged', 0)))
                if 'tag_action' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["tag_actions"].append(int(episode.last_info_for(agent_id).get('tag_action', 0)))
                if 'clean_action' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["clean_actions"].append(int(episode.last_info_for(agent_id).get('clean_action', 0)))
                if 'apple_collected' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["collected_apples"].append(int(episode.last_info_for(agent_id).get('apple_collected', 0)))
                if 'number_of_cleared_waste_cells' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["cleared_waste_cells"].append(int(episode.last_info_for(agent_id).get('number_of_cleared_waste_cells', 0)))
                if 'number_of_tagged_agents' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["tagged_agents"].append(int(episode.last_info_for(agent_id).get('number_of_tagged_agents', 0)))
                if 'type' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["type"] = episode.last_info_for(agent_id).get('type', None)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs
    ):

        T = episode.length
        N = len(episode.get_agents())

        agent_returns = {}
        agent_types = {}
        for agent_id in episode.get_agents():
            agent_type = episode.user_data[agent_id]["type"]
            if agent_type:
                if agent_type not in agent_types.keys():
                    agent_types[agent_type] = []
                agent_types[agent_type].append(agent_id)

            if len(episode.user_data[agent_id]["ext_rewards"]) > 0:
                agent_returns[agent_id] = sum(episode.user_data[agent_id]["ext_rewards"])
            else:
                agent_returns[agent_id] = sum(episode.user_data[agent_id]["rewards"])

            if len(episode.user_data[agent_id]["ext_rewards"]) > 0:
                episode.custom_metrics["ext_return_{}".format(agent_id)] = sum(episode.user_data[agent_id]["ext_rewards"])
            if len(episode.user_data[agent_id]["int_rewards"]) > 0:
                episode.custom_metrics["int_return_{}".format(agent_id)] = sum(episode.user_data[agent_id]["int_rewards"])

        collective_return = sum(agent_returns.values())
        collective_return_by_type = {}
        for agent_id in episode.get_agents():
            agent_type = episode.user_data[agent_id]["type"]
            if agent_type:
                if agent_type not in collective_return_by_type.keys():
                    collective_return_by_type[agent_type] = 0
                collective_return_by_type[agent_type] += agent_returns[agent_id]

        episode.custom_metrics["collective_return"] = collective_return
        episode.custom_metrics["average_return_per_agent"] = collective_return / N
        episode.custom_metrics["efficiency"] = collective_return / T

        equality = 0
        for i in episode.get_agents():
            for j in episode.get_agents():
                equality += abs(agent_returns[i] - agent_returns[j])
        equality = 1 - ( equality / (2 * N * (collective_return + 0.00001)) ) # to avoid nonzero division when collective_return = 0
        episode.custom_metrics["equality"] = equality

        sustainability = 0
        for agent_id in episode.get_agents():
            ti = []
            for t in range(T):
                if len(episode.user_data[agent_id]["ext_rewards"]) > 0:
                    if episode.user_data[agent_id]["ext_rewards"][t] > 0:
                        ti.append(t)
                elif episode.user_data[agent_id]["rewards"][t] > 0:
                    ti.append(t)
            sustainability += sum(ti) / len(ti) if len(ti) > 0 else 0
        sustainability = sustainability / N

        episode.custom_metrics["sustainability"] = sustainability

        temp_sum = 0
        for agent_id in episode.get_agents():
            temp_sum += sum(episode.user_data[agent_id]["taggeds"])
        peace = N - (temp_sum / T)

        episode.custom_metrics["peace"] = peace

        temp_sum = 0
        for agent_id in episode.get_agents():
            temp_sum += sum(episode.user_data[agent_id]["tag_actions"])
        episode.custom_metrics["tag_per_agent"] = temp_sum / N

        temp_sum = 0
        for agent_id in episode.get_agents():
            temp_sum += sum(episode.user_data[agent_id]["clean_actions"])
        episode.custom_metrics["clean_per_agent"] = temp_sum / N

        temp_sum = 0
        for agent_id in episode.get_agents():
            temp_sum += sum(episode.user_data[agent_id]["collected_apples"])
        episode.custom_metrics["apples_per_agent"] = temp_sum / N

        temp_sum = 0
        for agent_id in episode.get_agents():
            temp_sum += sum(episode.user_data[agent_id]["cleared_waste_cells"])
        episode.custom_metrics["cleared_waste_per_agent"] = temp_sum / N

        temp_sum = 0
        for agent_id in episode.get_agents():
            temp_sum += sum(episode.user_data[agent_id]["cleared_waste_cells"]) / sum(episode.user_data[agent_id]["clean_actions"]) if sum(episode.user_data[agent_id]["clean_actions"]) > 0 else 0
        episode.custom_metrics["average_cleaning_accuracy"] = temp_sum / N

        temp_sum = 0
        for agent_id in episode.get_agents():
            temp_sum += sum(episode.user_data[agent_id]["tagged_agents"]) / sum(episode.user_data[agent_id]["tag_actions"]) if sum(episode.user_data[agent_id]["tag_actions"]) > 0 else 0
        episode.custom_metrics["average_tagging_accuracy"] = temp_sum / N

        for type, type_agents in agent_types.items():
            episode.custom_metrics["collective_return_{}".format(type)] = collective_return_by_type[type]
            episode.custom_metrics["average_return_per_agent_{}".format(type)] = collective_return_by_type[type] / len(type_agents)
            episode.custom_metrics["efficiency_{}".format(type)] = collective_return_by_type[type] / T

            equality_by_type = 0
            for i in type_agents:
                for j in type_agents:
                    equality_by_type += abs(agent_returns[i] - agent_returns[j])
            equality_by_type = 1 - ( equality_by_type / (2 * N * (collective_return_by_type[type] + 0.00001)) )
            episode.custom_metrics["equality_{}".format(type)] = equality_by_type

            sustainability_by_type = 0
            for agent_id in type_agents:
                ti = []
                for t in range(T):
                    if len(episode.user_data[agent_id]["ext_rewards"]) > 0:
                        if episode.user_data[agent_id]["ext_rewards"][t] > 0:
                            ti.append(t)
                    elif episode.user_data[agent_id]["rewards"][t] > 0:
                        ti.append(t)
                sustainability_by_type += sum(ti) / len(ti) if len(ti) > 0 else 0
            sustainability_by_type = sustainability_by_type / len(type_agents)
            episode.custom_metrics["sustainability_{}".format(type)] = sustainability_by_type

            temp_sum = 0
            for agent_id in type_agents:
                temp_sum += sum(episode.user_data[agent_id]["taggeds"])
            peace_by_type = len(type_agents) - (temp_sum / T)

            episode.custom_metrics["peace_{}".format(type)] = peace_by_type

            temp_sum = 0
            for agent_id in type_agents:
                temp_sum += sum(episode.user_data[agent_id]["tag_actions"])
            episode.custom_metrics["tag_per_agent_{}".format(type)] = temp_sum / N

            temp_sum = 0
            for agent_id in type_agents:
                temp_sum += sum(episode.user_data[agent_id]["clean_actions"])
            episode.custom_metrics["clean_per_agent_{}".format(type)] = temp_sum / N

            temp_sum = 0
            for agent_id in type_agents:
                temp_sum += sum(episode.user_data[agent_id]["collected_apples"])
            episode.custom_metrics["apples_per_agent_{}".format(type)] = temp_sum / N

            temp_sum = 0
            for agent_id in type_agents:
                temp_sum += sum(episode.user_data[agent_id]["cleared_waste_cells"])
            episode.custom_metrics["cleared_waste_per_agent_{}".format(type)] = temp_sum / N

            temp_sum = 0
            for agent_id in type_agents:
                temp_sum += sum(episode.user_data[agent_id]["cleared_waste_cells"]) / sum(episode.user_data[agent_id]["clean_actions"]) if sum(episode.user_data[agent_id]["clean_actions"]) > 0 else 0
            episode.custom_metrics["average_cleaning_accuracy_{}".format(type)] = temp_sum / N

            temp_sum = 0
            for agent_id in type_agents:
                temp_sum += sum(episode.user_data[agent_id]["tagged_agents"]) / sum(episode.user_data[agent_id]["tag_actions"]) if sum(episode.user_data[agent_id]["tag_actions"]) > 0 else 0
            episode.custom_metrics["average_tagging_accuracy_{}".format(type)] = temp_sum / N
