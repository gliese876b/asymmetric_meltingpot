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
                episode.user_data[agent_id]["is_zappeds"] = []
                episode.user_data[agent_id]["zap_actions"] = []
                episode.user_data[agent_id]["clean_actions"] = []
                episode.user_data[agent_id]["collected_apples"] = []
                episode.user_data[agent_id]["cleared_waste_cells"] = []
                episode.user_data[agent_id]["num_zapped_agents"] = []
                episode.user_data[agent_id]["role"] = None

        if isinstance(episode, EpisodeV2):
            for agent_id in episode.get_agents():
                episode.user_data[agent_id]["rewards"].append(episode._agent_reward_history[agent_id][-1])
                if 'ext_reward' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["ext_rewards"].append(episode._last_infos[agent_id].get('ext_reward', 0))
                if 'int_reward' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["int_rewards"].append(episode._last_infos[agent_id].get('int_reward', 0))
                if 'is_zapped' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["is_zappeds"].append(int(episode._last_infos[agent_id].get('is_zapped', 0)))
                if 'zap_action' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["zap_actions"].append(int(episode._last_infos[agent_id].get('zap_action', 0)))
                if 'clean_action' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["clean_actions"].append(int(episode._last_infos[agent_id].get('clean_action', 0)))
                if 'apple_collected' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["collected_apples"].append(int(episode._last_infos[agent_id].get('apple_collected', 0)))
                if 'number_of_cleared_waste_cells' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["cleared_waste_cells"].append(int(episode._last_infos[agent_id].get('number_of_cleared_waste_cells', 0)))
                if 'number_of_zapped_agents' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["num_zapped_agents"].append(int(episode._last_infos[agent_id].get('number_of_zapped_agents', 0)))
                if 'role' in episode._last_infos[agent_id].keys():
                    episode.user_data[agent_id]["role"] = episode._last_infos[agent_id].get('role', None)
        else:
            for agent_id in episode.get_agents():
                episode.user_data[agent_id]["rewards"].append(episode.last_reward_for(agent_id))
                if 'ext_reward' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["ext_rewards"].append(episode.last_info_for(agent_id).get('ext_reward', 0))
                if 'int_reward' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["int_rewards"].append(episode.last_info_for(agent_id).get('int_reward', 0))
                if 'is_zapped' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["is_zappeds"].append(int(episode.last_info_for(agent_id).get('is_zapped', 0)))
                if 'zap_action' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["zap_actions"].append(int(episode.last_info_for(agent_id).get('zap_action', 0)))
                if 'clean_action' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["clean_actions"].append(int(episode.last_info_for(agent_id).get('clean_action', 0)))
                if 'apple_collected' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["collected_apples"].append(int(episode.last_info_for(agent_id).get('apple_collected', 0)))
                if 'number_of_cleared_waste_cells' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["cleared_waste_cells"].append(int(episode.last_info_for(agent_id).get('number_of_cleared_waste_cells', 0)))
                if 'number_of_zapped_agents' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["num_zapped_agents"].append(int(episode.last_info_for(agent_id).get('number_of_zapped_agents', 0)))
                if 'role' in episode.last_info_for(agent_id).keys():
                    episode.user_data[agent_id]["role"] = episode.last_info_for(agent_id).get('role', None)

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
        agent_roles = {}
        for agent_id in episode.get_agents():
            agent_role = episode.user_data[agent_id]["role"]
            if agent_role:
                if agent_role not in agent_roles.keys():
                    agent_roles[agent_role] = []
                agent_roles[agent_role].append(agent_id)

            if len(episode.user_data[agent_id]["ext_rewards"]) > 0:
                agent_returns[agent_id] = sum(episode.user_data[agent_id]["ext_rewards"])
            else:
                agent_returns[agent_id] = sum(episode.user_data[agent_id]["rewards"])

            if len(episode.user_data[agent_id]["ext_rewards"]) > 0:
                episode.custom_metrics["ext_return_{}".format(agent_id)] = sum(episode.user_data[agent_id]["ext_rewards"])
            if len(episode.user_data[agent_id]["int_rewards"]) > 0:
                episode.custom_metrics["int_return_{}".format(agent_id)] = sum(episode.user_data[agent_id]["int_rewards"])

        collective_return = sum(agent_returns.values())
        collective_return_by_role = {}
        for agent_id in episode.get_agents():
            agent_role = episode.user_data[agent_id]["role"]
            if agent_role:
                if agent_role not in collective_return_by_role.keys():
                    collective_return_by_role[agent_role] = 0
                collective_return_by_role[agent_role] += agent_returns[agent_id]

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
            temp_sum += sum(episode.user_data[agent_id]["is_zappeds"])
        peace = N - (temp_sum / T)

        episode.custom_metrics["peace"] = peace

        temp_sum = 0
        for agent_id in episode.get_agents():
            temp_sum += sum(episode.user_data[agent_id]["zap_actions"])
        episode.custom_metrics["zap_per_agent"] = temp_sum / N

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
            temp_sum += sum(episode.user_data[agent_id]["num_zapped_agents"]) / sum(episode.user_data[agent_id]["zap_actions"]) if sum(episode.user_data[agent_id]["zap_actions"]) > 0 else 0
        episode.custom_metrics["average_zapping_accuracy"] = temp_sum / N

        for role, role_agents in agent_roles.items():
            episode.custom_metrics["collective_return_{}".format(role)] = collective_return_by_role[role]
            episode.custom_metrics["average_return_per_agent_{}".format(role)] = collective_return_by_role[role] / len(role_agents)
            episode.custom_metrics["efficiency_{}".format(role)] = collective_return_by_role[role] / T

            equality_by_role = 0
            for i in role_agents:
                for j in role_agents:
                    equality_by_role += abs(agent_returns[i] - agent_returns[j])
            equality_by_role = 1 - ( equality_by_role / (2 * N * (collective_return_by_role[role] + 0.00001)) )
            episode.custom_metrics["equality_{}".format(role)] = equality_by_role

            sustainability_by_role = 0
            for agent_id in role_agents:
                ti = []
                for t in range(T):
                    if len(episode.user_data[agent_id]["ext_rewards"]) > 0:
                        if episode.user_data[agent_id]["ext_rewards"][t] > 0:
                            ti.append(t)
                    elif episode.user_data[agent_id]["rewards"][t] > 0:
                        ti.append(t)
                sustainability_by_role += sum(ti) / len(ti) if len(ti) > 0 else 0
            sustainability_by_role = sustainability_by_role / len(role_agents)
            episode.custom_metrics["sustainability_{}".format(role)] = sustainability_by_role

            temp_sum = 0
            for agent_id in role_agents:
                temp_sum += sum(episode.user_data[agent_id]["is_zappeds"])
            peace_by_role = len(role_agents) - (temp_sum / T)

            episode.custom_metrics["peace_{}".format(role)] = peace_by_role

            temp_sum = 0
            for agent_id in role_agents:
                temp_sum += sum(episode.user_data[agent_id]["zap_actions"])
            episode.custom_metrics["zap_per_agent_{}".format(role)] = temp_sum / N

            temp_sum = 0
            for agent_id in role_agents:
                temp_sum += sum(episode.user_data[agent_id]["clean_actions"])
            episode.custom_metrics["clean_per_agent_{}".format(role)] = temp_sum / N

            temp_sum = 0
            for agent_id in role_agents:
                temp_sum += sum(episode.user_data[agent_id]["collected_apples"])
            episode.custom_metrics["apples_per_agent_{}".format(role)] = temp_sum / N

            temp_sum = 0
            for agent_id in role_agents:
                temp_sum += sum(episode.user_data[agent_id]["cleared_waste_cells"])
            episode.custom_metrics["cleared_waste_per_agent_{}".format(role)] = temp_sum / N

            temp_sum = 0
            for agent_id in role_agents:
                temp_sum += sum(episode.user_data[agent_id]["cleared_waste_cells"]) / sum(episode.user_data[agent_id]["clean_actions"]) if sum(episode.user_data[agent_id]["clean_actions"]) > 0 else 0
            episode.custom_metrics["average_cleaning_accuracy_{}".format(role)] = temp_sum / N

            temp_sum = 0
            for agent_id in role_agents:
                temp_sum += sum(episode.user_data[agent_id]["num_zapped_agents"]) / sum(episode.user_data[agent_id]["zap_actions"]) if sum(episode.user_data[agent_id]["zap_actions"]) > 0 else 0
            episode.custom_metrics["average_zapping_accuracy_{}".format(role)] = temp_sum / N
