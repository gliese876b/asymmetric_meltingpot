import os
from meltingpot import substrate
import ray
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import argparse
from ray import air
from ray import tune
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy import policy as rllib_policy
from ray.tune.registry import register_env
from examples.rllib import utils
from asymmetric_ssds.utils.custom_ray_env import custom_env_creator
from ray.rllib.algorithms.algorithm import Algorithm
from asymmetric_ssds.utils.other import make_video_from_rgb_imgs, custom_log_creator
from asymmetric_ssds.utils.callbacks import AsymmetricSocialOutcomeCallbacks
from gymnasium import spaces

result_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../experiments/'))

def policy_mapping_function(agent_id, episode, worker, **kwargs):
    del kwargs
    #return agent_id.replace("player", "agent")
    return "agent_*"

def role_to_action_space(role, action_space):
    if role == 'consumer_who_cannot_zap':
        return spaces.Discrete(action_space.n - 1)
    return action_space

def train(config, checkpoint=None, num_iterations=50000, experiment_name=""):
    ray.init(address=None, log_to_driver=False)

    algo = config.build(logger_creator=custom_log_creator(result_directory, experiment_name))
    print("The algorithm is built. Logging directory: ", algo.logdir)

    debug_dir = "{}/checkpoints/".format(algo.logdir)

    if checkpoint:
        algo = Algorithm.from_checkpoint(checkpoint)
        print("The checkpoint is restored")

    for i in range(num_iterations):
        print("------------- Iteration", i+1, "-------------")
        print("Logging directory:", algo.logdir)
        start = time.time()
        results = algo.train()
        end = time.time()

        if (i+1) % 500 == 0:
            ma_checkpoint_dir = algo.save(checkpoint_dir=debug_dir)
            print("An Algorithm checkpoint has been created inside directory: ", ma_checkpoint_dir)

        print("date:", results['date'])
        print("timesteps_total:", results['timesteps_total'])
        print("num_env_steps_sampled_this_iter:", results['num_env_steps_sampled_this_iter'])
        print("num_env_steps_trained_this_iter:", results['num_env_steps_trained_this_iter'])
        for k in results['timers'].keys():
            print(k, results['timers'][k])
        print("time_this_iter_s:", results['time_this_iter_s'])
        print("iteration elapsed time (sec):", end - start)
        sys.stdout.flush()

    algo.stop()
    ray.shutdown()


def train_by_tune(config, checkpoint=None, num_iterations=50000, experiment_name=""):
    ray.init()
    stop = {
      "training_iteration": num_iterations,
    }
    return tune.Tuner(
      "DQN",
      param_space=config.to_dict(),
      run_config=air.RunConfig(stop=stop, verbose=1),
    ).fit()


def start_training(environment_config, experiment_name, alg_name):
    test_env = custom_env_creator(environment_config)

    policies = {}
    if alg_name == "DQN":
        """
        for i, role in enumerate(environment_config["roles"]):
            rgb_shape = test_env.observation_space[f"player_{i}"].shape
            sprite_x = rgb_shape[0] // 8
            sprite_y = rgb_shape[1] // 8

            policies[f"agent_{i}"] = rllib_policy.PolicySpec(
                policy_class=None,  # use default policy
                observation_space=test_env.observation_space[f"player_{i}"],
                action_space=role_to_action_space(role, test_env.action_space[f"player_{i}"]), # TODO: find a better solution
                config={
                    "model": {
                        "dim": rgb_shape[0],
                        "conv_filters": [[16, [8, 8], 8], [128, [sprite_x, sprite_y], 1]],
                        "conv_activation": "relu",
                    },
                }
            )
        """
        # Single Policy
        rgb_shape = test_env.observation_space["player_0"].shape
        sprite_x = rgb_shape[0] // 8
        sprite_y = rgb_shape[1] // 8

        policies["agent_*"] = rllib_policy.PolicySpec(
            policy_class=None,  # use default policy
            observation_space=test_env.observation_space["player_0"],
            action_space=role_to_action_space(environment_config["roles"][0], test_env.action_space["player_0"]), # TODO: find a better solution
            config={
                "model": {
                    "dim": rgb_shape[0],
                    "conv_filters": [[16, [8, 8], 8], [128, [sprite_x, sprite_y], 1]],
                    "conv_activation": "relu",
                },
            }
        )

        config = (
            DQNConfig()
            .environment(env=environment_config["substrate"], env_config=environment_config)
            .training(
                gamma=0.99,
                lr=1e-05,
                train_batch_size=128,
                replay_buffer_config={
                    'type': "MultiAgentPrioritizedReplayBuffer",
                    'prioritized_replay': -1,
                    'capacity': 50000,
                    'prioritized_replay_alpha': 0.6,
                    'prioritized_replay_beta': 0.4,
                    'prioritized_replay_eps': 1e-06,
                    'replay_sequence_length': 1,
                    'worker_side_prioritization': False
                }
            )
            .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_function)
            .resources(num_gpus=1)
            .env_runners(
                num_cpus_per_env_runner=4,
                num_env_runners=2,
                batch_mode="complete_episodes",
                #rollout_fragment_length=100,
                preprocessor_pref=None,
                explore=True,
                exploration_config={
                    "type": "EpsilonGreedy",
                    "initial_epsilon": 1.0,
                    "final_epsilon": 0.1,
                    "epsilon_timesteps": 1000000,
                }
            )
            #.reporting(metrics_num_episodes_for_smoothing=1)
            .framework("torch")
            .evaluation(
                evaluation_parallel_to_training=False,
                evaluation_sample_timeout_s=320,
                evaluation_interval=20,
                evaluation_duration=4,
                evaluation_num_env_runners=2,
                always_attach_evaluation_results=False
            )
            .fault_tolerance(recreate_failed_env_runners=True, restart_failed_sub_environments=True)
            .debugging(log_level="ERROR", logger_creator=custom_log_creator(result_directory, experiment_name))
            .callbacks(AsymmetricSocialOutcomeCallbacks)
        )
    elif alg_name == "PPO":
        for i, role in enumerate(environment_config["roles"]):
            rgb_shape = test_env.observation_space[f"player_{i}"].shape
            sprite_x = rgb_shape[0] // 8
            sprite_y = rgb_shape[1] // 8

            policies[f"agent_{i}"] = rllib_policy.PolicySpec(
                policy_class=None,  # use default policy
                observation_space=test_env.observation_space[f"player_{i}"],
                action_space=role_to_action_space(role, test_env.action_space[f"player_{i}"]), # TODO: find a better solution
                config={
                    "model": {
                        "dim": rgb_shape[0],
                        "conv_filters": [[16, [8, 8], 8], [128, [sprite_x, sprite_y], 1]],
                        "conv_activation": "relu",
                        "fcnet_hiddens": [64, 64],
                        "fcnet_activation": "relu",
                        "post_fcnet_hiddens": [256],
                        "use_lstm": True,
                        "lstm_cell_size": 256,
                        #"lstm_use_prev_action": True,
                        #"lstm_use_prev_reward": False
                    },
                })

        config = (
            PPOConfig()
            .environment(env=environment_config["substrate"], env_config=environment_config)
            .training(
                gamma=0.99,
                lr=1e-05,
                train_batch_size=4000
            )
            .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_function)
            .resources(num_gpus=1)
            .env_runners(
                num_cpus_per_env_runner=2,
                num_env_runners=2,
                #batch_mode="complete_episodes",
                rollout_fragment_length=100,
                preprocessor_pref=None,
            )
            #.reporting(metrics_num_episodes_for_smoothing=1)
            .framework("torch")
            .evaluation(
                evaluation_parallel_to_training=False,
                evaluation_sample_timeout_s=320,
                evaluation_interval=20,
                evaluation_duration=4,
                evaluation_num_env_runners=2,
                always_attach_evaluation_results=False
            )
            .fault_tolerance(recreate_failed_env_runners=True, restart_failed_sub_environments=True)
            .debugging(log_level="ERROR", logger_creator=custom_log_creator(result_directory, experiment_name))
            .callbacks(AsymmetricSocialOutcomeCallbacks)
        )


    # Training
    train(config, checkpoint=None, experiment_name=experiment_name)
    """
    train_by_tune(config, checkpoint=None, experiment_name=experiment_name)
    """


def start_searching(environment_config):
    test_env = custom_env_creator(environment_config)

    policies = {}
    for i, role in enumerate(environment_config["roles"]):
        rgb_shape = test_env.observation_space[f"player_{i}"].shape
        sprite_x = rgb_shape[0] // 8
        sprite_y = rgb_shape[1] // 8

        policies[f"agent_{i}"] = rllib_policy.PolicySpec(
            policy_class=None,  # use default policy
            observation_space=test_env.observation_space[f"player_{i}"],
            action_space=role_to_action_space(role, test_env.action_space[f"player_{i}"]), # TODO: find a better solution
            config={
                "model": {
                    "dim": rgb_shape[0],
                    "conv_filters": [[16, [8, 8], 8], [128, [sprite_x, sprite_y], 1]],
                    "conv_activation": "tanh",
                    #"fcnet_hiddens": [64, 64],
                    #"fcnet_activation": "relu",
                    #"post_fcnet_hiddens": [256],
                    #"use_lstm": tune.grid_search([True, False]),
                    #"lstm_cell_size": tune.grid_search([64, 256]),
                    #"lstm_use_prev_action": True,
                    #"lstm_use_prev_reward": False
                },
            })

    config = (
        DQNConfig()
        .environment(env=environment_config["substrate"], env_config=environment_config)
        .training(
            gamma=0.99,
            lr=1e-05,
            train_batch_size=128,
            replay_buffer_config={
                'type': "MultiAgentPrioritizedReplayBuffer",
                'prioritized_replay': -1,
                'capacity': 10000,
                'prioritized_replay_alpha': 0.6,
                'prioritized_replay_beta': 0.4,
                'prioritized_replay_eps': 1e-06,
                'replay_sequence_length': 1,
                'worker_side_prioritization': False})
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_function)
        .exploration(explore=True,
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.1,
                "epsilon_timesteps": 1000000,
            })
        .resources(num_gpus=1)
        .env_runners(
            num_cpus_per_env_runner=1,
            num_env_runners=2,
            batch_mode="complete_episodes",
            rollout_fragment_length=100,
            preprocessor_pref=None,
            explore=True,
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.1,
                "epsilon_timesteps": 1000000,
            }
        )
        .framework("torch")
        .fault_tolerance(recreate_failed_env_runners=True, restart_failed_sub_environments=True)
        .debugging(log_level="ERROR")
        .callbacks(AsymmetricSocialOutcomeCallbacks)
    )

    ray.init(address=None, log_to_driver=False, num_cpus=12)
    tuner = tune.Tuner(
        "DQN",
        run_config=air.RunConfig(stop={"timesteps_total": 1000000}),
        param_space=config.to_dict(),
    )

    results = tuner.fit()

    # Get the best result based on a particular metric.
    best_result = results.get_best_result(metric="episode_reward_min", mode="max")

    print(best_result)

    ray.shutdown()

def start_testing(environment_config, checkpoint=None, max_steps=1000, resize_width=1200, resize_height=800):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    test_env = custom_env_creator(environment_config)
    resize_dims=(resize_width, resize_height)

    lstm_used = {}
    lstm_cell_size = {}

    if checkpoint:
        debug_dir = '/'.join(checkpoint.split('/') + [timestr])
        checkpoint_name = checkpoint.split('/')[-1]

        algo = Algorithm.from_checkpoint(checkpoint)
        print("The checkpoint is restored:", checkpoint_name)
        policy_map = algo.workers.local_worker().policy_map

        for policy_id, policy in policy_map.items():
            if policy is not None:
                agent_id = policy_id.replace('agent', 'player')
                lstm_used[agent_id] = policy.config.get("model", {}).get("use_lstm", False)
                lstm_cell_size[agent_id] = policy.config.get("model", {}).get("lstm_cell_size", 0)
    else:
        print("Running an episode with random actions")

        debug_dir = os.path.join(result_directory, "{}_{}_test".format(environment_config['substrate'], timestr))
    os.makedirs(debug_dir, exist_ok=True)

    frames_top = []
    frames_pov = {}

    observations, infos = test_env.reset()
    agent_ids = observations.keys()

    agent_returns = {}

    states = {}
    prev_a = {}
    prev_r = {}
    for agent_id in agent_ids:
        if agent_id in lstm_used and lstm_used[agent_id]:
            states[agent_id] = [np.zeros([lstm_cell_size[agent_id]], np.float32) for _ in range(2)]
            prev_a[agent_id] = 0
            prev_r[agent_id] = 0.0

    frames_top.append(test_env.render())
    obs_shape = None
    for agent_id in agent_ids:
        rgb_obs = observations[agent_id]
        agent_returns[agent_id] = 0
        frames_pov[agent_id] = [rgb_obs]
        obs_shape = rgb_obs.shape
    step = 0
    start = time.time()
    for i in range(max_steps):
        print("====== Step", i, "======")
        actions = {agent_id: 0 for agent_id in agent_ids}
        if checkpoint:
            for agent_id, agent_obs in observations.items():
                policy_id = policy_mapping_function(agent_id, None, None)
                if agent_id in lstm_used and lstm_used[agent_id]:
                    actions[agent_id], states[agent_id], _ = algo.compute_single_action(agent_obs, states[agent_id], prev_action=prev_a[agent_id], prev_reward=prev_r[agent_id], policy_id=policy_id, explore=False)
                else:
                    actions[agent_id] = algo.compute_single_action(agent_obs, policy_id=policy_id, explore=False)
        else:
            for i, role in enumerate(environment_config['roles']):
                actions[f"player_{i}"] = role_to_action_space(role, test_env.action_space[f"player_{i}"]).sample() # random actions

        observations, rewards, terminateds, truncateds, infos = test_env.step(actions)
        frames_top.append(test_env.render())
        for agent_id in agent_ids:
            if agent_id in lstm_used and lstm_used[agent_id]:
                prev_a[agent_id] = actions[agent_id]
                prev_r[agent_id] = rewards[agent_id]
            if agent_id in rewards.keys():
                print("Agent:", agent_id, "takes action", actions[agent_id], "and gets reward", rewards[agent_id], infos[agent_id])
                agent_returns[agent_id] += rewards[agent_id]

        for agent_id in agent_ids:
            if agent_id in observations.keys():
                rgb_obs = observations[agent_id]
                frames_pov[agent_id].append(rgb_obs)
        step += 1
        if terminateds['__all__'] or truncateds['__all__']:
            break
    test_env.close()

    duration = time.time() - start

    for agent_id in agent_ids:
        print("Agent", agent_id, "collected", agent_returns[agent_id])

    top_video_name = "{}_{}".format(environment_config['substrate'], checkpoint_name) if checkpoint else "{}_RANDOM".format(environment_config['substrate'])
    make_video_from_rgb_imgs(frames_top, vid_path=debug_dir, video_name=top_video_name, resize=resize_dims)
    for agent_id in frames_pov.keys():
        pov_video_name = "{}_{}_{}".format(environment_config['substrate'], checkpoint_name, agent_id) if checkpoint else "{}_{}_RANDOM".format(environment_config['substrate'], agent_id)
        make_video_from_rgb_imgs(frames_pov[agent_id], vid_path=debug_dir, video_name=pov_video_name, resize=(obs_shape[0]*20, obs_shape[1]*20))
        break

    fps = step / duration
    print("{}: FPS: {:.2f} steps/second".format(environment_config['substrate'], fps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--mode", default="test", help="whether to train or test")
    parser.add_argument("-ch", "--checkpoint", default=None, help="checkpoint to use")
    parser.add_argument("-alg", "--algorithm", default="DQN", help="algorithm to train")
    args = vars(parser.parse_args())

    register_env("asymmetric_commons_harvest__open", lambda config: custom_env_creator(config))
    algorithm_name = args['algorithm']
    experiment_name = "{}_asymmetric_commons_harvest__open_10_consumers".format(algorithm_name)

    substrate_name = "asymmetric_commons_harvest__open"
    #player_roles = substrate.get_config(substrate_name).default_player_roles
    player_roles = ["consumer"] * 10
    #player_roles = ["consumer_who_has_apple_reward_advantage"] * 10
    #player_roles = ["consumer"] * 5 + ["consumer_who_has_apple_reward_advantage"] * 5
    environment_config = {"substrate": substrate_name, "roles": player_roles}

    if args['mode'] == 'train':
        start_training(environment_config, experiment_name, algorithm_name)
    elif args['mode'] == 'test':
        start_testing(environment_config, args['checkpoint'])
    elif args['mode'] == 'search':
        start_searching({"substrate": substrate_name, "roles": ["consumer"]})
