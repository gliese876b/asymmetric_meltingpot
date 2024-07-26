from ray.rllib.algorithms.ppo import PPOConfig
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from asymmetric_ssds.utils.custom_ray_env import custom_env_creator
from ray.tune.registry import register_env
from ray.rllib.policy import policy as rllib_policy
from gymnasium import spaces
import json

def policy_mapping_function(agent_id, episode, worker, **kwargs):
    del kwargs
    return agent_id.replace("player", "agent")


register_env("asymmetric_commons_harvest__open", lambda config: custom_env_creator(config))

substrate_name = "asymmetric_commons_harvest__open"
#player_roles = substrate.get_config(substrate_name).default_player_roles
player_roles = ["consumer"] * 10
#player_roles = ["consumer_who_has_apple_reward_advantage"] * 10
#player_roles = ["consumer"] * 5 + ["consumer_who_has_apple_reward_advantage"] * 5
environment_config = {"substrate": substrate_name, "roles": player_roles}




test_env = custom_env_creator(environment_config)
policies = {}
for i, role in enumerate(environment_config["roles"]):
    rgb_shape = test_env.observation_space[f"player_{i}"].shape
    sprite_x = rgb_shape[0] // 8
    sprite_y = rgb_shape[1] // 8

    policies[f"agent_{i}"] = rllib_policy.PolicySpec(
        policy_class=None,  # use default policy
        observation_space=test_env.observation_space[f"player_{i}"],
        action_space=test_env.action_space[f"player_{i}"] if role != "consumer_who_cannot_zap" else spaces.Discrete(test_env.action_space[f"player_{i}"].n - 1), # TODO: find a better solution
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


config = PPOConfig()
config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3,
    train_batch_size=128)
config = config.resources(num_gpus=1)
config = config.env_runners(num_cpus_per_env_runner=1, num_env_runners=2, preprocessor_pref=None)
config = config.environment(env=environment_config["substrate"], env_config=environment_config)
config = config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_function)

# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build()
print("---- algo is built ----")
results = algo.train()
print(results)
