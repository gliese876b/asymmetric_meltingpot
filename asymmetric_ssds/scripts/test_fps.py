import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from meltingpot.python import substrate
from asymmetric_ssds.utils.custom_ray_env import *
import time

available_subtrates = ['prisoners_dilemma_in_the_matrix__repeated', 'daycare', 'commons_harvest__closed', 'collaborative_cooking__forced', 'fruit_market__concentric_rivers', 'bach_or_stravinsky_in_the_matrix__repeated', 'predator_prey__alley_hunt', 'commons_harvest__open', 'bach_or_stravinsky_in_the_matrix__arena', 'stag_hunt_in_the_matrix__arena', 'chemistry__three_metabolic_cycles_with_plentiful_distractors', 'predator_prey__open', 'coins', 'running_with_scissors_in_the_matrix__arena', 'factory_commons__either_or', 'chemistry__three_metabolic_cycles', 'commons_harvest__partnership', 'coop_mining', 'stag_hunt_in_the_matrix__repeated', 'externality_mushrooms__dense', 'territory__open', 'chemistry__two_metabolic_cycles', 'paintball__capture_the_flag', 'collaborative_cooking__cramped', 'chicken_in_the_matrix__repeated', 'gift_refinements', 'collaborative_cooking__ring', 'pure_coordination_in_the_matrix__repeated', 'territory__rooms', 'collaborative_cooking__crowded', 'rationalizable_coordination_in_the_matrix__repeated', 'boat_race__eight_races', 'predator_prey__orchard', 'collaborative_cooking__figure_eight', 'clean_up', 'chemistry__two_metabolic_cycles_with_distractors', 'running_with_scissors_in_the_matrix__repeated', 'prisoners_dilemma_in_the_matrix__arena', 'rationalizable_coordination_in_the_matrix__arena', 'paintball__king_of_the_hill', 'collaborative_cooking__circuit', 'allelopathic_harvest__open', 'chicken_in_the_matrix__arena', 'predator_prey__random_forest', 'hidden_agenda', 'collaborative_cooking__asymmetric', 'running_with_scissors_in_the_matrix__one_shot', 'pure_coordination_in_the_matrix__arena', 'territory__inside_out']

#for substrate_name in sorted(['allelopathic_harvest__open', 'bach_or_stravinsky_in_the_matrix__arena', 'paintball__capture_the_flag', 'chicken_in_the_matrix__arena', 'clean_up', 'commons_harvest__open', 'commons_harvest__closed', 'prisoners_dilemma_in_the_matrix__arena', 'stag_hunt_in_the_matrix__arena']):
for substrate_name in sorted(['asymmetric_commons_harvest__open']):
    player_roles = substrate.get_config(substrate_name).default_player_roles
    env_config = {"substrate": substrate_name, "roles": player_roles}
    env = custom_env_creator(env_config)

    print(env.observation_space)
    print(env.action_space)

    env.reset()

    start_time = time.time()
    step = 0
    done = False
    while step < 10000:
        actions = env.action_space.sample()
        observations, rewards, terminateds, truncateds, infos = env.step(actions)
        if terminateds['__all__'] or truncateds['__all__']:
            print("episode ended", step)
            env.reset()
        step += 1
    duration = time.time() - start_time
    fps = step / duration
    print("{}: FPS: {:.2f} steps/second, duration: {}".format(substrate_name, fps, duration))
