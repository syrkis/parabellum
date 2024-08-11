# ludens.py
#    script for fucking around and finding out
# by: Noah Syrkis

# %% Imports
import parabellum as pb


# %% Constants
num_seeds = 100
num_areas = 10
num_units = 1000
num_meter = 100  # meters

# %% test
places = ["Vesterbro, Copenhagen, Denmark"] * 10
print(places)
mask, base = pb.terrain_fn(places[0], 100)
scenario = pb.scenarios["default"]
env = pb.Environment(scenario)

# rng, key = random.split(random.PRNGKey(seed := 0))
# obs, state = env.reset(key)
#
# print(obs.keys())
#
# state_seq = []
# for i in range(100):
# rng, act_rng, step_key = random.split(rng, 3)
# act_key = random.split(act_rng, len(env.agents))
# actions = {a: jnp.zeros_like(env.action_space(a).sample(k)) +2 for a, k in zip(env.agents, act_key)}
# state_seq.append((step_key, state, actions))
# obs, state, reward, done, info = env.step(step_key, state, actions)


# vis = pb.Visualizer(env, state_seq, skin=pb.Skin(maskmap=mask))
# vis.animate()
