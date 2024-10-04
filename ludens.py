# %% ludens.py
#    play with the world
# by: Noah Syrkis

# %% Imports
import parabellum as pb
from jax import random, vmap, jit
from tqdm import tqdm

# %% Constants
place = "Vesterbro, Copenhagen, Denmark"
scenario = pb.env.scenario_fn(place)
env = pb.Environment(scenario=scenario)

# %% Initialize
rng, step_key, act_rng = random.split(random.PRNGKey(0), 3)
obs, state = env.reset(rng)

# %% Compile
step = jit(env.step)


# %% Play
for i in tqdm(range(100)):
    act_rng, *keys = random.split(act_rng, 1 + len(env.agents))
    acts = {a: env.action_space(a).sample(keys[i]) for i, a in enumerate(env.agents)}
    obs, state, reward, done, infos = step(step_key, state, acts)
