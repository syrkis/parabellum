# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
import jax.numpy as jnp
from jax import random, lax, debug
import parabellum as pb
from omegaconf import DictConfig
from jax_tqdm import scan_tqdm


# %% Setup #################################################################
loc = dict(place="Palazzo della Civiltà Italiana, Rome, Italy", size=64)
red = dict(infantry=6, armor=6, airplane=6)
blue = dict(infantry=6, armor=6, airplane=6)
cfg = DictConfig(dict(steps=300, knn=4, blue=blue, red=red) | loc)

# Access values using dot notation
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
rng, key = random.split(random.PRNGKey(0))


# %% Functions ###############################################################
def action_fn(rng):
    coord = (random.uniform(rng, (env.num_units, 2)) - 0.5) * scene.unit_type_reach[scene.unit_types][..., None]
    types = random.randint(rng, (env.num_units,), minval=0, maxval=3)
    return pb.types.Action(coord=coord, types=types)


@scan_tqdm(cfg.steps)
def step(state, inputs):
    idx, rng = inputs
    action = action_fn(rng)
    obs, state = env.step(rng, scene, state, action)
    return state, (state, action)


# %% Main #####################################################################
obs, state = env.reset(key, scene)
rngs = random.split(rng, cfg.steps)
state, (seq, action) = lax.scan(step, state, (jnp.arange(cfg.steps), rngs))
pb.utils.svg_fn(scene, seq, action)
