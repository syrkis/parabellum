# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
from jax import random, lax
import parabellum as pb
from jaxtyping import Array
from functools import partial


# %% Functions ###############################################################
def action_fn(cfg: pb.types.Config, rng: Array) -> pb.types.Action:
    pos = random.normal((cfg.length, 2)) * cfg.rules.reach[cfg.types][..., None]
    types = random.randint(rng, (cfg.length,), minval=0, maxval=3)
    return pb.types.Action(pos=pos, types=types)


def step_fn(env: pb.env.Env, cfg: pb.types.Config, state: pb.types.State, rng: Array):
    action = action_fn(cfg, rng)
    obs, state = env.step(rng, cfg, state, action)
    return state, (state, action)


# %% Main #####################################################################
cfg = pb.types.Config()
env = pb.env.Env(cfg=cfg)
rng, key = random.split(random.PRNGKey(0))
obs, state = env.reset(key, cfg)
rngs = random.split(rng, cfg.steps)
state, (seq, action) = lax.scan(partial(step_fn, env, cfg), state, rngs)
# pb.utils.svg_fn(cfg, seq, action, fps=10)
