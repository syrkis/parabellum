# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
import time
from functools import partial
from typing import Tuple

from jax import block_until_ready, jit, lax, profiler, random, tree, vmap
from jax import numpy as jnp
from jaxtyping import Array

import parabellum as pb
from parabellum.types import Action, State, Config


# %% Functions ###############################################################
# %% Functions ###############################################################
def action_fn(cfg: Config, rng: Array) -> Action:
    pos = random.normal(rng, (cfg.length, 2)) * cfg.rules.reach[cfg.types][..., None]
    kind = random.randint(rng, (cfg.length,), minval=0, maxval=3)
    return Action(pos=pos, kind=kind)


def step_fn(cfg: Config, state: State, rng: Array) -> Tuple[State, Tuple[State, Action]]:
    action = action_fn(cfg, rng)
    obs, state = env.step(cfg, rng, state, action)
    return state, (state, action)


def traj_fn(cfg: Config, step, state, rng) -> Tuple[State, Tuple[State, Action]]:
    rngs = random.split(rng, cfg.steps)
    return lax.scan(step, state, rngs)


# %% Main #####################################################################
env, cfg = pb.env.Env(), Config()
init = vmap(jit(partial(env.init, cfg)))
# step = partial(env.step, cfg)
init_key, traj_key = random.split(random.PRNGKey(0), (2, cfg.sims))

print("Testing with config passed directly:")
tic = time.time()
obs, state = init(init_key)
state.hp.block_until_ready()
toc = time.time()
print(f"First call: {toc - tic:.4f} seconds")

tic = time.time()
obs, state = init(init_key)
state.hp.block_until_ready()
toc = time.time()
print(f"Second call (same): {toc - tic:.4f} seconds")

tic = time.time()
obs, state = init(traj_key)
state.hp.block_until_ready()
toc = time.time()
print(f"Third call (different): {toc - tic:.4f} seconds")
