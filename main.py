# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
import time
from functools import partial
from typing import Tuple

import numpy as np
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
env = pb.env.Env()
cfg = Config()  # Extract the config once

init_key, traj_key = random.split(random.PRNGKey(0), (2, cfg.sims))

# Pass config directly to compiled functions
vinit = jit(vmap(partial(env.init, cfg)))
# vstep = jit(partial(step_fn, cfg))
# vtraj = jit(vmap(partial(traj_fn, cfg, vstep)))

print("Testing with config passed directly:")
tic = time.time()
obs, state = vinit(init_key)
state.hp.block_until_ready()
toc = time.time()
print(f"First call: {toc - tic:.4f} seconds")

tic = time.time()
obs, state = vinit(init_key)  # Same input
state.hp.block_until_ready()
toc = time.time()
print(f"Second call (same): {toc - tic:.4f} seconds")

tic = time.time()
obs, state = vinit(traj_key)  # Different input
state.hp.block_until_ready()
toc = time.time()
print(f"Third call (different): {toc - tic:.4f} seconds")
