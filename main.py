# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
import time
from functools import partial
from typing import Tuple

from jax import jit, lax, profiler, random, tree, vmap
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


def step_fn(state: State, rng: Array) -> Tuple[State, Tuple[State, Action]]:
    action = action_fn(cfg, rng)
    obs, state = env.step(cfg, rng, state, action)
    return state, (state, action)


def traj_fn(state, rng) -> Tuple[State, Tuple[State, Action]]:
    rngs = random.split(rng, cfg.steps)
    return lax.scan(step_fn, state, rngs)


# %% Main #####################################################################
env, cfg = pb.env.Env(), Config()
init_key, traj_key = random.split(random.PRNGKey(0), (2, cfg.sims))

init = vmap(jit(partial(env.init, cfg)))
traj = vmap(jit(traj_fn))

obs, state = init(init_key)
print(state.hp[0, 0])
state, (seq, action) = traj(state, init_key)
print(state.hp[0, 0])
