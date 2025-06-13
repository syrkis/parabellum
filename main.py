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
from parabellum.types import Action, State


# %% Functions ###############################################################
# @profiler.annotate_function
def action_fn(env: pb.env.Env, rng: Array) -> Action:
    pos = random.normal(rng, (env.length, 2)) * env.rules.reach[env.types][..., None]
    kind = random.randint(rng, (env.length,), minval=0, maxval=3)
    return Action(pos=pos, kind=kind)


# @profiler.annotate_function
def step_fn(env: pb.env.Env, state: State, rng: Array) -> Tuple[State, Tuple[State, Action]]:
    action = action_fn(env, rng)
    obs, state = env.step(rng, state, action)
    return state, (state, action)


# @profiler.annotate_function
def traj_fn(env, step, state, rng) -> Tuple[State, Tuple[State, Action]]:
    rngs = random.split(rng, env.steps)
    return lax.scan(step, state, rngs)


# %% Main #####################################################################
env = pb.env.Env()
init_key, traj_key = random.split(random.PRNGKey(0), (2, env.sims))

traj = vmap(partial(traj_fn, env, partial(step_fn, env)))  # vector trajs
init = jit(vmap(env.init)).lower(init_key).compile()  # vector inits
tic = time.time()

obs, state = init(init_key)
print(state.hp[0, 0])
toc = time.time()
print(f"init time: {toc - tic:.2f} seconds")

obs, state = init(traj_key)
print(state.hp[0, 0])
tic = time.time()
print(f"init time: {tic - toc:.2f} seconds")


state, (state_seq, action_seq) = traj(state, traj_key)
print(state.hp[0, 0])
toc = time.time()
print(f"traj time: {toc - tic:.2f} seconds")
