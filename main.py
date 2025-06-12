# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
from jax import random, lax, vmap
import parabellum as pb
from jaxtyping import Array
from functools import partial
from typing import Tuple
from parabellum.types import Action, State, Config


# %% Functions ###############################################################
def action_fn(cfg: Config, rng: Array) -> Action:
    pos = random.normal(rng, (cfg.length, 2)) * cfg.rules.reach[cfg.types][..., None]
    kind = random.randint(rng, (cfg.length,), minval=0, maxval=3)
    return Action(pos=pos, kind=kind)


def step_fn(env: pb.env.Env, cfg: Config, state: State, rng: Array) -> Tuple[State, Tuple[State, Action]]:
    action = action_fn(cfg, rng)
    obs, state = env.step(rng, cfg, state, action)
    return state, (state, action)


# @partial(vmap, in_axes=(None, None, 0, 0))
def traj_fn(env, cfg, state, rng) -> Tuple[State, Tuple[State, Action]]:
    step = partial(step_fn, env, cfg)
    rngs = random.split(rng, cfg.steps)
    state, (seq, action) = lax.scan(step, state, rngs)
    return state, (seq, action)


# %% Main #####################################################################
cfg = pb.types.Config()
env = pb.env.Env(cfg=cfg)
rng, key = random.split(random.PRNGKey(0))
init_key, traj_key = random.split(rng, (2, cfg.sims))
obs, state = vmap(partial(env.reset, cfg))(init_key)
state, (seq, action) = vmap(partial(traj_fn, env, cfg))(state, traj_key)
# pb.utils.svg_fn(cfg, seq, action, fps=10)
