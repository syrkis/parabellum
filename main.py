# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
from functools import partial
from typing import Tuple

from jax import jit, lax, random, vmap
from jaxtyping import Array

import parabellum as pb
from parabellum.types import Action, Config, State


# %% Functions
def action_fn(cfg: Config, rng: Array) -> Action:
    pos = random.uniform(rng, (cfg.length, 2), minval=-1, maxval=1) * cfg.rules.reach[cfg.types][..., None]
    move = random.randint(rng, (cfg.length,), minval=0, maxval=2) == 1
    return Action(pos=pos, move=move)


def step_fn(state: State, rng: Array) -> Tuple[State, Tuple[State, Action]]:
    action = action_fn(cfg, rng)
    obs, state = env.step(cfg, rng, state, action)
    return state, (state, action)


def traj_fn(state, rng) -> Tuple[State, Tuple[State, Action]]:
    rngs = random.split(rng, cfg.steps)
    return lax.scan(step_fn, state, rngs)


# %% Main
env, cfg = pb.env.Env(), Config()
init_key, traj_key = random.split(random.PRNGKey(0), (2, cfg.sims))

init = vmap(jit(partial(env.init, cfg)))
traj = vmap(jit(traj_fn))

obs, state = init(init_key)
state, (seq, action) = traj(state, init_key)

pb.utils.svg_fn(cfg, seq, action, "/Users/nobr/desk/s3/parabellum/sims.svg", debug=True)
# %% Anim
# for i in range(seq.pos.shape[0]):  # sims
# for j in range(seq.pos.shape[2]):  # units
# shots = [(kdx, coord) for kdx, coord in enumerate(action.pos[i, :, j]) if action.shoot[i, kdx, j]]
# print(shots)
# print(tree.map(jnp.shape, action))
# print(tree.map(jnp.shape, seq))
# pb.utils.svg_fn(cfg, seq.pos, action)
# pb.utils.gif_fn(cfg, seq)
