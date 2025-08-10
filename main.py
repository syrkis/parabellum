# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
from functools import partial
from typing import Tuple

from jax import jit, lax, random, vmap
from jaxtyping import Array

import parabellum as pb
import mlxp
from parabellum.types import Action, State


# %% Functions
def action_fn(env: pb.Env, rng: Array) -> Action:
    pos = random.uniform(rng, (env.types.size, 2), minval=-1, maxval=1) * env.reach[env.types][..., None]
    move = random.randint(rng, (env.types.size,), minval=0, maxval=2) == 1
    return Action(pos=pos, move=move)


def step_fn(env: pb.Env, state: State, rng: Array) -> Tuple[State, Tuple[State, Action]]:
    action = action_fn(env, rng)
    obs, state = env.step(rng, state, action)
    return state, (state, action)


def traj_fn(env, state, rng) -> Tuple[State, Tuple[State, Action]]:
    rngs = random.split(rng, env.cfg.steps)
    return lax.scan(partial(step_fn, env), state, rngs)


@mlxp.launch(config_path="./conf")
def main(ctx: mlxp.Context) -> None:
    env = pb.Env(ctx.config)
    init_key, traj_key = random.split(random.PRNGKey(0), (2, ctx.config.sims))
    obs, state = vmap(jit(env.init))(init_key)
    state, (seq, action) = vmap(jit(partial(traj_fn, env)))(state, init_key)


if __name__ == "__main__":
    main()
