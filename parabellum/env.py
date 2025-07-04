# %% env.py
#   parabellum env
# by: Noah Syrkis

# Imports
from functools import partial
from typing import Tuple

import jax.numpy as jnp
import jaxkd as jk
from jax import random
from jaxtyping import Array

from parabellum.types import Action, Config, Obs, State


# %% Dataclass
class Env:
    def init(self, cfg: Config, rng: Array) -> Tuple[Obs, State]:
        state = init_fn(cfg, rng)  # without jit this takes forever
        return obs_fn(cfg, state), state

    def step(self, cfg: Config, rng: Array, state: State, action: Action) -> Tuple[Obs, State]:
        state = step_fn(cfg, rng, state, action)
        return obs_fn(cfg, state), state


# %% Functions
def init_fn(cfg: Config, rng: Array) -> State:
    prob = jnp.ones((cfg.size, cfg.size)).at[cfg.map].set(0).flatten()  # Set
    flat = random.choice(rng, jnp.arange(prob.size), shape=(cfg.length,), p=prob, replace=True)
    idxs = (flat // len(cfg.map), flat % len(cfg.map))
    pos = jnp.float32(jnp.column_stack(idxs))
    return State(pos=pos, hp=cfg.hp[cfg.types])


def obs_fn(cfg: Config, state: State) -> Obs:  # return info about neighbors ---
    idxs, dist = jk.extras.query_neighbors_pairwise(state.pos, state.pos, k=cfg.knn)
    mask = dist < cfg.sight[cfg.types[idxs][:, 0]][..., None]  # | (state.hp[idxs] > 0)
    pos = (state.pos[idxs] - state.pos[:, None, ...]).at[:, 0, :].set(state.pos) * mask[..., None]
    args = state.hp, cfg.types, cfg.teams, cfg.reach, cfg.sight, cfg.speed
    hp, type, team, reach, sight, speed = map(lambda x: x[idxs] * mask, args)
    return Obs(pos=pos, dist=dist, hp=hp, type=type, team=team, reach=reach, sight=sight, speed=speed, mask=mask)


def step_fn(cfg: Config, rng: Array, state: State, action: Action) -> State:
    idx, norm = jk.extras.query_neighbors_pairwise(state.pos + action.pos, state.pos, k=2)
    args = rng, cfg, state, action, idx, norm
    return State(pos=partial(push_fn, cfg, rng, idx, norm)(move_fn(*args)), hp=blast_fn(*args))  # type: ignore


def move_fn(rng: Array, cfg: Config, state: State, action: Action, idx: Array, norm: Array) -> Array:
    speed = cfg.speed[cfg.types][..., None]  # max speed of a unit (step size, really)
    pos = state.pos + action.pos.clip(-speed, speed) * action.move[..., None]  # new poss
    mask = ((pos < 0).any(axis=-1) | ((pos >= cfg.size).any(axis=-1)) | (cfg.map[*jnp.int32(pos).T] > 0))[..., None]
    return jnp.where(mask, state.pos, pos)  # compute new position


def blast_fn(rng: Array, cfg: Config, state: State, action: Action, idx: Array, norm: Array) -> Array:
    dam = (cfg.dam[cfg.types] * action.cast)[..., None] * jnp.ones_like(idx)
    return state.hp - jnp.zeros(cfg.length, dtype=jnp.int32).at[idx.flatten()].add(dam.flatten())


def push_fn(cfg: Config, rng: Array, idx: Array, norm: Array, pos: Array) -> Array:
    return pos + random.normal(rng, pos.shape) * 0.1
    # params need to be tweaked, and matched with unit size
    pos_diff = pos[:, None, :] - pos[idx]  # direction away from neighbors
    mask = (norm < cfg.r[cfg.types][..., None]) & (norm > 0)
    pos = pos + jnp.where(mask[..., None], pos_diff * cfg.force / (norm[..., None] + 1e-6), 0.0).sum(axis=1)
    return pos + random.normal(rng, pos.shape) * 0.1
