# env.py
#   parabellum env
# by: Noah Syrkis

# % Imports
from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import lax, random, vmap, jit, tree, debug
from jaxtyping import Array
import jaxkd as jk

# from chex import dataclass
from parabellum.types import Action, Obs, State, Config


class Env:
    def init(self, cfg: Config, rng: Array):  # -> Tuple[Obs, State]:
        state = init_fn(cfg, rng)  # without jit this takes forever
        return obs_fn(cfg, state), state

    def step(self, cfg: Config, rng: Array, state: State, action: Action) -> Tuple[Obs, State]:
        return obs_fn(cfg, state), step_fn(cfg, rng, state, action)


def init_fn(cfg: Config, rng: Array) -> State:
    prob = jnp.ones((cfg.size, cfg.size)).at[cfg.map].set(0).flatten()  # Set
    flat = random.choice(rng, jnp.arange(prob.size), shape=(cfg.length,), p=prob, replace=True)
    idxs = (flat // len(cfg.map), flat % len(cfg.map))
    pos = jnp.float32(jnp.column_stack(idxs))
    state = State(pos=pos, hp=cfg.rules.hp[cfg.types])
    return state


# @eqx.filter_jit
def obs_fn(cfg: Config, state: State):  # return info about neighbors ---
    idxs, dist = jk.extras.query_neighbors_pairwise(state.pos, state.pos, k=cfg.knn)
    mask = dist < cfg.rules.sight[cfg.types[idxs][:, 0]][..., None]  # | (state.hp[idxs] > 0)
    pos = (state.pos[idxs] - state.pos[:, None, ...]).at[:, 0, :].set(state.pos) * mask[..., None]
    hp, type, team, reach, sight, speed = map(
        lambda x: x[idxs] * mask, (state.hp, cfg.types, cfg.teams, cfg.rules.reach, cfg.rules.sight, cfg.rules.speed)
    )  # we are not masking dist. So we can see that SOMETHING is far-ish away
    return Obs(pos=pos, hp=hp, type=type, dist=dist, team=team, reach=reach, sight=sight, speed=speed)


# @eqx.filter_jit
def step_fn(cfg: Config, rng: Array, state: State, action: Action) -> State:
    idx, norm = jk.extras.query_neighbors_pairwise(state.pos + action.pos, state.pos, k=10)
    args = rng, cfg, state, action, idx, norm
    return State(pos=partial(push_fn, cfg, rng, idx, norm)(move_fn(*args)), hp=blast_fn(*args))  # type: ignore


def move_fn(rng: Array, cfg: Config, state: State, action: Action, idx: Array, norm: Array):
    speed = cfg.rules.speed[cfg.types][..., None]  # max speed of a unit (step size, really)
    pos = state.pos + action.pos.clip(-speed, speed) * action.move[..., None]  # new poss
    mask = ((pos < 0).any(axis=-1) | (pos >= cfg.size).any(axis=-1))[..., None]  # masking outside map
    mask = mask | (cfg.map[*pos.astype(jnp.int32).T] > 0)[..., None]  # type: ignore  # stuff in the way
    return jnp.where(mask, pos, state.pos)  # compute new position


def blast_fn(rng: Array, cfg: Config, state: State, action: Action, idx: Array, norm: Array) -> Array:
    dam = (cfg.rules.damage[cfg.types] * action.shoot)[..., None] * jnp.ones_like(idx)
    return state.hp - jnp.zeros(cfg.length, dtype=jnp.int32).at[idx.flatten()].add(dam.flatten())


def push_fn(cfg: Config, rng: Array, idx: Array, norm: Array, pos: Array) -> Array:
    radii = 2.0  # hardcoded minimum distance
    force = 0.5  # hardcoded repulsion force multiplier
    pos_diff = pos[:, None, :] - pos[idx]  # direction away from neighbors
    mask = (norm < radii) & (norm > 0)
    return pos + jnp.where(mask[..., None], pos_diff * force / (norm[..., None] + 1e-6), 0.0).sum(axis=1)


def sight_fn(cfg: Config, state: State, dists: Array, idxs: Array):
    mask = dists < cfg.rules.sight[cfg.types][..., None]  # mask for removing hidden
    mask = mask  # | obstacle_fn(cfg, state.pos[idxs].astype(jnp.int8))
    return mask


# @partial(vmap, in_axes=(None, 0))  # 5 x 2 # not the best name for a fn
# def obstacle_fn(cfg: Config, pos):
# return slice_fn(cfg, pos[0], pos)


# @partial(vmap, in_axes=(None, None, 0))
# def slice_fn(cfg: Config, source, target):  # returns a 10 x 10 view with unit at top left corner, and terrain downwards
# delta = ((source - target) >= 0) * 2 - 1
# pos = jnp.sort(jnp.stack((source, source + delta * 10)), axis=0)[0]
# slice = lax.dynamic_slice(scene.terrain.building, pos, (scene.mask.shape[-1], scene.mask.shape[-1]))
# slice = lax.cond(delta[0] == 1, lambda: jnp.flip(slice), lambda: slice)
# slice = lax.cond(delta[1] == 1, lambda: jnp.flip(slice, axis=1), lambda: slice)
# return (scene.mask[*jnp.abs(source - target)] * slice).sum() == 0  # type: ignore
