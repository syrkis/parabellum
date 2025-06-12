# env.py
#   parabellum env
# by: Noah Syrkis

# % Imports
from functools import partial
from typing import Tuple

import jax.numpy as jnp
import jax.numpy.linalg as la
from jax import Array, debug, lax, random, vmap

from parabellum.types import Action, Config, Obs, State


# %% Dataclass ################################################################
class Env:
    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self, rng: Array, cfg: Config) -> Tuple[Obs, State]:
        return init_fn(rng, self, cfg)

    def step(self, rng: Array, cfg: Config, state: State, action: Action) -> Tuple[Obs, State]:
        return obs_fn(self, cfg, state), step_fn(rng, self, cfg, state, action)


# @eqx.filter_jit
def knn(poss, k, n):
    def aux(inputs):
        batch_pos, batch_norms = inputs
        dots = jnp.dot(batch_pos, poss.T)
        dist = jnp.maximum(batch_norms[:, None] + norms[None, :] - 2 * dots, 0)
        return lax.approx_min_k(dist, k=k)

    norms = jnp.sum(poss**2, axis=1)
    dist, idxs = lax.map(aux, (poss.reshape((n, n, 2)), norms.reshape(n, n)))
    return dist.reshape((-1, k)) ** 0.5, idxs.reshape((-1, k))


# %% Functions
# @eqx.filter_jit
def init_fn(rng: Array, env: Env, cfg: Config) -> Tuple[Obs, State]:
    prob = jnp.ones((cfg.size, cfg.size)).at[cfg.map].set(0).flatten()  # Set
    flat = random.choice(rng, jnp.arange(prob.size), shape=(cfg.length,), p=prob / prob.sum(), replace=True)
    idxs = (flat // len(cfg.map), flat % len(cfg.map))
    pos = jnp.float32(jnp.column_stack(idxs))
    # pos = jnp.ones_like(pos)  # * jnp.arange(scene.terrain.building.shape[1])
    state = State(pos=pos, hp=cfg.rules.hp[cfg.types])
    return obs_fn(env, cfg, state), state


# @eqx.filter_jit  # knn from env.cfg never changes, so we can jit it
def obs_fn(env, cfg: Config, state: State) -> Obs:  # return info about neighbors ---
    dist, idxs = knn(state.pos, k=env.cfg.knn, n=int(cfg.length**0.5))
    mask = (dist < cfg.rules.sight[cfg.types[idxs][:, 0]][..., None]) | (state.hp[idxs] > 0)

    type = cfg.types[idxs] * mask
    team = cfg.teams[idxs] * mask
    hp = state.hp[idxs] * mask
    reach = cfg.rules.reach[type] * mask
    sight = cfg.rules.sight[type] * mask
    speed = cfg.rules.speed[type] * mask

    pos = unit_pos_fn((state.pos[idxs] - state.pos[:, None, ...]), state.pos) * mask[..., None]
    obs = Obs(pos=pos, hp=hp, type=type, dist=dist * mask, team=team, reach=reach, sight=sight, speed=speed)
    # debug.breakpoint()
    return obs


@partial(vmap, in_axes=(0, 0))
def unit_pos_fn(unit_pos, self_pos):
    return unit_pos.at[0].set(self_pos)


# @eqx.filter_jit
def step_fn(rng: Array, env: Env, cfg: Config, state: State, action: Action) -> State:
    args = rng, env, cfg, state, action
    return State(pos=move_fn(*args), hp=blast_fn(*args))  # type: ignore


def move_fn(rng: Array, env: Env, cfg: Config, state: State, action: Action):
    speed = cfg.rules.speed[cfg.types][..., None]  # max speed of a unit (step size, really)
    pos = state.pos + action.pos.clip(-speed, speed) * action.move[..., None]  # new poss
    bound = ((pos < 0).any(axis=-1) | (pos >= env.cfg.size).any(axis=-1))[..., None]  # masking outside map
    stuff = (cfg.map[*pos.astype(jnp.int32).T] > 0)[..., None]  # type: ignore  # stuff in the way
    return jnp.where(bound | stuff, state.pos, pos)  # compute new position


# def blast_fn(env: Env, cfg: Config, state: State, action: Action):
# damage = cfg.rules.damage[cfg.types] * action.shoot
# pos = action.pos + state.pos
# debug.breakpoint()
# return state.hp


# def push_fn(poss):
#     """Push units away from each other if they're too close."""
#     return poss
#     distances = la.norm(poss[:, None] - poss, axis=-1)
#     too_close = (distances > 0) & (distances < 2.0)  # Units closer than 2 units

#     # Calculate unit displacement vectors
#     disp_vectors = poss[:, None] - poss

#     # Normalize displacement vectors (avoiding division by zero)
#     norms = distances[..., None]
#     safe_norms = jnp.where(norms > 0, norms, 1.0)
#     normalized_disp = disp_vectors / safe_norms

#     # Calculate repulsion forces (stronger when closer)
#     repulsion_strength = jnp.where(too_close, 1.0 / jnp.maximum(distances, 0.1), 0.0)
#     repulsion = normalized_disp * repulsion_strength[..., None]

#     # Sum all repulsion forces for each unit
#     total_repulsion = jnp.sum(repulsion, axis=1)

#     # Apply the repulsion with a scaling factor
#     return poss + total_repulsion * 0.5


def blast_fn(rng, env: Env, cfg: Config, state: State, action: Action) -> Array:  # update agents ---
    dist = la.norm(state.pos[None, ...] - (state.pos + action.pos)[:, None, ...], axis=-1)  # todo make efficient
    hits = dist <= cfg.rules.reach[cfg.types][None, ...] * action.shoot[..., None]  # mask non attack act
    damage = (hits * cfg.rules.damage[cfg.types][None, ...]).sum(axis=-1)
    return state.hp - damage


# @eqx.filter_jit
# def scene_fn(cfg: Config) -> Scene:  # init's a scene
# unit_teams = jnp.concat((jnp.zeros(len(cfg.blu)), jnp.ones(len(cfg.red))))
# unit_types = jnp.concat((cfg.blu.types, cfg.red.types))
# terrain = geography_fn(cfg.place, cfg.size)
# return Scene(unit_teams=unit_teams, unit_types=unit_types, terrain=terrain, cfg=cfg)


# @eqx.filter_jit
def sight_fn(cfg: Config, state: State, dists, idxs):
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
