# env.py
#   parabellum env
# by: Noah Syrkis

# % Imports
from functools import partial
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.numpy.linalg as la
from jax import lax, random, vmap, jit, tree
from jaxtyping import Array
import jaxkd as jk

# from chex import dataclass
from parabellum.types import Action, Obs, State, Config


# %% Dataclass ################################################################
# @dataclass
class Env:
    def init(self, cfg: Config, rng: Array):  # -> Tuple[Obs, State]:
        state = init_fn(cfg, rng)  # without jit this takes forever
        return obs_fn(cfg, state), state

    def step(self, cfg: Config, rng: Array, state: State, action: Action) -> Tuple[Obs, State]:
        return obs_fn(cfg, state), step_fn(cfg, rng, state, action)


def init_fn(cfg: Config, rng: Array) -> State:
    prob = jnp.ones((cfg.size, cfg.size)).at[cfg.map].set(0).flatten()  # Set
    flat = random.choice(rng, jnp.arange(prob.size), shape=(cfg.length,), p=prob / prob.sum(), replace=True)
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
    args = rng, cfg, state, action
    return State(pos=move_fn(*args), hp=state.hp)  # blast_fn(*args))  # type: ignore


def move_fn(rng: Array, cfg: Config, state: State, action: Action):
    speed = cfg.rules.speed[cfg.types][..., None]  # max speed of a unit (step size, really)
    pos = state.pos + action.pos.clip(-speed, speed) * action.move[..., None]  # new poss
    bound = ((pos < 0).any(axis=-1) | (pos >= cfg.size).any(axis=-1))[..., None]  # masking outside map
    stuff = (cfg.map[*pos.astype(jnp.int32).T] > 0)[..., None]  # type: ignore  # stuff in the way
    return jnp.where(bound | stuff, state.pos, pos)  # compute new position


def blast_fn(rng, cfg: Config, state: State, action: Action) -> Array:  # update agents ---
    dist = la.norm(state.pos[None, ...] - (state.pos + action.pos)[:, None, ...], axis=-1)  # todo make efficient
    hits = dist <= cfg.rules.reach[cfg.types][None, ...] * action.shoot[..., None]  # mask non attack act
    damage = (hits * cfg.rules.damage[cfg.types][None, ...]).sum(axis=-1)
    return state.hp - damage


# @eqx.filter_jit
def sight_fn(cfg: Config, state: State, dists, idxs):
    mask = dists < cfg.rules.sight[cfg.types][..., None]  # mask for removing hidden
    mask = mask  # | obstacle_fn(cfg, state.pos[idxs].astype(jnp.int8))
    return mask


# def blast_fn(cfg: Config, cfg: Config, state: State, action: Action):
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


# @eqx.filter_jit
# def scene_fn(cfg: cfg) -> Scene:  # init's a scene
# unit_teams = jnp.concat((jnp.zeros(len(cfg.blu)), jnp.ones(len(cfg.red))))
# unit_types = jnp.concat((cfg.blu.types, cfg.red.types))
# terrain = geography_fn(cfg.place, cfg.size)
# return Scene(unit_teams=unit_teams, unit_types=unit_types, terrain=terrain, cfg=cfg)


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
