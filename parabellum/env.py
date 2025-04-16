# env.py
#   parabellum env
# by: Noah Syrkis

# % Imports
import jax.numpy as jnp
from jax import random, Array, lax, vmap
import jax.numpy.linalg as la
from typing import Tuple
from functools import partial

from parabellum.geo import geography_fn
from parabellum.types import Action, State, Obs, Scene
from parabellum import aid
import equinox as eqx


# %% Dataclass ################################################################
class Env:
    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self, rng: Array, scene: Scene) -> Tuple[Obs, State]:
        return init_fn(rng, self, scene)

    def step(self, rng: Array, scene: Scene, state: State, action: Action) -> Tuple[Obs, State]:
        return obs_fn(self, scene, state), step_fn(rng, self, scene, state, action)

    @property
    def num_units(self):
        return sum(self.cfg.counts.allies.values()) + sum(self.cfg.counts.enemies.values())

    @property
    def num_allies(self):
        return sum(self.cfg.counts.allies.values())

    @property
    def num_enemies(self):
        return sum(self.cfg.counts.enemies.values())


# %% Functions ################################################################
@eqx.filter_jit
def init_fn(rng: Array, env: Env, scene: Scene) -> Tuple[Obs, State]:  # initialize -----
    keys = random.split(rng)
    health = jnp.ones(env.num_units) * scene.unit_type_health[scene.unit_types]
    pos = random.normal(keys[1], (scene.unit_types.size, 2)) * 2 + env.cfg.size / 2
    state = State(unit_position=pos, unit_health=health, unit_cooldown=jnp.zeros(env.num_units))  # state --
    return obs_fn(env, scene, state), state  # return observation and state of agents --


@eqx.filter_jit  # knn from env.cfg never changes, so we can jit it
def obs_fn(env, scene: Scene, state: State) -> Obs:  # return info about neighbors ---
    distances = la.norm(state.unit_position[:, None] - state.unit_position, axis=-1)  # all dist --
    dists, idxs = lax.approx_min_k(distances, k=env.cfg.knn)
    mask = sight_fn(scene, state, dists, idxs)
    health = state.unit_health[idxs] * mask
    cooldown = state.unit_cooldown[idxs] * mask
    self_pos = state.unit_position[idxs[0]]
    unit_pos = ((state.unit_position[:, None, ...] - state.unit_position[idxs]) * mask[..., None]).at[0].set(self_pos)
    return Obs(unit_id=idxs, unit_pos=unit_pos, unit_health=health, unit_cooldown=cooldown)


@eqx.filter_jit
def step_fn(rng, env: Env, scene: Scene, state: State, action: Action) -> State:  # update agents ---
    newpos = state.unit_position + action.coord * (1 - action.kinds[..., None])
    bounds = ((newpos < 0).any(axis=-1) | (newpos >= env.cfg.size).any(axis=-1))[..., None]
    builds = (scene.terrain.building[*newpos.astype(jnp.int32).T] > 0)[..., None]
    newpos = jnp.where(bounds | builds, state.unit_position, newpos)  # use old pos if new is not valid
    new_hp = blast_fn(rng, env, scene, state, action)
    return State(unit_position=newpos, unit_health=new_hp, unit_cooldown=state.unit_cooldown)  # return -


def blast_fn(rng, env: Env, scene: Scene, state: State, action: Action):  # update agents ---
    dist = la.norm(state.unit_position[None, ...] - (state.unit_position + action.coord)[:, None, ...], axis=-1)
    hits = dist <= scene.unit_type_reach[scene.unit_types][None, ...] * action.kinds[..., None]  # mask non attack act
    damage = (hits * scene.unit_type_damage[scene.unit_types][None, ...]).sum(axis=-1)
    return state.unit_health - damage


# @eqx.filter_jit
def scene_fn(cfg):  # init's a scene
    aux = lambda key: jnp.array([x[key] for x in sorted(cfg.types, key=lambda x: x.name)])  # noqa
    attrs = ["health", "damage", "reload", "reach", "sight", "speed"]
    kwargs = {f"unit_type_{a}": aux(a) for a in attrs} | {"terrain": geography_fn(cfg.place, cfg.size)}
    num_allies, num_enemies = sum(cfg.counts.allies.values()), sum(cfg.counts.enemies.values())
    unit_teams = jnp.concat((jnp.zeros(num_allies), jnp.ones(num_enemies))).astype(jnp.int32)
    aux = lambda t: jnp.concat([jnp.zeros(x) + i for i, x in enumerate([x[1] for x in sorted(cfg.counts[t].items())])])  # noqa
    unit_types = jnp.concat((aux("allies"), aux("enemies"))).astype(jnp.int32)
    mask = aid.obstacle_mask_fn(max([x["sight"] for x in cfg.types]))
    return Scene(unit_teams=unit_teams, unit_types=unit_types, mask=mask, **kwargs)  # type: ignore


@eqx.filter_jit
def sight_fn(scene, state, dists, idxs):
    mask = dists < scene.unit_type_sight[scene.unit_types][..., None]  # mask for removing hidden
    mask = mask | obstacle_fn(scene, state.unit_position[idxs].astype(jnp.int8))
    return mask


@partial(vmap, in_axes=(None, 0))  # 5 x 2 # not the best name for a fn
def obstacle_fn(scene, pos):
    return slice_fn(scene, pos[0], pos)


@partial(vmap, in_axes=(None, None, 0))
def slice_fn(scene, source, target):  # returns a 10 x 10 view with unit at top left corner, and terrain downwards
    delta = ((source - target) >= 0) * 2 - 1
    coord = jnp.sort(jnp.stack((source, source + delta * 10)), axis=0)[0]
    slice = lax.dynamic_slice(scene.terrain.building, coord, (scene.mask.shape[-1], scene.mask.shape[-1]))
    slice = lax.cond(delta[0] == 1, lambda: jnp.flip(slice), lambda: slice)
    slice = lax.cond(delta[1] == 1, lambda: jnp.flip(slice, axis=1), lambda: slice)
    return (scene.mask[*jnp.abs(source - target)] * slice).sum() == 0
