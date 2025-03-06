# env.py
#   parabellum env
# by: Noah Syrkis

# % Imports
import jax.numpy as jnp
from jax import random, Array, lax
from typing import Tuple
from parabellum.geo import geography_fn
from parabellum.types import Action, State, Obs, Scene
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
    health = jnp.ones(env.num_units)  # health of agents by type for starting
    pos = random.normal(keys[1], (scene.unit_types.size, 2)) + env.cfg.size / 2
    state = State(unit_position=pos, unit_health=health, unit_cooldown=jnp.zeros(env.num_units))  # state --
    return obs_fn(env, scene, state), state  # return observation and state of agents --


@eqx.filter_jit  # knn from env.cfg never changes, so we can jit it
def obs_fn(env, scene: Scene, state: State) -> Obs:  # return info about neighbors ---
    distances = jnp.linalg.norm(state.unit_position[:, None] - state.unit_position, axis=-1)  # all dist --
    dists, idxs = lax.approx_min_k(distances, k=env.cfg.knn)
    mask = dists < scene.unit_type_sight[scene.unit_types][..., None]  # mask for removing hidden -
    health = state.unit_health[idxs] * mask
    cooldown = state.unit_cooldown[idxs] * mask
    pos = state.unit_position[idxs] * mask[..., None]
    return Obs(unit_id=idxs, unit_pos=pos, unit_health=health, unit_cooldown=cooldown)


@eqx.filter_jit
def step_fn(rng, env: Env, scene: Scene, state: State, action: Action) -> State:  # update agents ---
    new_pos = state.unit_position + action.coord * (1 - action.kinds[..., None])
    bounds = ((new_pos < 0).any(axis=-1) | (new_pos >= env.cfg.size).any(axis=-1))[..., None]
    builds = (scene.terrain.building[*new_pos.astype(jnp.int32).T] > 0)[..., None]
    pos = jnp.where(bounds | builds, state.unit_position, new_pos)  # use old pos if new is not valid
    return State(unit_position=pos, unit_health=state.unit_health, unit_cooldown=state.unit_cooldown)  # return -


# @eqx.filter_jit
def scene_fn(cfg):
    aux = lambda key: jnp.array([x[key] for x in sorted(cfg.types, key=lambda x: x.name)])  # noqa
    attrs = ["health", "damage", "reload", "reach", "sight", "speed"]
    kwargs = {f"unit_type_{a}": aux(a) for a in attrs} | {"terrain": geography_fn(cfg.place)}
    num_allies, num_enemies = sum(cfg.counts.allies.values()), sum(cfg.counts.enemies.values())
    unit_teams = jnp.concat((jnp.zeros(num_allies), jnp.ones(num_enemies))).astype(jnp.int32)
    aux = lambda t: jnp.concat([jnp.zeros(x) + i for i, x in enumerate([x[1] for x in sorted(cfg.counts[t].items())])])  # noqa
    unit_types = jnp.concat((aux("allies"), aux("enemies"))).astype(jnp.int32)
    return Scene(unit_teams=unit_teams, unit_types=unit_types, **kwargs)  # type: ignore
