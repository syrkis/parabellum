# env.py
#   parabellum env
# by: Noah Syrkis

import jax.numpy as jnp
from jax import random, Array
from chex import dataclass
from typing import Tuple
from dataclasses import field
from parabellum.geo import geography_fn
from parabellum.aid import Terrain


# %% Dataclasses ################################################################
@dataclass
class State:
    unit_position: Array
    unit_health: Array
    unit_cooldown: Array


@dataclass
class Conf:  # TODO: add water, trees, etc in terrain
    place: str = "Copenhagen, Denmark"
    size: int = 100
    knn: int = 5
    num_allies: int = 4
    num_enemies: int = 4
    unit_type_health: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_damage: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_reload: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_reach: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_sight: Array = field(default_factory=lambda: jnp.array([10, 1, 1]))
    unit_type_speed: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))


@dataclass
class Scene:
    terrain: Terrain
    unit_types: Array
    unit_teams: Array
    num_agents: int


@dataclass
class Obs:
    dist: Array


@dataclass
class Env:
    cfg: Conf
    scene: Scene

    def __init__(self, cfg):
        self.cfg = cfg
        terrain = geography_fn(self.cfg.place, buffer=cfg.size)
        num_agents = cfg.num_allies + cfg.num_enemies
        unit_types = jnp.zeros(num_agents).astype(jnp.int32)
        unit_teams = jnp.concat((jnp.zeros(cfg.num_allies), jnp.ones(cfg.num_enemies))).astype(jnp.int32)
        self.scene = Scene(terrain=terrain, unit_types=unit_types, unit_teams=unit_teams, num_agents=num_agents)

    def reset(self, rng: Array) -> Tuple[Obs, State]:
        return init_fn(rng, self)

    def step(self, rng, state, action) -> Tuple[Obs, State]:
        return obs_fn(self, state), step_fn(self, state, action)


@dataclass
class Action:
    health: Array | None  # attack/heal
    moving: Array  # move agents


# %% Functions ################################################################
# @eqx.filter_jit
def init_fn(rng: Array, env: Env) -> Tuple[Obs, State]:  # initialize -----
    keys, num_agents = random.split(rng), env.cfg.num_allies + env.cfg.num_enemies  # meta ----
    health = jnp.take(env.cfg.unit_type_health, env.scene.unit_types)  # health of agents by type for starting
    pos = random.normal(keys[1], (num_agents, 2)) + env.cfg.size / 2
    state = State(unit_position=pos, unit_health=health, unit_cooldown=jnp.zeros(num_agents))  # state --
    return obs_fn(env, state), state  # return observation and state of agents --


# @eqx.filter_jit
def obs_fn(env, state: State) -> Obs:  # return info about neighbors ---
    distances = jnp.linalg.norm(state.unit_position[:, None] - state.unit_position, axis=-1)  # all dist --
    mask = distances < env.cfg.unit_type_sight[env.scene.unit_types][..., None]  # mask for removing hidden -
    return Obs(dist=jnp.where(mask, distances, jnp.inf)[0])


# @eqx.filter_jit
def step_fn(env: Env, state: State, action: Action) -> State:  # update agents ---
    new_pos = state.unit_position + action.moving
    bounds = ((new_pos < 0).any(axis=-1) | (new_pos >= env.cfg.size).any(axis=-1))[..., None]
    builds = (env.scene.terrain.building[*new_pos.astype(jnp.int32).T] > 0)[..., None]
    pos = jnp.where(bounds | builds, state.unit_position, new_pos)  # use old pos if new is not valid
    return State(unit_position=pos, unit_health=state.unit_health, unit_cooldown=state.unit_cooldown)  # return -


def compute_line_of_sight_discretization(unit_type_sight_ranges):
    resolution = jnp.array(jnp.max(unit_type_sight_ranges), dtype=jnp.int32) * 2
    return jnp.tile(jnp.linspace(0, 1, resolution.item()), (2, 1))  # the constant line of sight discretization
