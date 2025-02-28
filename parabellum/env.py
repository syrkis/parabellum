# env.py
#   parabellum env
# by: Noah Syrkis

import jax.numpy as jnp
from jax import random, Array
from chex import dataclass
from typing import Tuple
from dataclasses import field
import equinox as eqx
from parabellum.geo import geography_fn
from parabellum.aid import Terrain


# %% Dataclasses ################################################################
@dataclass
class Obs:
    dist: Array


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
    num_agents: int = 8
    unit_types: Array = field(default_factory=lambda: jnp.zeros(8).astype(jnp.int8))
    unit_team: Array = field(default_factory=lambda: jnp.array([0, 0, 0, 0, 1, 1, 1, 1]))
    line_of_sight: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_radiuses: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_health: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_attacks: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_attack_ranges: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_sight_ranges: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_velocities: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_weapon_cooldowns: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))


@dataclass
class Scene:
    terrain: Terrain


@dataclass
class Env:
    cfg: Conf
    scene: Scene

    def __init__(self, cfg):
        self.cfg = cfg
        self.scene = Scene(terrain=geography_fn(self.cfg.place, buffer=500))

    def reset(self, rng: Array) -> Tuple[Obs, State]:
        return init_fn(rng, self.cfg, self)

    def step(self, rng, state, action) -> Tuple[Obs, State]:
        return obs_fn(self.cfg, state), step_fn(self, state, action)


@dataclass
class Action:
    health: Array | None  # attack/heal
    moving: Array  # move agents


# %% Functions ################################################################
# @eqx.filter_jit
def init_fn(rng: Array, cfg: Conf, env: Env) -> Tuple[Obs, State]:  # initialize -----
    keys, num_agents = random.split(rng), env.cfg.num_allies + env.cfg.num_enemies  # meta ----
    types = random.choice(keys[0], jnp.arange(env.cfg.unit_type_attacks.size), (num_agents,))  #
    health = jnp.take(env.cfg.unit_type_health, types)  # health of agents by type for starting
    pos = random.normal(keys[1], (num_agents, 2)) + cfg.size / 2
    state = State(unit_position=pos, unit_health=health, unit_cooldown=jnp.zeros(num_agents))  # state --
    return obs_fn(cfg, state), state  # return observation and state of agents --


# @eqx.filter_jit
def obs_fn(cfg: Conf, state: State) -> Obs:  # return info about neighbors ---
    distances = jnp.linalg.norm(state.unit_position[:, None] - state.unit_position, axis=-1)  # all dist --
    mask = distances < cfg.unit_type_sight_ranges[cfg.unit_types][..., None]  # mask for removing hidden -
    return Obs(dist=jnp.where(mask, distances, jnp.inf)[0])


#
@eqx.filter_jit
def step_fn(env: Env, state: State, action: Action) -> State:  # update agents ---
    # print(state.unit_position.shape, action.moving)
    # exit()
    new_pos = state.unit_position + action.moving
    mask = ((new_pos < 0).any(axis=-1) | (new_pos >= env.cfg.size).any(axis=-1))[..., None]
    pos = jnp.where(mask, state.unit_position, new_pos)

    # hp = state.unit_health + action.health * env.cfg.unit_type_attacks[env.cfg.unit_types][..., None]  #
    return State(unit_position=pos, unit_health=state.unit_health, unit_cooldown=state.unit_cooldown)  # return -
