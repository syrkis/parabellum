# env.py
#   parabellum env
# by: Noah Syrkis

import jax.numpy as jnp
from jax import random, Array
from chex import dataclass
from typing import Tuple
from dataclasses import field

# import equinox as eqx
from parabellum.geo import geography_fn
from parabellum.aid import Terrain
from parabellum import tim


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
    distance_map: Array
    movement_randomness = 5.0
    units_push_back_firmness = 0.02
    # FILL THIS
    unit_target_position_id: Array  # the idx of the distance_map (used by the following map atomics)
    unit_starting_sectors: Array  # the inital spawing aera for each unit
    # FILL THIS
    unit_team: Array = field(default_factory=lambda: jnp.array([0, 0, 0, 0, 1, 1, 1, 1]))
    unit_type: Array = field(default_factory=lambda: jnp.array([0, 0, 0, 0, 1, 1, 1, 1]))
    unit_type_radiuses: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_health: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_attacks: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_attack_ranges: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_sight_ranges: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_velocities: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    unit_type_weapon_cooldowns: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))
    line_of_sight: Array = field(default_factory=lambda: jnp.array([1, 1, 1]))


@dataclass
class Env:
    cfg: Conf
    scene: Scene

    def __init__(self, cfg):
        self.cfg = cfg
        terrain = geography_fn(self.cfg.place, buffer=500)
        line_of_sight = compute_line_of_sight_discretization(self.cfg.unit_type_sight_ranges)
        distance_map = tim.compute_distance_map(cfg, terrain, jnp.array([cfg.size / 2, cfg.size / 2]).astype(jnp.int32))
        unit_starting_sectors = [
            ([i for i in range(cfg.num_allies)], [0, 0, 1.0, 1.0]),
            ([i + cfg.num_allies for i in range(cfg.num_enemies)], [0, 0, 1.0, 1.0]),
        ]
        valid_sectors = tim.valid_sectors_fn(
            jnp.array(unit_starting_sectors), jnp.where(terrain.building + terrain.water > 0, 1, 0)
        )
        allies_target = [1] * cfg.num_allies
        enemies_target = [0] * cfg.num_enemies
        unit_target_position_id = jnp.array(allies_target + enemies_target, dtype=jnp.uint32)
        self.scene = Scene(
            terrain=terrain,
            line_of_sight=line_of_sight,
            distance_map=distance_map,
            unit_starting_sectors=valid_sectors,
            unit_target_position_id=unit_target_position_id,
        )

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


# @eqx.filter_jit
def step_fn(env: Env, state: State, action: Action) -> State:  # update agents ---
    new_pos = state.unit_position + action.moving
    mask = ((new_pos < 0).any(axis=-1) | (new_pos >= env.cfg.size).any(axis=-1))[..., None]
    pos = jnp.where(mask, state.unit_position, new_pos)
    return State(unit_position=pos, unit_health=state.unit_health, unit_cooldown=state.unit_cooldown)  # return -


def compute_line_of_sight_discretization(unit_type_sight_ranges):
    resolution = jnp.array(jnp.max(unit_type_sight_ranges), dtype=jnp.int32) * 2
    return jnp.tile(jnp.linspace(0, 1, resolution.item()), (2, 1))  # the constant line of sight discretization
