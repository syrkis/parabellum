# types.py
#   parabellum types
# by: Noah Syrkis

# imports
from chex import dataclass
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float32, Int
from dataclasses import field


# dataclasses
@dataclass
class State:
    coord: Array
    hp: Array
    # target: Array


@dataclass
class Obs:
    # idxs: Array
    hp: Array
    type: Array
    team: Array
    dist: Array
    coord: Array
    reach: Array
    sight: Array
    speed: Array

    @property
    def ally(self):
        return self.team == self.team[0]

    @property
    def enemy(self):
        return self.team != self.team[0]


@dataclass
class Action:
    coord: Float32[Array, "... 2"]  # noqa
    types: Int[Array, "..."]  # 0 = invalid, 1 = move, 2 = shoot

    @property
    def invalid(self):
        return self.types == 0

    @property
    def move(self):
        return self.types == 1

    @property
    def shoot(self):
        return self.types == 2


@dataclass
class Terrain:
    building: Array
    water: Array
    forest: Array
    basemap: Array


@dataclass
class Scene:
    mask: Array
    terrain: Terrain

    unit_types: Array
    unit_teams: Array

    unit_type_health: Array
    unit_type_damage: Array
    unit_type_reload: Array

    unit_type_reach: Array
    unit_type_sight: Array
    unit_type_blast: Array
    unit_type_speed: Array
