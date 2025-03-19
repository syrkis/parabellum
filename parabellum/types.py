# types.py
#   parabellum types
# by: Noah Syrkis

# imports
from chex import dataclass
from jaxtyping import Array, Bool, Float16


# dataclasses
@dataclass
class State:
    unit_position: Array
    unit_health: Array
    unit_cooldown: Array


@dataclass
class Obs:
    unit_id: Array
    unit_pos: Array
    unit_health: Array
    unit_cooldown: Array


@dataclass
class Action:
    coord: Float16[Array, "... 2"]  # noqa
    kinds: Bool[Array, "..."]


@dataclass
class Terrain:
    building: Array
    water: Array
    forest: Array
    basemap: Array


@dataclass
class Scene:
    terrain: Terrain
    mask: Array

    unit_types: Array
    unit_teams: Array

    unit_type_health: Array
    unit_type_damage: Array
    unit_type_reload: Array

    unit_type_reach: Array
    unit_type_sight: Array
    unit_type_speed: Array
