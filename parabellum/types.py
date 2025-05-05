# types.py
#   parabellum types
# by: Noah Syrkis

# imports
from chex import dataclass
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float16
from dataclasses import field


# dataclasses
@dataclass
class State:
    coords: Array
    health: Array
    # target: Array


@dataclass
class Obs:
    idxs: Array
    type: Array
    coords: Array
    health: Array
    # target: Array


@dataclass
class Action:
    # coord: Float16[Array, "... 2"] = jnp.array([[0, 0]])  # noqa
    # kinds: Bool[Array, "..."] = jnp.array([0])
    coord: Float16[Array, "... 2"] = field(default_factory=lambda: jnp.zeros(2))  # noqa
    shoot: Bool[Array, "..."] = field(default_factory=lambda: jnp.array(True))  # self-harm by default

    @property
    def move(self):
        return ~self.shoot


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
