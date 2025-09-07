# types.py
#   parabellum types
# by: Noah Syrkis

# imports
from chex import dataclass
from jaxtyping import Array, Bool
from jax import lax
import jax.numpy as jnp
# from parabellum.geo import geography_fn


@dataclass
class State:
    pos: Array
    hp: Array
    # target: Array


@dataclass
class Obs:
    hp: Array
    pos: Array
    idx: Array
    type: Array
    team: Array
    dist: Array
    mask: Array
    reach: Array
    sight: Array
    speed: Array

    @property
    def ally(self):
        return (self.team == self.team[..., 0, None]) & self.mask

    @property
    def enemy(self):
        return (self.team != self.team[..., 0, None]) & self.mask

    @property
    def krypt(self):  # hard coding rules
        return (self.type == (self.type[..., 0, None] + 1) % 3) & self.enemy


@dataclass
class Action:
    pos: Array
    move: Bool[Array, "..."]  # 0 = invalid, 1 = move, 2 = cast

    @property
    def cast(self):  # cast bomb, bullet or medicin
        return ~self.move
