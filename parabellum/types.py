# types.py
#   parabellum types
# by: Noah Syrkis

# imports
from chex import dataclass
from jaxtyping import Array, Float32, Int
import jax.numpy as jnp


@dataclass
class Kind:
    hp: int
    damage: int
    speed: int
    reach: int
    sight: int
    reload: int
    blast: int


class Rules:
    troop = Kind(hp=120, damage=15, speed=2, reach=5, sight=8, reload=1, blast=1)
    armor = Kind(hp=150, damage=12, speed=1, reach=10, sight=16, reload=2, blast=3)
    plane = Kind(hp=80, damage=20, speed=4, reach=20, sight=32, reload=4, blast=2)
    civil = Kind(hp=100, damage=0, speed=3, reach=3, sight=10, reload=3, blast=1)
    medic = Kind(hp=100, damage=-10, speed=3, reach=3, sight=10, reload=3, blast=1)

    @property
    def hp(self) -> Array:
        return jnp.array((self.troop.hp, self.armor.hp, self.plane.hp, self.civil.hp, self.medic.hp))

    @property
    def damage(self) -> Array:
        return jnp.array(
            (self.troop.damage, self.armor.damage, self.plane.damage, self.civil.damage, self.medic.damage)
        )

    @property
    def reach(self) -> Array:
        return jnp.array((self.troop.reach, self.armor.reach, self.plane.reach, self.civil.reach, self.medic.reach))

    @property
    def speed(self) -> Array:
        return jnp.array((self.troop.speed, self.armor.speed, self.plane.speed, self.civil.speed, self.medic.speed))

    @property
    def sight(self) -> Array:
        return jnp.array((self.troop.sight, self.armor.sight, self.plane.sight, self.civil.sight, self.medic.sight))

    @property
    def reload(self) -> Array:
        return jnp.array(
            (self.troop.reload, self.armor.reload, self.plane.reload, self.civil.reload, self.medic.reload)
        )

    @property
    def blast(self) -> Array:
        return jnp.array((self.troop.blast, self.armor.blast, self.plane.blast, self.civil.blast, self.medic.blast))


@dataclass
class Team:
    troop: int = 1_000
    armor: int = 1_000
    plane: int = 1_000
    civil: int = 1_000
    medic: int = 1_000

    def __post_init__(self):
        # Precompute static arrays to avoid JAX concretization errors
        self._length = self.troop + self.armor + self.plane + self.civil + self.medic
        self._types = jnp.repeat(jnp.arange(5), jnp.array((self.troop, self.armor, self.plane, self.civil, self.medic)))

    @property
    def length(self):
        return self._length

    @property
    def types(self) -> Array:
        return self._types


# dataclasses
@dataclass
class State:
    pos: Array
    hp: Array
    # target: Array


@dataclass
class Obs:
    # idxs: Array
    hp: Array
    type: Array
    team: Array
    dist: Array
    pos: Array
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
    pos: Float32[Array, "... 2"]  # noqa
    kind: Int[Array, "..."]  # 0 = invalid, 1 = move, 2 = shoot

    @property
    def invalid(self):
        return self.kind == 0

    @property
    def move(self):
        return self.kind == 1

    @property
    def shoot(self):
        return self.kind == 2
