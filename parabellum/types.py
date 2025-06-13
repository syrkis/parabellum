# types.py
#   parabellum types
# by: Noah Syrkis

# imports
from chex import dataclass
from jaxtyping import Array, Float32, Int
import jax.numpy as jnp
from parabellum.geo import geography_fn
from dataclasses import field


@dataclass
class Kind:
    hp: int
    dam: int
    speed: int
    reach: int
    sight: int
    reload: int
    blast: int
    radius: int


@dataclass
class Rules:
    troop = Kind(hp=120, dam=15, speed=2, reach=5, sight=8, reload=1, blast=1, radius=2)
    armor = Kind(hp=150, dam=12, speed=1, reach=10, sight=16, reload=2, blast=3, radius=2)
    plane = Kind(hp=80, dam=20, speed=4, reach=20, sight=32, reload=4, blast=2, radius=2)
    civil = Kind(hp=100, dam=0, speed=3, reach=3, sight=10, reload=3, blast=1, radius=2)
    medic = Kind(hp=100, dam=-10, speed=3, reach=3, sight=10, reload=3, blast=1, radius=2)

    def __post_init__(self):
        self.hp = jnp.array((self.troop.hp, self.armor.hp, self.plane.hp, self.civil.hp, self.medic.hp))
        self.dam = jnp.array((self.troop.dam, self.armor.dam, self.plane.dam, self.civil.dam, self.medic.dam))
        self.radii = jnp.array(
            (self.troop.radius, self.armor.radius, self.plane.radius, self.civil.radius, self.medic.radius)
        )
        self.speed = jnp.array(
            (self.troop.speed, self.armor.speed, self.plane.speed, self.civil.speed, self.medic.speed)
        )
        self.reach = jnp.array(
            (self.troop.reach, self.armor.reach, self.plane.reach, self.civil.reach, self.medic.reach)
        )
        self.sight = jnp.array(
            (self.troop.sight, self.armor.sight, self.plane.sight, self.civil.sight, self.medic.sight)
        )
        self.reload = jnp.array(
            (self.troop.reload, self.armor.reload, self.plane.reload, self.civil.reload, self.medic.reload)
        )
        self.blast = jnp.array(
            (self.troop.blast, self.armor.blast, self.plane.blast, self.civil.blast, self.medic.blast)
        )


@dataclass
class Team:
    troop: int = 100
    armor: int = 100
    plane: int = 100
    civil: int = 100
    medic: int = 100

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


@dataclass
class Config:  # Remove frozen=True for now
    steps: int = 100
    place: str = "Palazzo della Civiltà Italiana, Rome, Italy"
    force: float = 0.5
    sims: int = 9
    size: int = 64
    knn: int = 5
    blu: Team = field(default_factory=lambda: Team())
    red: Team = field(default_factory=lambda: Team())
    rules: Rules = field(default_factory=lambda: Rules())

    def __post_init__(self):
        # Pre-compute everything once
        self.types: Array = jnp.concat((self.blu.types, self.red.types))
        self.teams: Array = jnp.repeat(jnp.arange(2), jnp.array((self.blu.length, self.red.length)))
        self.map: Array = geography_fn(self.place, self.size)  # Computed once here
        self.length: int = self.blu.length + self.red.length
        self.root: Array = jnp.int32(jnp.sqrt(self.length))
        self.hp: Array = self.rules.hp
        self.dam: Array = self.rules.dam
        self.radii: Array = self.rules.radii
        self.speed: Array = self.rules.speed
        self.reach: Array = self.rules.reach
        self.sight: Array = self.rules.sight
        self.reload: Array = self.rules.reload
        self.blast: Array = self.rules.blast
        # Copy all rules properties to self for easier access
        # for attr in ['hp', 'dam', 'radii', 'speed', 'reach', 'sight', 'reload', 'blast']:
        # setattr(self, attr, getattr(self.rules, attr))
