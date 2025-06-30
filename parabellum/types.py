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
    blast: int
    r: float


@dataclass
class Rules:
    troop = Kind(hp=120, dam=15, speed=2, reach=4, sight=4, blast=1, r=1)
    armor = Kind(hp=150, dam=12, speed=1, reach=8, sight=16, blast=3, r=2)
    plane = Kind(hp=80, dam=20, speed=4, reach=16, sight=32, blast=2, r=2)
    civil = Kind(hp=100, dam=0, speed=3, reach=5, sight=10, blast=1, r=2)
    medic = Kind(hp=100, dam=-10, speed=3, reach=5, sight=10, blast=1, r=2)

    def __post_init__(self):
        self.hp = jnp.array((self.troop.hp, self.armor.hp, self.plane.hp, self.civil.hp, self.medic.hp))
        self.dam = jnp.array((self.troop.dam, self.armor.dam, self.plane.dam, self.civil.dam, self.medic.dam))
        self.r = jnp.array((self.troop.r, self.armor.r, self.plane.r, self.civil.r, self.medic.r))
        self.speed = jnp.array(
            (self.troop.speed, self.armor.speed, self.plane.speed, self.civil.speed, self.medic.speed)
        )
        self.reach = jnp.array(
            (self.troop.reach, self.armor.reach, self.plane.reach, self.civil.reach, self.medic.reach)
        )
        self.sight = jnp.array(
            (self.troop.sight, self.armor.sight, self.plane.sight, self.civil.sight, self.medic.sight)
        )
        self.blast = jnp.array(
            (self.troop.blast, self.armor.blast, self.plane.blast, self.civil.blast, self.medic.blast)
        )


@dataclass
class Team:
    troop: int = 1
    armor: int = 0
    plane: int = 0
    civil: int = 0
    medic: int = 0

    def __post_init__(self):
        self.length: int = self.troop + self.armor + self.plane + self.civil + self.medic
        self.types: Array = jnp.repeat(
            jnp.arange(5), jnp.array((self.troop, self.armor, self.plane, self.civil, self.medic))
        )


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
    pos: Array
    type: Array
    team: Array
    dist: Array
    mask: Array
    reach: Array
    sight: Array
    speed: Array

    @property
    def ally(self):
        return (self.team == self.team[0]) & self.mask

    @property
    def enemy(self):
        return (self.team != self.team[0]) & self.mask


@dataclass
class Action:
    pos: Array
    kind: Int[Array, "..."]  # 0 = invalid, 1 = move, 2 = cast

    @property
    def invalid(self):
        return self.kind == 0

    @property
    def move(self):
        return self.kind == 1

    @property
    def cast(self):  # cast bomb, bullet or medicin
        return self.kind == 2


@dataclass
class Config:  # Remove frozen=True for now
    steps: int = 123
    place: str = "Palazzo della Civiltà Italiana, Rome, Italy"
    force: float = 0.5
    sims: int = 2
    size: int = 64
    knn: int = 2
    blu: Team = field(default_factory=lambda: Team())
    red: Team = field(default_factory=lambda: Team())
    rules: Rules = field(default_factory=lambda: Rules())

    def __post_init__(self):
        # Pre-compute everything once
        self.types: Array = jnp.concat((self.blu.types, self.red.types))
        self.teams: Array = jnp.repeat(jnp.arange(2), jnp.array((self.blu.length, self.red.length)))
        self.map: Array = geography_fn(self.place, self.size)  # Computed once here
        self.hp: Array = self.rules.hp
        self.dam: Array = self.rules.dam
        self.r: Array = self.rules.r
        self.speed: Array = self.rules.speed
        self.reach: Array = self.rules.reach
        self.sight: Array = self.rules.sight
        self.blast: Array = self.rules.blast
        self.length: int = self.blu.length + self.red.length
        self.root: Array = jnp.int32(jnp.sqrt(self.length))
