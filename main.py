# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
from typing import Tuple

import esch
import jax.numpy as jnp
from chex import dataclass
from jax import random, lax
import equinox as eqx
from jaxtyping import Array

import parabellum as pb

# %% Types ####################################################################
Obs = Array


# %% Dataclasses ################################################################
@dataclass
class State:
    pos: Array
    types: Array
    teams: Array
    health: Array


@dataclass
class Conf:
    place: str = "Copenhagen, Denmark"
    knn: int = 10
    size: int = 100


# %% Functions
@eqx.filter_jit
def init_fn(rng: Array, cfg: Conf, env) -> Tuple[Obs, State]:  # initialize agents ###
    keys = random.split(rng, 2)  # split keys for random number generation for agentts
    types = random.choice(keys[0], env.type_radius, (env.num_agents,))  # random types
    pos = random.uniform(keys[1], (env.num_agents, 2), minval=0, maxval=cfg.size)  # p
    teams = jnp.where(jnp.arange(env.num_agents) < env.num_allies, 0, 1)  # agent team
    health = jnp.take(env.type_health, types)  # health of agents by type for starting
    state = State(pos=pos, health=health, types=types, teams=teams)  # state of agents
    return obs_fn(cfg, env, state), state  # return observation and state of agents ##


@eqx.filter_jit
def obs_fn(cfg: Conf, env, state: State) -> Obs:  # return infoabout nearby neighborss
    distances = jnp.linalg.norm(state.pos[:, None] - state.pos, axis=-1)  # all distss
    dist, idxs = lax.approx_min_k(distances, cfg.knn)  # dists and idxs of my neighhss
    directions = jnp.take(state.pos, idxs, axis=0) - state.pos[:, None]  # directionss
    obs = jnp.stack([dist, state.health[idxs], state.types[idxs]], axis=-1)  # concats
    mask = dist < env.type_ranges[state.types][..., None]  # mask for removing hiddens
    return jnp.concat([obs, directions], axis=-1) * mask[..., None]  # an observations


@eqx.filter_jit
def step_fn(rng: Array, env, state: State, action) -> State:  # update agents state---
    pos = state.pos + action.direction * env.type_speeds[state.types]  # move agent---
    hp = state.health - action.attack * env.type_attack[state.types]  # attack stuf---
    return State(pos=pos, health=hp, types=state.types, teams=state.teams)  # retur---


# %% Gym
class Parabellum:
    def __init__(self, cfg: Conf):
        self.cfg = cfg
        self.geo = pb.geo.geography_fn(cfg.place, cfg.size)
        self.num_agents = 20
        self.num_allies = 10
        self.num_rivals = 10
        self.type_radius = jnp.array([5, 5, 5])
        self.type_health = jnp.array([100, 100, 100])
        self.type_attack = jnp.array([10, 10, 10])
        self.type_ranges = jnp.array([10, 10, 10])
        self.type_sights = jnp.array([10, 10, 10])
        self.type_speeds = jnp.array([1, 1, 1])
        self.type_reload = jnp.array([10, 10, 10])

    def reset(self, rng: Array) -> Tuple[Obs, State]:
        return init_fn(rng, self.cfg, self)

    def step(self, rng, state, action) -> Tuple[Obs, State]:
        state = step_fn(rng, self, state, action)
        obs = obs_fn(self.cfg, self, state)
        return obs, state


# %% Main #####################################################################
env = Parabellum(cfg := Conf())
obs, state = env.reset(rng := random.PRNGKey(0))
# %% Plot #####################################################################
# esch.grid(env.geo.building, path="tmp.svg")
