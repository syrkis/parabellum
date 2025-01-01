# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from chex import dataclass
from jax import lax, random
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


@dataclass
class Env:
    cfg: Conf
    geo: pb.tps.Terrain
    num_allies = 10
    num_rivals = 10
    type_health = jnp.array([100, 100, 100])
    type_damage = jnp.array([10, 10, 10])
    type_ranges = jnp.array([10, 10, 10])
    type_sights = jnp.array([10, 10, 10])
    type_speeds = jnp.array([1, 1, 1])
    type_reload = jnp.array([10, 10, 10])

    def reset(self, rng: Array) -> Tuple[Obs, State]:
        return init_fn(rng, self.cfg, self)

    def step(self, rng, state, action) -> Tuple[Obs, State]:
        return obs_fn(self.cfg, self, state), step_fn(rng, self, state, action)


# %% Functions
@eqx.filter_jit
def init_fn(rng: Array, cfg: Conf, env: Env) -> Tuple[Obs, State]:  # initialize -----
    keys, num_agents = random.split(rng), env.num_allies + env.num_rivals  # meta ----
    types = random.choice(keys[0], jnp.arange(env.type_damage), (num_agents,))  # type
    pos = random.uniform(keys[1], (num_agents, 2), minval=0, maxval=cfg.size)  # pos -
    teams = jnp.where(jnp.arange(num_agents) < env.num_allies, 0, 1)  # agent team ---
    health = jnp.take(env.type_health, types)  # health of agents by type for starting
    state = State(pos=pos, health=health, types=types, teams=teams)  # state of agents
    return obs_fn(cfg, env, state), state  # return observation and state of agents --


@eqx.filter_jit
def obs_fn(cfg: Conf, env: Env, state: State) -> Obs:  # return infoabout neighbors ---
    distances = jnp.linalg.norm(state.pos[:, None] - state.pos, axis=-1)  # all dist --
    dist, idxs = lax.approx_min_k(distances, cfg.knn)  # dists and idxs of close by ---
    directions = jnp.take(state.pos, idxs, axis=0) - state.pos[:, None]  # direction --
    obs = jnp.stack([dist, state.health[idxs], state.types[idxs]], axis=-1)  # concat -
    mask = dist < env.type_ranges[state.types][..., None]  # mask for removing hidden -
    return jnp.concat([obs, directions], axis=-1) * mask[..., None]  # an observation -


@eqx.filter_jit
def step_fn(rng: Array, env: Env, state: State, action) -> State:  # update agents ---
    pos = state.pos + action.direction * env.type_speeds[state.types]  # move agent --
    hp = state.health - action.attack * env.type_damage[state.types]  # attack stuff -
    return State(pos=pos, health=hp, types=state.types, teams=state.teams)  # return -


# %% Main #####################################################################
cfg, geo = Conf(), pb.geo.geography_fn("Copenhagen, Denmark")
env = Env(cfg=Conf(), geo=geo)
obs, state = env.reset(rng := random.PRNGKey(0))
