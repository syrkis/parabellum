# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
from typing import Tuple

import equinox as eqx

# import esch
import jax.numpy as jnp
from chex import dataclass
from jax import lax, random
from jaxtyping import Array

# import parabellum as pb

# %% Types ####################################################################
Obs = Array


# %% Dataclasses ################################################################
@dataclass
class State:
    unit_position: Array
    unit_health: Array
    unit_cooldown: Array


@dataclass
class Conf:
    place: str = "Copenhagen, Denmark"
    knn: int = 10
    size: int = 100


@dataclass
class Env:
    cfg: Conf
    unit_types: Array
    num_agents: int
    num_allies: int
    num_enemies: int
    line_of_sight: Array
    unit_type_radiuses: Array
    unit_type_health: Array
    unit_type_attacks: Array
    unit_type_attack_ranges: Array
    unit_type_sight_ranges: Array
    unit_type_velocities: Array  # distance per steps
    unit_type_weapon_cooldowns: Array  # in number of steps

    def reset(self, rng: Array) -> Tuple[Obs, State]:
        return init_fn(rng, self.cfg, self)

    def step(self, rng, state, action) -> Tuple[Obs, State]:
        return obs_fn(self, state), step_fn(self, state, action)


@dataclass
class Action:
    health: Array  # attack/heal
    moving: Array  # move agents


# %% Functions ################################################################
@eqx.filter_jit
def init_fn(rng: Array, cfg: Conf, env: Env) -> Tuple[Obs, State]:  # initialize -----
    keys, num_agents = random.split(rng), env.num_allies + env.num_enemies  # meta ----
    types = random.choice(keys[0], jnp.arange(env.unit_type_attacks.size), (num_agents,))  #
    pos = random.uniform(keys[1], (num_agents, 2), minval=0, maxval=cfg.size)  # pos -
    health = jnp.take(env.unit_type_health, types)  # health of agents by type for starting
    state = State(unit_position=pos, unit_health=health, unit_cooldown=jnp.zeros(num_agents))  # state --
    return obs_fn(env, state), state  # return observation and state of agents --


@eqx.filter_jit
def obs_fn(env: Env, state: State) -> Obs:  # return infoabout neighbors ---
    distances = jnp.linalg.norm(state.unit_position[:, None] - state.unit_position, axis=-1)  # all dist --
    dist, idxs = lax.approx_min_k(distances, env.cfg.knn)  # dists and idx
    directions = jnp.take(state.unit_position, idxs, axis=0) - state.unit_position[:, None]  # direction --
    obs = jnp.stack([idxs, dist, state.unit_health[idxs], env.unit_types[idxs]], axis=-1)  # --
    mask = dist < env.unit_type_sight_ranges[env.unit_types][..., None]  # mask for removing hidden -
    return jnp.concat([obs, directions], axis=-1) * mask[..., None]  # an observation -


@eqx.filter_jit
def step_fn(env: Env, state: State, action: Action) -> State:  # update agents ---
    pos = state.unit_position + action.moving * env.unit_type_velocities[env.unit_types][..., None]
    hp = state.unit_health + action.health * env.unit_type_attacks[env.unit_types][..., None]  #
    return State(unit_position=pos, unit_health=hp, unit_cooldown=state.unit_cooldown)  # return -


def render_fn(env: Env, state: State) -> None:  # render the state of the agents ---
    pass  # render the state of the agents -----------------------------------------


# %% Main #####################################################################
# cfg, geo = Conf(), pb.geo.geography_fn("Copenhagen, Denmark")
# env = Env(cfg=cfg, geo=geo)
# obs, state = env.reset(rng := random.PRNGKey(0))


# state_seq = []
# for _ in range(100):
# rng, key = random.split(rng)
# action = Action(
# health=random.choice(key, jnp.array([0, 1]), (env.num_allies + env.num_rivals,)),
# moving=random.uniform(key, (env.num_allies + env.num_rivals, 2), minval=-1, maxval=1),
# )
#
# state = step_fn(env, state, action)
# state_seq.append((state, action))
# break
#
# with open("state.pkl", "wb") as f:
# pickle.dump(state, f)
