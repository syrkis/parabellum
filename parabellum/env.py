# env.py
#   parabellum env
# by: Noah Syrkis

# % Imports
from functools import partial
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.numpy.linalg as la
from jax import lax, random, vmap, jit
from jaxtyping import Array
from chex import dataclass
from parabellum.geo import geography_fn
from parabellum.types import Action, Obs, State, Rules, Team
from dataclasses import field


# %% Dataclass ################################################################
@dataclass
class Env:
    steps: int = 1000
    rules: Rules = Rules()
    place: str = "Palazzo della Civiltà Italiana, Rome, Italy"
    sims: int = 4
    size: int = 128
    knn: int = 5
    blu: Team = field(default_factory=lambda: Team())
    red: Team = field(default_factory=lambda: Team())

    def __post_init__(self):
        # Precompute static arrays to avoid JAX concretization errors
        self.init_fn = partial(init_fn, self)
        self.step_fn = partial(step_fn, self)
        self.obs_fn = partial(obs_fn, self)
        self._types = jnp.concat((self.blu.types, self.red.types))
        self._teams = jnp.repeat(jnp.arange(2), jnp.array((self.blu.length, self.red.length)))
        self._map = geography_fn(self.place, self.size)
        self._length = self._teams.size

    def init(self, rng: Array) -> Tuple[Obs, State]:
        state = self.init_fn(rng)  # without jit this takes forever
        return self.obs_fn(state), state

    def step(self, rng: Array, state: State, action: Action) -> Tuple[Obs, State]:
        return self.obs_fn(state), self.step_fn(rng, state, action)

    def traj(self, rng):
        pass

    @property
    def types(self):
        return self._types

    @property
    def teams(self):
        return self._teams

    @property
    def map(self):
        return self._map

    @property
    def length(self):
        return self._length


def knn(poss: Array, k: int, n: int):
    def aux(inputs):
        batch_pos, batch_norms = inputs
        dots = jnp.dot(batch_pos, poss.T)
        dist = jnp.maximum(batch_norms[:, None] + norms[None, :] - 2 * dots, 0)
        return lax.approx_min_k(dist, k=k, recall_target=0.8)

    norms = jnp.sum(poss**2, axis=1)
    dist, idxs = lax.map(aux, (poss.reshape((n, n, 2)), norms.reshape(n, n)))
    return dist.reshape((-1, k)) ** 0.5, idxs.reshape((-1, k))


def init_fn(env: Env, rng: Array) -> State:
    prob = jnp.ones((env.size, env.size)).at[env.map].set(0).flatten()  # Set
    flat = random.choice(rng, jnp.arange(prob.size), shape=(env.length,), p=prob / prob.sum(), replace=True)
    idxs = (flat // len(env.map), flat % len(env.map))
    pos = jnp.float32(jnp.column_stack(idxs))
    state = State(pos=pos, hp=env.rules.hp[env.types])
    return state


def obs_fn(env: Env, state: State) -> Obs:  # return info about neighbors ---
    dist, idxs = knn(state.pos, k=env.knn, n=int(env.length**0.5))
    mask = (dist < env.rules.sight[env.types[idxs][:, 0]][..., None]) | (state.hp[idxs] > 0)

    type = env.types[idxs] * mask
    team = env.teams[idxs] * mask
    hp = state.hp[idxs] * mask
    reach = env.rules.reach[type] * mask
    sight = env.rules.sight[type] * mask
    speed = env.rules.speed[type] * mask

    pos = unit_pos_fn((state.pos[idxs] - state.pos[:, None, ...]), state.pos) * mask[..., None]
    obs = Obs(pos=pos, hp=hp, type=type, dist=dist * mask, team=team, reach=reach, sight=sight, speed=speed)
    # debug.breakpoint()
    return obs


@partial(vmap, in_axes=(0, 0))
def unit_pos_fn(unit_pos, self_pos):
    return unit_pos.at[0].set(self_pos)


# @eqx.filter_jit
def step_fn(env: Env, rng: Array, state: State, action: Action) -> State:
    args = rng, env, state, action
    return State(pos=move_fn(*args), hp=state.hp)  # blast_fn(*args))  # type: ignore


def move_fn(rng: Array, env: Env, state: State, action: Action):
    speed = env.rules.speed[env.types][..., None]  # max speed of a unit (step size, really)
    pos = state.pos + action.pos.clip(-speed, speed) * action.move[..., None]  # new poss
    bound = ((pos < 0).any(axis=-1) | (pos >= env.size).any(axis=-1))[..., None]  # masking outside map
    stuff = (env.map[*pos.astype(jnp.int32).T] > 0)[..., None]  # type: ignore  # stuff in the way
    return jnp.where(bound | stuff, state.pos, pos)  # compute new position


def blast_fn(rng, env: Env, state: State, action: Action) -> Array:  # update agents ---
    dist = la.norm(state.pos[None, ...] - (state.pos + action.pos)[:, None, ...], axis=-1)  # todo make efficient
    hits = dist <= env.rules.reach[env.types][None, ...] * action.shoot[..., None]  # mask non attack act
    damage = (hits * env.rules.damage[env.types][None, ...]).sum(axis=-1)
    return state.hp - damage


# @eqx.filter_jit
def sight_fn(env: Env, state: State, dists, idxs):
    mask = dists < env.rules.sight[env.types][..., None]  # mask for removing hidden
    mask = mask  # | obstacle_fn(cfg, state.pos[idxs].astype(jnp.int8))
    return mask


# def blast_fn(env: Env, cfg: Config, state: State, action: Action):
# damage = cfg.rules.damage[cfg.types] * action.shoot
# pos = action.pos + state.pos
# debug.breakpoint()
# return state.hp


# def push_fn(poss):
#     """Push units away from each other if they're too close."""
#     return poss
#     distances = la.norm(poss[:, None] - poss, axis=-1)
#     too_close = (distances > 0) & (distances < 2.0)  # Units closer than 2 units

#     # Calculate unit displacement vectors
#     disp_vectors = poss[:, None] - poss

#     # Normalize displacement vectors (avoiding division by zero)
#     norms = distances[..., None]
#     safe_norms = jnp.where(norms > 0, norms, 1.0)
#     normalized_disp = disp_vectors / safe_norms

#     # Calculate repulsion forces (stronger when closer)
#     repulsion_strength = jnp.where(too_close, 1.0 / jnp.maximum(distances, 0.1), 0.0)
#     repulsion = normalized_disp * repulsion_strength[..., None]

#     # Sum all repulsion forces for each unit
#     total_repulsion = jnp.sum(repulsion, axis=1)

#     # Apply the repulsion with a scaling factor
#     return poss + total_repulsion * 0.5


# @eqx.filter_jit
# def scene_fn(cfg: Env) -> Scene:  # init's a scene
# unit_teams = jnp.concat((jnp.zeros(len(cfg.blu)), jnp.ones(len(cfg.red))))
# unit_types = jnp.concat((cfg.blu.types, cfg.red.types))
# terrain = geography_fn(cfg.place, cfg.size)
# return Scene(unit_teams=unit_teams, unit_types=unit_types, terrain=terrain, cfg=cfg)


# @partial(vmap, in_axes=(None, 0))  # 5 x 2 # not the best name for a fn
# def obstacle_fn(cfg: Config, pos):
# return slice_fn(cfg, pos[0], pos)


# @partial(vmap, in_axes=(None, None, 0))
# def slice_fn(cfg: Config, source, target):  # returns a 10 x 10 view with unit at top left corner, and terrain downwards
# delta = ((source - target) >= 0) * 2 - 1
# pos = jnp.sort(jnp.stack((source, source + delta * 10)), axis=0)[0]
# slice = lax.dynamic_slice(scene.terrain.building, pos, (scene.mask.shape[-1], scene.mask.shape[-1]))
# slice = lax.cond(delta[0] == 1, lambda: jnp.flip(slice), lambda: slice)
# slice = lax.cond(delta[1] == 1, lambda: jnp.flip(slice, axis=1), lambda: slice)
# return (scene.mask[*jnp.abs(source - target)] * slice).sum() == 0  # type: ignore
