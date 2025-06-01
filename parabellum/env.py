# env.py
#   parabellum env
# by: Noah Syrkis

# % Imports
import jax.numpy as jnp
from jax import random, Array, lax, vmap, debug
import jax.numpy.linalg as la
from typing import Tuple
from functools import partial
# from einops import rearrange

from parabellum.geo import geography_fn
from parabellum.types import Action, State, Obs, Scene
from parabellum.utils import obstacle_mask_fn

import equinox as eqx
from collections import namedtuple


# %% Rules ####################################################################
kind = namedtuple("kind", ["health", "damage", "speed", "reach", "sight", "reload"])

# Infantry (Rock) - Strong vs Armor, Weak vs Airplane
infantry = kind(health=120, damage=15, speed=2, reach=2, sight=12, reload=2)

# Airplane (Paper) - Strong vs Infantry, Weak vs Armor
airplane = kind(health=80, damage=20, speed=4, reach=6, sight=15, reload=1)

# Armor (Scissors) - Strong vs Airplane, Weak vs Infantry
armor = kind(health=150, damage=12, speed=1, reach=3, sight=8, reload=3)

kinds = dict(infantry=infantry, airplane=airplane, armor=armor)


# %% Dataclass ################################################################
class Env:
    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self, rng: Array, scene: Scene) -> Tuple[Obs, State]:
        return init_fn(rng, self, scene)

    def step(self, rng: Array, scene: Scene, state: State, action: Action) -> Tuple[Obs, State]:
        return obs_fn(self, scene, state), step_fn(rng, self, scene, state, action)

    @property
    def num_units(self):
        return sum(self.cfg.blue.values()) + sum(self.cfg.red.values())

    @property
    def num_blue(self):
        return sum(self.cfg.blue.values())

    @property
    def num_red(self):
        return sum(self.cfg.red.values())


# @eqx.filter_jit
def knn(coords, k, n):
    def aux(inputs):
        batch_coord, batch_norms = inputs
        dots = jnp.dot(batch_coord, coords.T)
        dist = jnp.maximum(batch_norms[:, None] + norms[None, :] - 2 * dots, 0)
        return lax.approx_min_k(dist, k=k)

    norms = jnp.sum(coords**2, axis=1)
    dist, idxs = lax.map(aux, (coords.reshape((n, n, 2)), norms.reshape(n, n)))
    return dist.reshape((-1, k)) ** 0.5, idxs.reshape((-1, k))


# %% Functions
# @eqx.filter_jit
def init_fn(rng: Array, env: Env, scene: Scene) -> Tuple[Obs, State]:
    keys = random.split(rng, 3)
    health = jnp.ones(env.num_units) * scene.unit_type_health[scene.unit_types]

    # Create a probability mask: 0 where buildings exist, uniform elsewhere
    terrain_shape = scene.terrain.building.shape
    prob_mask = jnp.ones(terrain_shape)  # Start with all ones
    prob_mask = prob_mask.at[scene.terrain.building].set(0)  # Set to 0 where buildings exist

    # Normalize the mask to create a probability distribution
    prob_mask = prob_mask / jnp.sum(prob_mask)

    # Flatten the mask and create indices for all grid positions
    flat_probs = prob_mask.flatten()
    indices = jnp.arange(flat_probs.size)

    # Sample positions using categorical distribution
    flat_indices = random.choice(keys[0], indices, shape=(env.num_units,), p=flat_probs, replace=True)

    # Convert flat indices back to 2D coordinates
    pos = jnp.float32(jnp.column_stack([flat_indices // terrain_shape[1], flat_indices % terrain_shape[1]]))

    state = State(coords=pos + random.uniform(rng, pos.shape) * 0.1 - 0.5, health=health)
    return obs_fn(env, scene, state), state


# @eqx.filter_jit  # knn from env.cfg never changes, so we can jit it
def obs_fn(env, scene: Scene, state: State) -> Obs:  # return info about neighbors ---
    dist, idxs = knn(state.coords, k=env.cfg.knn, n=int(env.num_units**0.5))
    mask = (dist < scene.unit_type_sight[scene.unit_types[idxs][:, 0]][..., None]) | (state.health[idxs] > 0)

    type = scene.unit_types[idxs] * mask
    team = scene.unit_teams[idxs] * mask
    health = state.health[idxs] * mask
    reach = scene.unit_type_reach[type] * mask
    sight = scene.unit_type_sight[type] * mask
    speed = scene.unit_type_speed[type] * mask

    coord = unit_pos_fn((state.coords[idxs] - state.coords[:, None, ...]), state.coords) * mask[..., None]
    obs = Obs(coord=coord, health=health, type=type, dist=dist * mask, team=team, reach=reach, sight=sight, speed=speed)
    # debug.breakpoint()
    return obs


@partial(vmap, in_axes=(0, 0))
def unit_pos_fn(unit_pos, self_pos):
    return unit_pos.at[0].set(self_pos)


@eqx.filter_jit
def step_fn(rng, env: Env, scene: Scene, state: State, action: Action) -> State:  # update agents ---
    # deltas = action.coord / jnp.linalg.norm(action.coord + random.normal(rng) * 0.01, axis=1)[..., None]
    speeds = scene.unit_type_speed[scene.unit_types][..., None]
    coords = state.coords + action.coord.clip(-speeds, speeds) * action.move[..., None]
    # coords = push_fn(coords + random.normal(rng, coords.shape) * 0.01)
    bounds = ((coords < 0).any(axis=-1) | (coords >= env.cfg.size).any(axis=-1))[..., None]
    builds = (scene.terrain.building[*coords.astype(jnp.int32).T] > 0)[..., None]
    coords = jnp.where(bounds | builds, state.coords, coords)
    # health = blast_fn(rng, env, scene, state, action)
    return State(coords=coords, health=state.health)  # type: ignore


def push_fn(coords):
    """Push units away from each other if they're too close."""
    return coords
    distances = la.norm(coords[:, None] - coords, axis=-1)
    too_close = (distances > 0) & (distances < 2.0)  # Units closer than 2 units

    # Calculate unit displacement vectors
    disp_vectors = coords[:, None] - coords

    # Normalize displacement vectors (avoiding division by zero)
    norms = distances[..., None]
    safe_norms = jnp.where(norms > 0, norms, 1.0)
    normalized_disp = disp_vectors / safe_norms

    # Calculate repulsion forces (stronger when closer)
    repulsion_strength = jnp.where(too_close, 1.0 / jnp.maximum(distances, 0.1), 0.0)
    repulsion = normalized_disp * repulsion_strength[..., None]

    # Sum all repulsion forces for each unit
    total_repulsion = jnp.sum(repulsion, axis=1)

    # Apply the repulsion with a scaling factor
    return coords + total_repulsion * 0.5


def blast_fn(rng, env: Env, scene: Scene, state: State, action: Action):  # update agents ---
    dist = la.norm(state.coords[None, ...] - (state.coords + action.coord)[:, None, ...], axis=-1)
    hits = dist <= scene.unit_type_reach[scene.unit_types][None, ...] * action.shoot[..., None]  # mask non attack act
    damage = (hits * scene.unit_type_damage[scene.unit_types][None, ...]).sum(axis=-1)
    return state.health - damage


# @eqx.filter_jit
def scene_fn(cfg):  # init's a scene
    aux = lambda key: jnp.array([x.__getattribute__(key) for x in kinds.values()])  # noqa
    attrs = ["health", "damage", "reload", "reach", "sight", "speed"]
    kwargs = {f"unit_type_{a}": aux(a) for a in attrs} | {"terrain": geography_fn(cfg.place, cfg.size)}
    num_blue, num_red = sum(cfg.blue.values()), sum(cfg.red.values())
    unit_teams = jnp.concat((jnp.ones(num_blue), -jnp.ones(num_red))).astype(jnp.int32)
    aux = lambda t: jnp.concat([jnp.zeros(x) + i for i, x in enumerate([x[1] for x in sorted(cfg[t].items())])])  # noqa
    unit_types = jnp.concat((aux("blue"), aux("red"))).astype(jnp.int32)
    mask = obstacle_mask_fn(max([x.sight for x in kinds.values()]))
    return Scene(unit_teams=unit_teams, unit_types=unit_types, mask=mask, **kwargs)  # type: ignore


@eqx.filter_jit
def sight_fn(scene: Scene, state: State, dists, idxs):
    mask = dists < scene.unit_type_sight[scene.unit_types][..., None]  # mask for removing hidden
    mask = mask | obstacle_fn(scene, state.coords[idxs].astype(jnp.int8))
    return mask


@partial(vmap, in_axes=(None, 0))  # 5 x 2 # not the best name for a fn
def obstacle_fn(scene, pos):
    return slice_fn(scene, pos[0], pos)


@partial(vmap, in_axes=(None, None, 0))
def slice_fn(scene, source, target):  # returns a 10 x 10 view with unit at top left corner, and terrain downwards
    delta = ((source - target) >= 0) * 2 - 1
    coord = jnp.sort(jnp.stack((source, source + delta * 10)), axis=0)[0]
    slice = lax.dynamic_slice(scene.terrain.building, coord, (scene.mask.shape[-1], scene.mask.shape[-1]))
    slice = lax.cond(delta[0] == 1, lambda: jnp.flip(slice), lambda: slice)
    slice = lax.cond(delta[1] == 1, lambda: jnp.flip(slice, axis=1), lambda: slice)
    return (scene.mask[*jnp.abs(source - target)] * slice).sum() == 0
