# %% env.py
#   parabellum env
# by: Noah Syrkis

# Imports
from typing import Tuple

import geopandas as gpd
import jax.numpy as jnp
import jaxkd as jk
import osmnx as ox
from cachier import cachier
from geopy.geocoders import Nominatim
from jax import random
from jaxtyping import Array
from omegaconf import DictConfig
from rasterio import features, transform
from shapely.geometry import Point, box

# import numpy as np
# import matplotlib.pyplot as plt
# import contextily as ctx
from parabellum.types import Action, Obs, State


# %% Dataclass
class Env:
    def __init__(self, cfg: DictConfig):
        # config
        self.cfg = cfg
        self.map = jnp.array(world_fn(cfg.place, cfg.size))
        self.num = sum([sum(x.values()) for x in cfg.teams.values()])
        self.see = int((self.num / jnp.log(self.num)).item())

        # length n units
        self.types = jnp.concat([jnp.repeat(jnp.arange(5), jnp.array(list(x.values()))) for x in cfg.teams.values()])
        self.teams = jnp.repeat(jnp.arange(2), jnp.array((sum(cfg.teams.blu.values()), sum(cfg.teams.red.values()))))

        # constans
        self.blu_num = self.teams.sum().item()
        self.red_num = self.num - self.blu_num

        # length n types
        self.damage = jnp.array(list(map(lambda x: getattr(x, "damage"), cfg.rules.values())))
        self.radius = jnp.array(list(map(lambda x: getattr(x, "radius"), cfg.rules.values())))
        self.health = jnp.array(list(map(lambda x: getattr(x, "health"), cfg.rules.values())))
        self.speed = jnp.array(list(map(lambda x: getattr(x, "speed"), cfg.rules.values())))
        self.reach = jnp.array(list(map(lambda x: getattr(x, "reach"), cfg.rules.values())))
        self.sight = jnp.array(list(map(lambda x: getattr(x, "sight"), cfg.rules.values())))
        self.blast = jnp.array(list(map(lambda x: getattr(x, "blast"), cfg.rules.values())))

    def init(self, rng: Array) -> Tuple[Obs, State]:
        state = init_fn(self, rng)  # without jit this takes forever
        return obs_fn(self, state), state

    def step(self, rng: Array, state: State, action: Action) -> Tuple[Obs, State]:
        state = step_fn(self, rng, state, action)
        return obs_fn(self, state), state


@cachier()
def world_fn(place, size):
    # Get location coordinates
    location = Nominatim(user_agent="parabellum").geocode(place)

    # Get building footprints from OSM within a radius
    data = ox.features_from_point((location.latitude, location.longitude), tags={"building": True}, dist=size // 2)

    # Project to a metric CRS to work in meters
    data = data.to_crs(data.estimate_utm_crs())

    # Create a point from the location and project it
    center = gpd.GeoSeries([Point(location.longitude, location.latitude)], crs="EPSG:4326").to_crs(data.crs).iloc[0]  # type: ignore
    # Create exact square bounding box centered on location
    bbox = box(center.x - size // 2, center.y - size // 2, center.x + size // 2, center.y + size // 2)

    # Clip buildings to exact bounding box
    data["geometry"] = data.geometry.clip(bbox)

    # Filter out empty/invalid geometries
    data = data[~data.geometry.is_empty & data.geometry.is_valid & data.geometry.notna()]

    # Transformation stuff - now using the exact bbox bounds
    t = transform.from_bounds(*bbox.bounds, size, size)  # type: ignore

    # Rasterize buildings into a binary grid
    raster = features.rasterize([(geom, 1) for geom in data.geometry if geom], (size, size), transform=t)

    return raster


# %% Functions
def init_fn(env: Env, rng: Array) -> State:
    pos = spawn(env, rng)
    return State(pos=pos, hp=jnp.float32(env.health[env.types]))


def spawn(env, rng) -> Array:
    loc = jnp.zeros_like(env.map).at[jnp.arange(env.cfg.size // 4), jnp.arange(env.cfg.size // 4)].set(1)
    aux = lambda x, loc: random.choice(rng, jnp.arange(env.cfg.size**2), (x,), True, ((1 - env.map) & loc).flatten())  # noqa
    flat = jnp.concatenate((aux(env.blu_num, loc), aux(env.red_num, jnp.flip(loc).T)))
    pos = jnp.float32(jnp.column_stack((flat // env.cfg.size, flat % env.cfg.size)))
    return pos


def obs_fn(env: Env, state: State) -> Obs:  # return info about neighbors ---
    idxs, dist = jk.extras.query_neighbors_pairwise(state.pos, state.pos, k=env.cfg.knn)
    mask: Array = dist < env.sight[env.types[idxs][:, 0]][..., None] | (state.hp[idxs] > 0)
    pos: Array = (state.pos[idxs] - state.pos[:, None, ...]).at[:, 0, :].set(state.pos) * mask[..., None]
    args = state.hp, env.types, env.teams, env.reach, env.sight, env.speed
    hp, type, team, reach, sight, speed = map(lambda x: x[idxs] * mask, args)
    return Obs(
        pos=pos, dist=dist, hp=hp, type=type, team=team, reach=reach, sight=sight, speed=speed, mask=mask, idx=idxs * mask
    )
    # ,  # TODO: note that units not in sight has idx 0 (but so does the first unit of blue team)


def step_fn(env: Env, rng: Array, state: State, action: Action) -> State:
    idx, norm = jk.extras.query_neighbors_pairwise(state.pos + action.pos, state.pos, k=2)  # neast neighs?
    hp = blast_fn(rng, env, state, action, idx, norm)
    pos = move_fn(rng, env, state, action, idx, norm)
    return State(pos=pos, hp=hp)  # type: ignore


def move_fn(rng: Array, env: Env, state: State, action: Action, idx: Array, norm: Array) -> Array:
    speed = env.speed[env.types][..., None]  # max speed of a unit (step size, really)
    pos = (
        state.pos
        + action.pos.clip(-speed, speed) * action.move[..., None]
        + random.normal(rng, state.pos.shape) * env.cfg.noise
    )
    mask = ((pos < 0).any(axis=-1) | ((pos >= env.cfg.size).any(axis=-1)) | (env.map[*jnp.int32(pos).T] > 0) | (state.hp <= 0))[
        ..., None
    ]
    return jnp.where(mask, state.pos, pos)  # compute new position


def blast_fn(rng: Array, env: Env, state: State, action: Action, idx: Array, norm: Array) -> Array:
    dam = (env.damage[env.types] * action.cast * (state.hp > 0))[..., None] * jnp.ones_like(idx)
    return jnp.float32(state.hp - jnp.zeros(env.types.size).at[idx.flatten()].add(dam.flatten()))


def push_fn(env: Env, rng: Array, idx: Array, norm: Array, pos: Array) -> Array:
    # return pos + random.normal(rng, pos.shape)
    # params need to be tweaked, and matched with unit size
    # pos_diff = pos[:, None, :] - pos[idx]  # direction away from neighbors
    # mask = (norm < env.radius[env.types][..., None]) & (norm > 0)
    # pos = pos + jnp.where(mask[..., None], pos_diff * env.cfg.force / (norm[..., None] + 1e-6), 0.0).sum(axis=1)
    raise NotImplementedError("push_fn is not implemented yet")
