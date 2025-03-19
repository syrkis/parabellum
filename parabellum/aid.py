# aid.py
#   what you call utils.py when you want file names to be 3 letters
# by: Noah Syrkis

# imports
from collections import namedtuple
import cartopy.crs as ccrs
import jax.numpy as jnp

# types
BBox = namedtuple("BBox", ["north", "south", "east", "west"])  # type: ignore


# coordinate function
def to_mercator(bbox: BBox) -> BBox:
    proj = ccrs.Mercator()
    west, south = proj.transform_point(bbox.west, bbox.south, ccrs.PlateCarree())
    east, north = proj.transform_point(bbox.east, bbox.north, ccrs.PlateCarree())
    return BBox(north=north, south=south, east=east, west=west)


def to_platecarree(bbox: BBox) -> BBox:
    proj = ccrs.PlateCarree()
    west, south = proj.transform_point(bbox.west, bbox.south, ccrs.Mercator())

    east, north = proj.transform_point(bbox.east, bbox.north, ccrs.Mercator())
    return BBox(north=north, south=south, east=east, west=west)


def obstacle_mask_fn(limit):
    def aux(i, j):
        xs = jnp.linspace(0, i + 1, i + j + 1)
        ys = jnp.linspace(0, j + 1, i + j + 1)
        cc = jnp.stack((xs, ys)).astype(jnp.int8)
        mask = jnp.zeros((limit, limit)).at[*cc].set(1)
        return mask

    x = jnp.repeat(jnp.arange(limit), limit)
    y = jnp.tile(jnp.arange(limit), limit)
    mask = jnp.stack([aux(*c) for c in jnp.stack((x, y)).T])
    return mask.astype(jnp.int8).reshape(limit, limit, limit, limit)
