# aid.py
#   what you call utils.py when you want file names to be 3 letters
# by: Noah Syrkis

# imports
import os
from collections import namedtuple
from typing import Tuple
import cartopy.crs as ccrs
import jax.numpy as jnp
import numpy as np
from PIL import Image

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


def anim(seq, scale=8, width=10):  # animate positions
    idxs = jnp.concatenate((jnp.arange(seq.shape[0]).repeat(seq.shape[1])[None, ...], seq.reshape(2, -1)))
    imgs = np.array(jnp.zeros((seq.shape[0], width, width)).at[*idxs].set(255)).astype(np.uint8)  # setting color
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs]  # type: ignore
    imgs[0].save("output.gif", save_all=True, append_images=imgs[1:], duration=256, loop=0)
