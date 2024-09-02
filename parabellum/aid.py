# aid.py
#   what you call utils.py when you want file names to be 3 letters
# by: Noah Syrkis

# imports
import os
from collections import namedtuple
from typing import Tuple
import cartopy.crs as ccrs

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
