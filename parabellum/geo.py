# geo.py
#   script for geospatial level generation
# by: Noah Syrkis

# %% Imports
import rasterio
from geopy.geocoders import Nominatim
import contextily as cx
import osmnx as ox
import geopandas as gpd
from pyproj import CRS
import numpy as np
import jax.numpy as jnp
from rasterio import features
from typing import List, Tuple, Union, Dict


# %% Types
Coords = Tuple[float, float]
BBox = Tuple[float, float, float, float]


# %% Functions
def get_coordinates(place: str) -> Coords:
    geolocator = Nominatim(user_agent="parabellum")
    point = geolocator.geocode(place)
    return point.latitude, point.longitude  # type: ignore


def project_coords(coords: Coords, from_crs: int, to_crs: int) -> Coords:
    in_proj = Proj(init=f"epsg:{from_crs}")
    out_proj = Proj(init=f"epsg:{to_crs}")
    x, y = transform(in_proj, out_proj, coords[1], coords[0])
    return y, x


# %% Test
place = "New York City"
coords = get_coordinates(place)
proj_coords = project_coords(coords, from_crs=4326, to_crs=3857)
print(f"Coordinates: {coords}\nProjected Coordinates: {proj_coords}")
