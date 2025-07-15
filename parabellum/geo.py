# %% geo.py
#   script for geospatial level generation
# by: Noah Syrkis

# Imports
from collections import namedtuple
from typing import Tuple

from typing import Dict
import geopandas as gpd
import jax.numpy as jnp
import osmnx as ox
from cachier import cachier
from geopy.distance import distance
from geopy.geocoders import Nominatim
from jaxtyping import Array
from rasterio import features, transform
from shapely import box

# %% Types
Coords = Tuple[float, float]
BBox = namedtuple("BBox", ["north", "south", "east", "west"])  # type: ignore

# %% Constants
tags: Dict[str, bool] = {"building": True}  # , "water": True}


# %% Coordinate function
def get_coordinates(place: str) -> Coords:
    geolocator = Nominatim(user_agent="parabellum")
    point = geolocator.geocode(place)
    return point.latitude, point.longitude  # type: ignore


def get_bbox(place: str, buffer) -> BBox:
    """Get bounding box from place name in crs 4326."""
    coords = get_coordinates(place)
    north = distance(meters=buffer).destination(coords, bearing=0).latitude
    south = distance(meters=buffer).destination(coords, bearing=180).latitude
    east = distance(meters=buffer).destination(coords, bearing=90).longitude
    west = distance(meters=buffer).destination(coords, bearing=270).longitude
    return BBox(north, south, east, west)  # type: ignore


@cachier()
def geography_fn(place, buffer):
    bbox = get_bbox(place, buffer)
    map_data = ox.features_from_bbox(bbox=bbox, tags=tags)
    gdf = gpd.GeoDataFrame(map_data)
    gdf = gdf.clip(box(bbox.west, bbox.south, bbox.east, bbox.north)).to_crs("EPSG:3857")
    raster = raster_fn(gdf, shape=(buffer, buffer))
    trans = lambda x: jnp.bool(x)  # jnp.rot90(x, 3)  # noqa
    terrain = trans(raster[0])
    return terrain


def raster_fn(gdf, shape) -> Array:
    bbox = gdf.total_bounds
    t = transform.from_bounds(*bbox, *shape)  # type: ignore
    raster = jnp.array([feature_fn(t, feature, gdf, shape) for feature in tags])
    return raster


def feature_fn(t, feature, gdf, shape):
    if feature not in gdf.columns:
        return jnp.zeros(shape)
    gdf = gdf[~gdf[feature].isna()]
    raster = features.rasterize(gdf.geometry, out_shape=shape, transform=t, fill=0)  # type: ignore
    return raster


# %%
if __name__ == "__main__":
    place = "Copenhagen, Denmark"
    bbox = get_bbox(place, 32)
    map_data = ox.features_from_bbox(bbox=bbox, tags=tags)
    map_data.plot()


# BBox = namedtuple("BBox", ["north", "south", "east", "west"])  # type: ignore

# def to_mercator(bbox: BBox) -> BBox:
# proj = ccrs.Mercator()
# west, south = proj.transform_point(bbox.west, bbox.south, ccrs.PlateCarree())
# east, north = proj.transform_point(bbox.east, bbox.north, ccrs.PlateCarree())
# return BBox(north=north, south=south, east=east, west=west)
#
#
# def to_platecarree(bbox: BBox) -> BBox:
# proj = ccrs.PlateCarree()
# west, south = proj.transform_point(bbox.west, bbox.south, ccrs.Mercator())
#
# east, north = proj.transform_point(bbox.east, bbox.north, ccrs.Mercator())
# return BBox(north=north, south=south, east=east, west=west)
#
