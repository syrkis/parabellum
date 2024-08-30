# %% geo.py
#   script for geospatial level generation
# by: Noah Syrkis

# %% Imports
import rasterio
from geopy.geocoders import Nominatim
from geopy.distance import distance
import contextily as cx
from contextily import Place
import cartopy.crs as ccrs
from jaxtyping import Array
import osmnx as ox
import geopandas as gpd
from collections import namedtuple
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
from rasterio import features
from typing import Tuple
import os

# %% Types
Coords = Tuple[float, float]
BBox = namedtuple("BBox", ["north", "south", "east", "west"])  # type: ignore

# %% Constants
provider = cx.providers.Esri.WorldImagery
# api_key="86d0d32b-d2fe-49af-8db8-f7751f58e83f"
# )
# provider["url"] = provider["url"] + "?api_key={api_key}"


# %% Coordinate function
def get_coordinates(place: str) -> Coords:
    geolocator = Nominatim(user_agent="parabellum")
    point = geolocator.geocode(place)
    return point.latitude, point.longitude  # type: ignore


# %% Bounding box function
def get_bbox(place: str, buffer: int = 1000) -> BBox:
    """Get bounding box from place name in crs 4326."""
    coords = get_coordinates(place)
    north = distance(meters=buffer).destination(coords, bearing=0).latitude
    south = distance(meters=buffer).destination(coords, bearing=180).latitude
    east = distance(meters=buffer).destination(coords, bearing=90).longitude
    west = distance(meters=buffer).destination(coords, bearing=270).longitude
    return BBox(north, south, east, west)


def raster_fn(bbox: BBox, shape=(1000, 1000)) -> Array:
    buildings = ox.features_from_bbox(bbox=bbox, tags={"building": True})
    gdf = gpd.GeoDataFrame(buildings).set_crs("EPSG:4326").to_crs("EPSG:3857")
    t = rasterio.transform.from_bounds(*gdf.total_bounds, shape[0], shape[1])  # type: ignore
    raster = features.rasterize(gdf.geometry, out_shape=shape, transform=t, fill=0)  # type: ignore
    # img, ext = cx.bounds2img(*gdf.total_bounds, source=provider)
    return jnp.array(raster)  # , jnp.array(img)


def basemap_fn(bbox: BBox) -> Array:
    buildings = ox.features_from_bbox(bbox=bbox, tags={"building": True})
    gdf = gpd.GeoDataFrame(buildings).set_crs("EPSG:4326").to_crs("EPSG:3857")
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={"projection": ccrs.Mercator()})
    gdf.plot(ax=ax, color="black", alpha=0.5, edgecolor="black")
    cx.add_basemap(ax, crs=gdf.crs, source=provider, zoom="auto")
    return jnp.array(fig.canvas.renderer._renderer)  # type: ignore


# raster = raster_fn(bbox)


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


# %% Ludens
# make cartopy map of bbox
place = "caen, france"
bbox = get_bbox(place, buffer=1000)
basemap = basemap_fn(bbox)


# %% Plot
plt.figure(figsize=(10, 10))
plt.imshow(basemap)
plt.tight_layout()
plt.axis("off")
plt.show()
