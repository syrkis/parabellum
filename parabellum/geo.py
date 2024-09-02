# %% geo.py
#   script for geospatial level generation
# by: Noah Syrkis

# %% Imports
from parabellum import tps
import rasterio
from rasterio import features, transform
from geopy.geocoders import Nominatim
from geopy.distance import distance
import contextily as cx
import jax.numpy as jnp
import cartopy.crs as ccrs
from jaxtyping import Array
import numpy as np
from shapely import box
import osmnx as ox
import geopandas as gpd
from collections import namedtuple
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os

# %% Types
Coords = Tuple[float, float]
BBox = namedtuple("BBox", ["north", "south", "east", "west"])  # type: ignore

# %% Constants
provider = cx.providers.Stadia.StamenTerrainBackground(  # type: ignore
    api_key="86d0d32b-d2fe-49af-8db8-f7751f58e83f"
)
provider["url"] = provider["url"] + "?api_key={api_key}"
tags = {"building": True, "water": True, "landuse": "forest"}  #  "road": True}


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
    return BBox(north, south, east, west)


def basemap_fn(bbox: BBox, gdf) -> Array:
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={"projection": ccrs.Mercator()})
    gdf.plot(ax=ax, color="black", alpha=1, edgecolor="black")  # type: ignore
    cx.add_basemap(ax, crs=gdf.crs, source=provider, zoom="auto") # type: ignore
    bbox = gdf.total_bounds
    ax.set_extent([bbox[0], bbox[2], bbox[1], bbox[3]], crs=ccrs.Mercator())
    plt.axis("off")
    plt.tight_layout()
    fig.canvas.draw()
    image = jnp.array(fig.canvas.renderer._renderer)  # type: ignore
    plt.close(fig)
    return image

def geography_fn(place, buffer):
    bbox = get_bbox(place, buffer)
    map_data = ox.features_from_bbox(bbox=bbox, tags=tags)
    gdf = gpd.GeoDataFrame(map_data)
    gdf = gdf.clip(box(bbox.west, bbox.south, bbox.east, bbox.north)).to_crs("EPSG:3857")
    raster = raster_fn(gdf, shape=(buffer, buffer))
    basemap = basemap_fn(bbox, gdf)
    terrain = tps.Terrain(land=raster[0], water=raster[1], forest=raster[2], basemap=basemap)
    return terrain


def raster_fn(gdf, shape) -> Array:
    bbox = gdf.total_bounds
    t = transform.from_bounds(*bbox, *shape)  # type: ignore
    raster = jnp.array([feature_fn(t, feature, gdf, shape) for feature in ["building", "water", "landuse"]])
    # terrain = tps.Terrain(land=raster[0], water=raster[1], forest=raster[2])
    return raster
    # return terrain

def feature_fn(t, feature, gdf, shape):
    # subset where feature is not nan
    if feature not in gdf.columns:
        return jnp.zeros(shape)
    gdf = gdf[~gdf[feature].isna()]
    raster = features.rasterize(gdf.geometry, out_shape=shape, transform=t, fill=0)  # type: ignore
    return raster

place = "Thun, Switzerland"
terrain = geography_fn(place, 200)
# %%
fig, axes = plt.subplots(1, 5, figsize=(20, 20))
axes[0].imshow(terrain.land, cmap="gray")
axes[1].imshow(terrain.water, cmap="gray")
axes[2].imshow(terrain.forest, cmap="gray")
axes[3].imshow(terrain.basemap)
axes[4].imshow(terrain.land + terrain.water + terrain.forest)
terrain.land.shape, terrain.water.shape, terrain.forest.shape, terrain.basemap.shape
