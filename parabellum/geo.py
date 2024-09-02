# %% geo.py
#   script for geospatial level generation
# by: Noah Syrkis

# %% Imports
import rasterio
from geopy.geocoders import Nominatim
from geopy.distance import distance
import contextily as cx
import cartopy.crs as ccrs
from jaxtyping import Array
from shapely import box
import osmnx as ox
import geopandas as gpd
import numpy as np
from collections import namedtuple
import jax.numpy as jnp
import matplotlib.pyplot as plt
from rasterio import features, transform
import os
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Tuple

# %% Types
Coords = Tuple[float, float]
BBox = namedtuple("BBox", ["north", "south", "east", "west"])  # type: ignore

# %% Constants
provider = cx.providers.Stadia.StamenTerrainBackground(  # type: ignore
    api_key="86d0d32b-d2fe-49af-8db8-f7751f58e83f"
)
provider["url"] = provider["url"] + "?api_key={api_key}"
tags = {"building": True, "water": True, "landuse": "forest"}  #  "road": True}


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


# def raster_fn(bbox: BBox, shape=(1000, 1000)) -> Array:
#     buildings = ox.features_from_bbox(bbox=bbox, tags={"building": True})
#     gdf = gpd.GeoDataFrame(buildings).set_crs("EPSG:4326").to_crs("EPSG:3857")
#     t = rasterio.transform.from_bounds(*gdf.total_bounds, shape[0], shape[1])  # type: ignore
#     raster = features.rasterize(gdf.geometry, out_shape=shape, transform=t, fill=0)  # type: ignore
#     # img, ext = cx.bounds2img(*gdf.total_bounds, source=provider)
#     return jnp.array(raster)  # , jnp.array(img)


def basemap_fn(bbox: BBox) -> Array:
    buildings = ox.features_from_bbox(
        bbox=bbox,
        tags={
            "building": True,  # <- can't move through buildings
            # "water": True,  # <- can move slowly through and not shoot while in water
            # "landuse": "forest",  # <- can move through forest and hide in it
            "highway": True,  # <- can move quickly on roads
        },
    )
    gdf = gpd.GeoDataFrame(buildings).set_crs("EPSG:4326").to_crs("EPSG:3857")
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={"projection": ccrs.Mercator()})
    gdf.plot(ax=ax, color="black", alpha=1, edgecolor="black")
    cx.add_basemap(ax, crs=gdf.crs, source=provider, zoom="auto")
    image = jnp.array(fig.canvas.renderer._renderer)  # type: ignore
    # crop image

    plt.close(fig)
    return image

def geography_fn(bbox: BBox):
    map_data = ox.features_from_bbox(bbox=bbox, tags=tags)
    gdf = gpd.GeoDataFrame(map_data)
    # .clip to bbox
    gdf = gdf.clip(box(bbox.west, bbox.south, bbox.east, bbox.north))
    gdf = gdf.to_crs("EPSG:3857")
    # gdf = gdf.clip(box(mercator_bbox.west, mercator_bbox.south, mercator_bbox.east, mercator_bbox.north))
    raster = raster_fn(gdf)
    basemap = basemap_fn(bbox)
    return raster, basemap


def raster_fn(gdf, shape=(1000, 1000)) -> Array:
    bbox = gdf.total_bounds
    t = transform.from_bounds(*bbox, *shape)  # type: ignore
    raster = jnp.array([feature_fn(t, feature, gdf) for feature in ["building", "water", "landuse"]])
    # raster = features.rasterize(gdf.geometry, out_shape=shape, transform=t, fill=0)  # type: ignore
    return jnp.array(raster)

def feature_fn(t, feature, gdf):
    # subset where feature is not nan
    if feature not in gdf.columns:
        return jnp.zeros((1000, 1000))
    gdf = gdf[~gdf[feature].isna()]
    raster = features.rasterize(gdf.geometry, out_shape=(1000, 1000), transform=t, fill=0)  # type: ignore
    return raster

place = "Thun, Switzerland"
bbox = get_bbox(place, buffer=1000)
raster, basemap = geography_fn(bbox)
# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 20))
for i, ax in enumerate(axes[:-1]):
    ax.imshow(raster[i], cmap="gray")
    ax.axis("off")
axes[-1].imshow(raster.any(0), cmap="gray")
