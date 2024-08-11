"""
This module provides functions to retrieve a mask and image of a given place on Earth.
Specifically, it provides functions to:
    1. Get coordinates for a given place (get_coords)
    2. Get building geometry for a given point and size (get_geometry)
    3. Get a raster mask (0 is background and 1 is building) of a given place (get_raster)
    4. Get a map image of a given place (get_image)
"""

# %% Imports
import numpy as np
import contextily as cx
from pyproj import Transformer
from typing import Tuple
import geopandas as gpd
from geopy.geocoders import Nominatim
import jax.numpy as jnp
from shapely.geometry import Point
import osmnx as ox
from PIL import Image
from rasterio import features
import rasterio


# %% Constants
BUILDING_TAGS = {"building": True}
geolocator = Nominatim(user_agent="parabellum")
provider = cx.providers.OpenStreetMap.Mapnik  # type: ignore
# cache dir is the parent directory of the current file (home file of pyproject.toml)
# dio not use __file__ as this might be run in repl


# %% Functions
def get_coords(place: str) -> Tuple[float, float]:
    """Get coordinates for a given place."""
    coords = geolocator.geocode(place)
    assert coords is not None, f"Could not geocode the place: {place}"
    return (coords.latitude, coords.longitude)  # type: ignore


def get_geometry(coord: Tuple[float, float], size: int) -> gpd.GeoDataFrame:
    """Get building geometry for a given point and size."""
    geometry = gpd.GeoDataFrame(
        geometry=[Point(coord[1], coord[0])], crs="EPSG:4326"
    ).buffer(size / 111320)  # Approximate degrees for the given pixel size
    return geometry


def get_raster(place: str, size: int) -> jnp.ndarray:
    """Rasterize geometry and return as a JAX array."""
    coord = get_coords(place)
    geom = ox.features_from_point(coord, tags=BUILDING_TAGS, dist=size // 2)
    gdf = gpd.GeoDataFrame(geom).set_crs("EPSG:4326")
    t = rasterio.transform.from_bounds(*gdf.total_bounds, size, size)  # type: ignore
    raster = features.rasterize(geom.geometry, out_shape=(size, size), transform=t)
    return jnp.array(raster)  # jnp.array(jnp.flip(raster, 0)).astype(jnp.uint8)


def get_image(place: str, size: int = 1000):
    """
    Get a map image of the given place as a numpy array, where each pixel covers exactly one meter.
    :param place: Name of the place to retrieve the map for
    :param size: Size of the image in pixels (default 1000)
    :return: Numpy array of shape (size, size, 3) representing the map image
    """
    lon, lat = get_coords(place)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(lon, lat)
    # Convert the center point to web mercator coordinates

    # Calculate the extent (500 meters in each direction from the center)
    extent = [x - 500, x + 500, y - 500, y + 500]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set the extent of the axis
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Add the basemap
    cx.add_basemap(ax, crs="EPSG:3857", source=provider, zoom=10)

    # Remove axis ticks and labels
    ax.set_axis_off()

    return fig


# %% Test the functions
import seaborn as sns
import matplotlib.pyplot as plt

place = "Copenhagen, Denmark"

# raster = get_raster(place, size=3000)
# sns.heatmap(1 - raster, cbar=False, square=True)
# plt.show()

# %% Test get_coords
fig = get_image(place, size=3000)
plt.show()
# plt.imshow(image)
