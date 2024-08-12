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
import os
import contextily as cx
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
provider = cx.providers.CartoDB.Positron  # type: ignore


# %% Functions
def get_coords(place: str) -> Tuple[float, float]:
    """Get coordinates for a given place."""
    coords = geolocator.geocode(place)
    assert coords is not None, f"Could not geocode the place: {place}"
    coords = (coords.longitude, coords.latitude)  # type: ignore
    return coords


def get_raster(place: str, size: int) -> jnp.ndarray:
    """Rasterize geometry and return as a JAX array."""
    coord = get_coords(place)
    geom = ox.features_from_point(coord, tags=BUILDING_TAGS, dist=size // 2)
    gdf = gpd.GeoDataFrame(geom).set_crs("EPSG:4326")
    t = rasterio.transform.from_bounds(*gdf.total_bounds, size, size)  # type: ignore
    raster = features.rasterize(geom.geometry, out_shape=(size, size), transform=t)
    return jnp.array(raster)  # jnp.array(jnp.flip(raster, 0)).astype(jnp.uint8)


def get_image(place: str, meters: int = 1000):
    """
    Get a map image of the given place as a numpy array, where each pixel covers exactly one meter.
    :param place: Name of the place to retrieve the map for
    :param size: Size of the image in pixels (default 1000)
    :return: Numpy array of shape (size, size, 3) representing the map image
    """
    lon, lat = get_coords(place)
    print(lon, lat)
    gdf = (
        gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
        .to_crs(epsg=3857)
        .buffer(meters)  # type: ignore  # TODO: confirm that we should indeed divide by 2
    )
    dpi = 300
    fig, ax = plt.subplots(figsize=(meters / dpi, meters / dpi), dpi=dpi)
    gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=0)
    # hide copywright
    cx.add_basemap(ax, source=provider, zoom="auto", attribution=False)
    ax.set_ylim([gdf.total_bounds[1], gdf.total_bounds[3]])  # type: ignore
    ax.set_xlim([gdf.total_bounds[0], gdf.total_bounds[2]])  # type: ignore
    # remove axis, padding, etc.
    plt.tight_layout()
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    random_tmp_name = hash(f"{place}_{meters}")
    plt.savefig(f"{random_tmp_name}.png", bbox_inches="tight", pad_inches=0)
    # convert to numpy array
    plt.close()
    image = np.array(Image.open(f"{random_tmp_name}.png"))
    os.remove(f"{random_tmp_name}.png")
    # remove alpha channel if present
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    return image


# %% Test the functions
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    place = "Vesterbro, Copenhagen, Denmark"

    # raster = get_raster(place, size=3000)
    # print(raster.shape)
    # sns.heatmap(1 - raster, cbar=False, square=True)
    # plt.show()

    # %% Test get_coords
    # image = get_image(place, 1000)
