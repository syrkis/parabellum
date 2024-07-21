# map.py
#   parabellum map functions
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from geopy.geocoders import Nominatim
import geopandas as gpd
import osmnx as ox
import asyncio
import geopy
import asyncio
import rasterio
from jax import random
from rasterio import features
import rasterio.transform
from typing import Optional, Tuple
from geopy.location import Location

# constants
geolocator = Nominatim(user_agent="parabellum")
tags = {"building": True}


# functions
def terrain_fn(place: str, size: int = 1000):
    """Returns a rasterized map of a given location."""

    # Get location info
    coords: Optional[Location] = geolocator.geocode(place)  # type: ignore

    if coords is None:
        raise ValueError(f"Could not geocode the place: {place}")

    # Convert coords to a tuple of (latitude, longitude)
    point = (coords.latitude, coords.longitude)

    # shape info
    geometry = ox.features_from_point(point, tags=tags, dist=size // 2)
    gdf = gpd.GeoDataFrame(geometry).set_crs("EPSG:4326")

    # raster info
    w, s, e, n = gdf.total_bounds
    t = rasterio.transform.from_bounds(w, s, e, n, size, size)
    raster = features.rasterize(gdf.geometry, out_shape=(size, size), transform=t)

    # rotate 180 degrees
    raster = jnp.rot90(raster, 2)

    return jnp.array(raster).astype(jnp.uint8)

if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt

    place = "Vesterbro, Copenhagen, Denmark"
    terrain = terrain_fn(place)
    rng, key = random.split(random.PRNGKey(0))
    sns.heatmap(terrain)
    plt.show()
    # agents = spawn_fn(terrain, 12, 100, key)
    # print(agents)
