# map.py
#   parabellum map functions
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from geopy.geocoders import Nominatim
import geopandas as gpd
import osmnx as ox
import rasterio
from jax import random
from rasterio import features
import rasterio.transform

# constants
geolocator = Nominatim(user_agent="parabellum")
tags = {"building": True}


# functions
def terrain_fn(place: str, size: int = 1000):
    """Returns a rasterized map of a given location."""

    # location info
    location = geolocator.geocode(place)
    coords = (location.latitude, location.longitude)

    # shape info
    geometry = ox.features_from_point(coords, tags=tags, dist=size // 2)
    gdf = gpd.GeoDataFrame(geometry).set_crs("EPSG:4326")

    # raster info
    t = rasterio.transform.from_bounds(*gdf.total_bounds, size, size)
    raster = features.rasterize(gdf.geometry, out_shape=(size, size), transform=t)

    # rotate 180 degrees
    raster = jnp.rot90(raster, 2)

    return jnp.array(raster).astype(jnp.uint8)


if __name__ == "__main__":
    import seaborn as sns

    place = "Vesterbro, Copenhagen, Denmark"
    terrain = terrain_fn(place)
    rng, key = random.split(random.PRNGKey(0))
    agents = spawn_fn(terrain, 12, 100, key)
    print(agents)
