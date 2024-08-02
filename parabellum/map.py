# map.py
    # parabellum map functions
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from geopy.geocoders import Nominatim
import geopandas as gpd
import osmnx as ox
import contextily as cx
import matplotlib.pyplot as plt
from rasterio import features
import rasterio.transform
from typing import Optional, Tuple
from geopy.location import Location
from shapely.geometry import Point
import os
import pickle 

# constants
geolocator = Nominatim(user_agent="parabellum")
BUILDING_TAGS = {"building": True}

def get_location(place: str) -> Tuple[float, float]:
    """Get coordinates for a given place."""
    coords: Optional[Location] = geolocator.geocode(place)  # type: ignore
    if coords is None:
        raise ValueError(f"Could not geocode the place: {place}")
    return (coords.latitude, coords.longitude)

def get_building_geometry(point: Tuple[float, float], size: int) -> gpd.GeoDataFrame:
    """Get building geometry for a given point and size."""
    geometry = ox.features_from_point(point, tags=BUILDING_TAGS, dist=size // 2)
    return gpd.GeoDataFrame(geometry).set_crs("EPSG:4326")

def rasterize_geometry(gdf: gpd.GeoDataFrame, size: int) -> jnp.ndarray:
    """Rasterize geometry and return as a JAX array."""
    w, s, e, n = gdf.total_bounds
    transform = rasterio.transform.from_bounds(w, s, e, n, size, size)
    raster = features.rasterize(gdf.geometry, out_shape=(size, size), transform=transform)
    return jnp.array(jnp.flip(raster, 0) ).astype(jnp.uint8)

# +
def get_from_cache(place, size):
    if os.path.exists("./cache"):
        name = str(hash((place, size))) + ".pk"
        if os.path.exists("./cache/" + name):
            with open("./cache/" + name, "rb") as f:
                (mask, base) = pickle.load(f)
            return (mask, base.astype(jnp.int64))
    return (None, None)
    
def save_in_cache(place, size, mask, base):
    if not os.path.exists("./cache"):
        os.makedirs("./cache")
    name = str(hash((place, size))) + ".pk"
    with open("./cache/" + name, "wb") as f:
        pickle.dump((mask, base), f)

def terrain_fn(place: str, size: int = 1000, with_cache: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a rasterized map of buildings for a given location."""
    if with_cache:
        mask, base = get_from_cache(place, size)
    if not with_cache or mask is None:
        point = get_location(place)
        gdf = get_building_geometry(point, size)
        mask = rasterize_geometry(gdf, size)
        base = get_basemap(place, size)
        if with_cache:
            save_in_cache(place, size, mask, base)
    return mask, base


# -

def get_basemap(place: str, size: int = 1000) -> jnp.ndarray:
    """Returns a basemap for a given place as a JAX array."""
    point = get_location(place)
    gdf = get_building_geometry(point, size)
    basemap, _ = cx.bounds2img(*gdf.total_bounds, ll=True)
    # get the middle size x size square
    basemap = basemap[(basemap.shape[0] - size) // 2:(basemap.shape[0] + size) // 2,
                        (basemap.shape[1] - size) // 2:(basemap.shape[1] + size) // 2]
    return basemap # jnp.array(jnp.rot90(basemap, 2)).astype(jnp.uint8)


if __name__ == "__main__":
    place = "Cauvicourt, 14190, France"
    mask, base = terrain_fn(place, 500)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(jnp.flip(mask,0)) # type: ignore
    ax[1].imshow(base) # type: ignore
    ax[2].imshow(base) # type: ignore
    ax[2].imshow(jnp.flip(mask,0), alpha=jnp.flip(mask,0)) # type: ignore
    plt.show()




