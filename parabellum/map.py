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
    return jnp.array(jnp.rot90(raster, 2)).astype(jnp.uint8)

def terrain_fn(place: str, size: int = 1000) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a rasterized map of buildings for a given location."""
    point = get_location(place)
    gdf = get_building_geometry(point, size)
    mask = rasterize_geometry(gdf, size)
    base = get_basemap(place, size)
    return mask, base

def get_basemap(place: str, size: int = 1000) -> jnp.ndarray:
    """Returns a basemap for a given place as a JAX array."""
    point = get_location(place)
    gdf = get_building_geometry(point, size)
    basemap, _ = cx.bounds2img(*gdf.total_bounds, ll=True)
    # get the middle size x size square
    basemap = basemap[(basemap.shape[0] - size) // 2:(basemap.shape[0] + size) // 2,
                        (basemap.shape[1] - size) // 2:(basemap.shape[1] + size) // 2]
    return jnp.array(jnp.rot90(basemap, 2)).astype(jnp.uint8)


if __name__ == "__main__":
    import seaborn as sns
    place = "Thun, Switzerland"
    mask, base = terrain_fn(place)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(mask) # type: ignore
    ax[1].imshow(base) # type: ignore
    plt.show()
