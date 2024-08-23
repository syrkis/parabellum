# ludens.py
#    script for fucking around and finding out
# by: Noah Syrkis


# %% Imports
# import parabellum as pb
import matplotlib.pyplot as plt
import osmnx as ox
from geopy.geocoders import Nominatim
import numpy as np
import contextily as cx
import jax.numpy as jnp
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import Point
from typing import List

# %% Constants
geolocator = Nominatim(user_agent="parabellum")
source = cx.providers.OpenStreetMap.Mapnik  # type: ignore


def get_raster(
    place: str, meters: int = 1000, tags: List[dict] | dict = {"building": True}
) -> jnp.ndarray:
    # look here for tags https://wiki.openstreetmap.org/wiki/Map_features
    def aux(place, tag):
        """Rasterize geometry and return as a JAX array."""
        place = geolocator.geocode(place)  # type: ignore
        point = place.latitude, place.longitude  # type: ignore  # confusing order of lat/lon
        geom = ox.features_from_point(point, tags=tag, dist=meters // 2)
        gdf = gpd.GeoDataFrame(geom).set_crs("EPSG:4326")
        # crop everythin outside of the meters x meters square
        gdf = gdf.cx[
            place.longitude - meters / 2 : place.longitude + meters / 2,
            place.latitude - meters / 2 : place.latitude + meters / 2,
        ]

        # bounds should be meters, meters
        t = rasterio.transform.from_bounds(*bounds, meters, meters)  # type: ignore
        raster = features.rasterize(
            gdf.geometry, out_shape=(meters, meters), transform=t
        )
        return jnp.array(raster)

    if isinstance(tags, dict):
        return aux(place, tags)
    else:
        return jnp.stack([aux(place, tag) for tag in tags])


def get_basemap(
    place: str, size: int = 1000
) -> np.ndarray:  # TODO: image is slightly off from raster. Fix this.
    # Create a GeoDataFrame with the center point
    place = geolocator.geocode(place)  # type: ignore
    lon, lat = place.longitude, place.latitude  # type: ignore
    gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3857")

    # Create a buffer around the center point
    # buffer = gdf.buffer(size)  # type: ignore
    buffer = gdf
    bounds = buffer.total_bounds  # i think this is wrong, since it ignores empty space
    # modify bounds to include empty space
    bounds = (bounds[0] - size, bounds[1] - size, bounds[2] + size, bounds[3] + size)

    # Create a figure and axis
    dpi = 300
    fig, ax = plt.subplots(figsize=(size / dpi, size / dpi), dpi=dpi)
    buffer.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=0)

    # Calculate the zoom level for the basemap

    # Add the basemap to the axis
    cx.add_basemap(ax, source=source, zoom="auto", attribution=False)

    # Set the x and y limits of the axis
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    # convert the image (without axis or border) to a numpy array
    plt.axis("off")
    plt.tight_layout()

    # remove whitespace
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return jnp.array(image)  # type: ignore
