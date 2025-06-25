# %% geo.py
#   script for geospatial level generation
# by: Noah Syrkis

# %% Imports
from rasterio import features, transform

# from jax import tree
from geopy.geocoders import Nominatim
from geopy.distance import distance
import contextily as cx
import jax.numpy as jnp
import cartopy.crs as ccrs
from jaxtyping import Array
from shapely import box
import osmnx as ox
import geopandas as gpd
from collections import namedtuple
from typing import Tuple
import matplotlib.pyplot as plt
from cachier import cachier
# from jax.scipy.signal import convolve
# from parabellum.types import Terrain

# %% Types
Coords = Tuple[float, float]
BBox = namedtuple("BBox", ["north", "south", "east", "west"])  # type: ignore

# %% Constants
provider = cx.providers.Stadia.StamenTerrain(  # type: ignore
    api_key="86d0d32b-d2fe-49af-8db8-f7751f58e83f"
)
provider["url"] = provider["url"] + "?api_key={api_key}"
tags = {
    "building": True,
    "water": True,
    "highway": True,
    "landuse": [
        "grass",
        "forest",
        "flowerbed",
        "greenfield",
        "village_green",
        "recreation_ground",
    ],
    "leisure": "garden",
}  #  "road": True}


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
    return BBox(north, south, east, west)  # type: ignore


def basemap_fn(bbox: BBox, gdf) -> Array:
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={"projection": ccrs.Mercator()})
    gdf.plot(ax=ax, color="black", alpha=0, edgecolor="black")  # type: ignore
    cx.add_basemap(ax, crs=gdf.crs, source=provider, zoom="auto")  # type: ignore
    bbox = gdf.total_bounds
    ax.set_extent([bbox[0], bbox[2], bbox[1], bbox[3]], crs=ccrs.Mercator())  # type: ignore
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    image = jnp.array(fig.canvas.renderer._renderer)  # type: ignore
    plt.close(fig)
    return image


@cachier()
def geography_fn(place, buffer):
    bbox = get_bbox(place, buffer)
    map_data = ox.features_from_bbox(bbox=bbox, tags=tags)
    gdf = gpd.GeoDataFrame(map_data)
    gdf = gdf.clip(box(bbox.west, bbox.south, bbox.east, bbox.north)).to_crs("EPSG:3857")
    raster = raster_fn(gdf, shape=(buffer, buffer))
    # basemap = jnp.rot90(basemap_fn(bbox, gdf), 3)
    # kernel = jnp.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    trans = lambda x: jnp.bool(x)  # jnp.rot90(x, 3)  # noqa
    terrain = trans(raster[0])  # Terrain(
    #        building=trans(raster[0]),
    #        water=trans(raster[1] - convolve(raster[1] * raster[2], kernel, mode="same") > 0),
    #        forest=trans(jnp.logical_or(raster[3], raster[4])),
    #        basemap=basemap,
    #    )
    # terrain = tree.map(lambda x: x.astype(jnp.int16), terrain)
    return terrain


# =======
#     terrain = tps.Terrain(building=trans(raster[0] - convolve(raster[0]*raster[2], kernel, mode='same')>0),
#                           water=trans(raster[1] - convolve(raster[1]*raster[2], kernel, mode='same')>0),
#                           forest=trans(jnp.logical_or(raster[3], raster[4])),
#                           basemap=basemap)
#     return terrain, gdf
# >>>>>>> aeb13033e57083cc512a60f8f60a3db47a65ac32


def raster_fn(gdf, shape) -> Array:
    bbox = gdf.total_bounds
    t = transform.from_bounds(*bbox, *shape)  # type: ignore
    raster = jnp.array([feature_fn(t, feature, gdf, shape) for feature in tags])
    return raster


def feature_fn(t, feature, gdf, shape):
    if feature not in gdf.columns:
        return jnp.zeros(shape)
    gdf = gdf[~gdf[feature].isna()]
    raster = features.rasterize(gdf.geometry, out_shape=shape, transform=t, fill=0)  # type: ignore
    return raster


# %%
# def normalize(x):
#     return (np.array(x) - m) / (M - m)


# def get_bridges(gdf):
#     xmin, ymin, xmax, ymax = gdf.total_bounds
#     m = np.array([xmin, ymin])
#     M = np.array([xmax, ymax])

#     bridges = {}
#     for idx, bridge in gdf[gdf["bridge"] == "yes"].iterrows():
#         if type(bridge["name"]) == str:
#             bridges[idx[1]] = {
#                 "name": bridge["name"],
#                 "coords": normalize(
#                     [bridge.geometry.centroid.x, bridge.geometry.centroid.y]
#                 ),
#             }
#     return bridges


"""
# %%
if __name__ == "__main__":
    place = "Thun, Switzerland"
<<<<<<< HEAD
    terrain = geography_fn(place, 300)

=======
    terrain, gdf = geography_fn(place, 300)

>>>>>>> aeb13033e57083cc512a60f8f60a3db47a65ac32
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes[0].imshow(jnp.rot90(terrain.building), cmap="gray")
    axes[1].imshow(jnp.rot90(terrain.water), cmap="gray")
    axes[2].imshow(jnp.rot90(terrain.forest), cmap="gray")
    axes[3].imshow(jnp.rot90(terrain.building + terrain.water + terrain.forest))
    axes[4].imshow(jnp.rot90(terrain.basemap))

    # %%
    W, H, _  = terrain.basemap.shape
    bridges = get_bridges(gdf)

    # %%
    print("Bridges:")
    for bridge in bridges.values():
        x, y = int(bridge["coords"][0]*300), int(bridge["coords"][1]*300)
        print(bridge["name"], f"at ({x}, {y})")

    # %%
    plt.subplots(figsize=(7,7))
    plt.imshow(jnp.rot90(terrain.basemap))
    X = [b["coords"][0]*W for b in bridges.values()]
    Y = [(1-b["coords"][1])*H for b in bridges.values()]
    plt.scatter(X, Y)
    for i in range(len(X)):
        x,y = int(X[i]), int(Y[i])
        plt.text(x, y, str((int(x/W*300), int((1-(y/H))*300))))

# %%

# %% [raw]
# fig, ax = plt.subplots(figsize=(10, 10))
# gdf.plot(ax=ax, color='lightgray')  # Plot all features
# bridges.plot(ax=ax, color='red')     # Highlight bridges in red
# plt.show()

# %%

"""

# BBox = namedtuple("BBox", ["north", "south", "east", "west"])  # type: ignore


# def to_mercator(bbox: BBox) -> BBox:
# proj = ccrs.Mercator()
# west, south = proj.transform_point(bbox.west, bbox.south, ccrs.PlateCarree())
# east, north = proj.transform_point(bbox.east, bbox.north, ccrs.PlateCarree())
# return BBox(north=north, south=south, east=east, west=west)
#
#
# def to_platecarree(bbox: BBox) -> BBox:
# proj = ccrs.PlateCarree()
# west, south = proj.transform_point(bbox.west, bbox.south, ccrs.Mercator())
#
# east, north = proj.transform_point(bbox.east, bbox.north, ccrs.Mercator())
# return BBox(north=north, south=south, east=east, west=west)
#
