# ludens.py
#    script for fucking around and finding out
# by: Noah Syrkis

# %% Imports
import rasterio
from geopy.geocoders import Nominatim
import contextily as cx
import osmnx as ox

#
geolocator = Nominatim(user_agent="parabellum")
source = cx.providers.OpenStreetMap.Mapnik  # type: ignore
tags = {"building": True, "natural": "water"}


"""
def get_raster(
    place: str, meters: int = 1000, tags: List[dict] | dict = {"building": True}
) -> jnp.ndarray:
    # look here for tags https://wiki.openstreetmap.org/wiki/Map_features
    def aux(place, tag):
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
"""


place = "New York City"
point = geolocator.geocode(place)
# square kilometer around point
bounds = (
    point.longitude - 0.01,
    point.latitude - 0.01,
    point.longitude + 0.01,
    point.latitude + 0.01,
)
t = rasterio.transform.from_bounds(*bounds, 1000, 1000)  # type: ignore
gdf = ox.geometries_from_point(point, tags=tags, dist=500)
