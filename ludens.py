# ludens.py
#    script for fucking around and finding out
# by: Noah Syrkis

# %% Imports
import parabellum as pb
import jax.numpy as jnp
from jax import jit, random, vmap
import contextily as cx
import osmnx as ox
from geopy.geocoders import Nominatim
from functools import lru_cache, partial


# %% Hyper params
num_seeds = 100
num_areas = 10
num_units = 1000
num_meter = 100  # meters

# %% Geography
place = "New York, USA"
geolocator = Nominatim(user_agent="parabellum")
geocode = geolocator.geocode(place)
coords = geocode.longitude, geocode.latitude  # type: ignore
