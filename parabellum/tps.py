# %%
# tps.py
#   parabellum types and dataclasses
# by: Noah Syrkis

# %% Imports
from chex import dataclass
from jaxtyping import Array


# %% Dataclasses
@dataclass
class Terrain:
    building: Array
    water: Array
    forest: Array
    basemap: Array
