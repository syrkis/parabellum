# map.py
#   parabellum map functions
# by: Noah Syrkis

# imports
import jax.numpy as jnp
import jax


# functions
def map_fn(width, height, obst_coord, obst_delta):
    """Create a map from the given width, height, and obstacle coordinates and deltas."""
    m = jnp.zeros((width, height))
    for (x, y), (dx, dy) in zip(obst_coord, obst_delta):
        m = m.at[x : x + dx, y : y + dy].set(1)
    return m
