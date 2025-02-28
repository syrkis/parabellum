# tim.py
#   stuff that timothee wrote that i don't understand but use
# by: Noah Syrkis

# Imports
import jax.numpy as jnp
import equinox as eqx
import jax
from jax import vmap
from jax.scipy.signal import convolve


# Functions
@eqx.filter_jit
def compute_distance_map(cfg, terrain, starting_pos):
    kernel = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    valid = ~jnp.logical_or(terrain.building, terrain.water)
    distance_map = jnp.ones(valid.shape, dtype=jnp.uint32) * cfg.size**2
    current_cells = jnp.zeros(valid.shape, dtype=jnp.bool_)
    current_cells = current_cells.at[starting_pos[0], starting_pos[1]].set(True)

    def init_cond_fn(carry):
        _, current_cells, i = carry
        return jnp.all(
            jnp.where(current_cells, jnp.logical_and(current_cells, ~valid), True)
        )  # while all current are not valid (the target was invalid)

    def init_body_fn(carry):
        distance_map, current_cells, i = carry
        distance_map = jnp.where(current_cells, jnp.array([i], dtype=jnp.uint32), distance_map)
        current_cells = jnp.logical_xor(
            convolve(current_cells, kernel, mode="same"), current_cells
        )  # compute neighbors
        current_cells = jnp.logical_and(current_cells, distance_map == cfg.size**2)  # remove already visited
        return (distance_map, current_cells, i + 1)

    def main_cond_fn(carry):
        _, current_cells, i = carry
        return jnp.logical_and(i < cfg.size**2, jnp.any(current_cells))

    def main_body_fn(carry):
        distance_map, current_cells, i = carry
        distance_map = jnp.where(current_cells, jnp.array([i], dtype=jnp.uint32), distance_map)
        current_cells = jnp.logical_xor(
            convolve(current_cells, kernel, mode="same"), current_cells
        )  # compute neighbors
        current_cells = jnp.logical_and(current_cells, distance_map == cfg.size**2)  # remove already visited
        current_cells = jnp.logical_and(current_cells, valid)  # remove invalid
        return (distance_map, current_cells, i + 1)

    init_val = (distance_map, current_cells, 0)
    _, current_cells, i = jax.lax.while_loop(init_cond_fn, init_body_fn, init_val)
    init_val = (distance_map, jnp.logical_and(current_cells, valid), i)
    return jax.lax.while_loop(main_cond_fn, main_body_fn, init_val)[0]


def valid_sectors_fn(sectors: jnp.ndarray, invalid_spawn_areas: jnp.ndarray):
    """
    sectors must be of shape (num_units, 4) where sectors[i] = (x, y, width, height) of the ith unit's spawning sector (in % of the real map)
    """
    width, height = invalid_spawn_areas.shape

    def compute_valid_sector(sector):
        width, height = invalid_spawn_areas.shape
        coordx, coordy = jnp.array(sector[0] * width, dtype=jnp.int32), jnp.array(sector[1] * height, dtype=jnp.int32)
        valid_area = 1 - invalid_spawn_areas
        valid_area = jnp.where(jnp.arange(valid_area.shape[0]) >= coordx, valid_area.T, 0).T
        valid_area = jnp.where(
            jnp.arange(valid_area.shape[0]) <= coordx + jnp.ceil(sector[2] * width), valid_area.T, 0
        ).T
        valid_area = jnp.where(jnp.arange(valid_area.shape[1]) >= coordy, valid_area, 0)
        valid_area = jnp.where(jnp.arange(valid_area.shape[1]) <= coordy + jnp.ceil(sector[3] * height), valid_area, 0)
        return valid_area

    return vmap(compute_valid_sector)(sectors)
