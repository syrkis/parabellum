# %%
import numpy as np
import jax.numpy as jnp
from parabellum import tps


# %%
def map_raster_from_line(raster, line, size):
    x0, y0, dx, dy = line
    x0 = int(x0*size)
    y0 = int(y0*size)
    dx = int(dx*size)
    dy = int(dy*size)
    max_T = int(2**0.5 * size)
    for t in range(max_T+1):
        alpha = t/float(max_T)
        x = x0 if dx == 0 else int((1 - alpha) * x0 + alpha * (x0+dx))
        y = y0 if dy == 0 else  int((1 - alpha) * y0 + alpha * (y0+dy))
        if 0<=x<size and 0<=y<size:
            raster[x, y] = 1
    return raster


# %%
def map_raster_from_rect(raster, rect, size):
    x0, y0, dx, dy = rect
    x0 = int(x0*size)
    y0 = int(y0*size)
    dx = int(dx*size)
    dy = int(dy*size)
    raster[x0:x0+dx, y0:y0+dy] = 1
    return raster


# %%
building_color = jnp.array([201,199,198, 255])
water_color = jnp.array([193, 237, 254, 255])
forest_color = jnp.array([197,214,185, 255])
empty_color = jnp.array([255, 255, 255, 255])

def make_terrain(terrain_args, size):
    args = {}
    for key, config in terrain_args.items():
        raster = np.zeros((size, size))
        if config is not None:
            for elem in config:
                if "line" in elem:
                    raster = map_raster_from_line(raster, elem["line"], size)
                elif "rect" in elem:
                    raster = map_raster_from_rect(raster, elem["rect"], size)
        args[key] = jnp.array(raster.T)
    basemap = jnp.where(args["building"][:,:,None], jnp.tile(building_color, (size, size, 1)), jnp.tile(empty_color, (size,size, 1)))
    basemap = jnp.where(args["water"][:,:,None], jnp.tile(water_color, (size, size, 1)), basemap)
    basemap = jnp.where(args["forest"][:,:,None], jnp.tile(forest_color, (size, size, 1)), basemap)
    args["basemap"] = basemap
    return tps.Terrain(**args)


# %%
db = {
    "blank": {'building': None, 'water': None, 'forest': None},
    "F": {'building': [{"line": [0.25, 0.33, 0.5, 0]}, {"line":[0.75, 0.33, 0., 0.25]}, {"line":[0.50, 0.33, 0., 0.25]}], 'water': None, 'forest': None},
    "stronghold": {'building': [
    {"line":[0.2, 0.275, 0.2, 0.]}, {"line":[0.2, 0.275, 0.0, 0.2]}, 
    {"line":[0.4, 0.275, 0.0, 0.2]}, {"line":[0.2, 0.475, 0.2, 0.]}, 

    {"line":[0.2, 0.525, 0.2, 0.]}, {"line": [0.2, 0.525, 0.0, 0.2]}, 
    {"line":[0.4, 0.525, 0.0, 0.2]}, {"line": [0.2, 0.725, 0.525, 0.]}, 

    {"line":[0.75, 0.25, 0., 0.2]}, {"line":[0.75, 0.55, 0., 0.19]},
    {"line":[0.6, 0.25, 0.15, 0.]}], 'water': None, 'forest': None},
    "playground": {'building': [{"line":[0.5, 0.5, 0.5, 0.]}], 'water': None, 'forest': None},
    "water_park": {
        'building': [{"line":[0.5, 0.5, 0.5, 0.]}], 
        "water": [{"rect":[0., 0.8, 0.1, 0.05]}, {"rect": [0.2, 0.8, 0.8, 0.05]}], 
        "forest": [{"rect": [0., 0., 1., 0.2]}]
    },
    "triangle":  {'building': [{"line": [0.33, 0., 0., 1.]}, {"line": [0.66, 0., 0., 1.]}], 'water': None, 'forest': None},
    "u_shape": {
        'building': [], 
        "water": [{"rect": [0.15, 0.2, 0.1, 0.5]}, {"rect": [0.4, 0.2, 0.1, 0.5]}, {"rect": [0.2, 0.2, 0.25, 0.1]}], 
        "forest": []
    },
}

# %% [raw]
#     import matplotlib.pyplot as plt
#     size = 50
#     raster = np.zeros((size, size))
#     rect = [0.2, 0.3, 0.05, 0.4]
#     raster = map_raster_from_rect(raster, rect, size)
#     rect = [0.4, 0.3, 0.05, 0.4]
#     raster = map_raster_from_rect(raster, rect, size)
#     rect = [0.2, 0.3, 0.25, 0.05]
#     raster = map_raster_from_rect(raster, rect, size)
#     rect = [0.2, 0.7, 0.25, 0.05]
#     raster = map_raster_from_rect(raster, rect, size)
#     rect = [0.6, 0.3, 0.4, 0.45]
#     raster = map_raster_from_rect(raster, rect, size)
#     plt.imshow(jnp.rot90(raster))

# %% [markdown]
# # Main

# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # %%
    terrain = make_terrain(db["u_shape"], size=50)

    # %%
    plt.imshow(jnp.rot90(terrain.basemap))

# %%

# %%
