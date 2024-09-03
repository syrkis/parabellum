# %%
import numpy as np
import jax.numpy as jnp
from parabellum import tps


# %%
def map_raster_map(lines, size):
    raster = np.zeros((size, size))
    for line in lines:
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
    return jnp.array(raster)


# %%
building_color = jnp.array([201,199,198, 255])
water_color = jnp.array([193, 237, 254, 255])
forest_color = jnp.array([197,214,185, 255])
empty_color = jnp.array([255, 255, 255, 255])

def make_terrain(terrain_args, size):
    args = {}
    for key, lines in terrain_args.items():
        args[key] = map_raster_map(lines, size)
    basemap = jnp.where(args["building"][:,:,None], jnp.tile(building_color, (size, size, 1)), jnp.tile(empty_color, (size,size, 1)))
    basemap = jnp.where(args["water"][:,:,None], jnp.tile(water_color, (size, size, 1)), basemap)
    basemap = jnp.where(args["forest"][:,:,None], jnp.tile(forest_color, (size, size, 1)), basemap)
    args["basemap"] = jnp.rot90(basemap, 2)
    return tps.Terrain(**args)


# %%
db = {
    "blank": {'building': [], 'water': [], 'forest': []},
    "F": {'building': [[0.25, 0.33, 0.5, 0], [0.75, 0.33, 0., 0.25], [0.50, 0.33, 0., 0.25]], 'water': [], 'forest': []},
    "stronghold": {'building': [
    [0.2, 0.275, 0.2, 0.], [0.2, 0.275, 0.0, 0.2], 
    [0.4, 0.275, 0.0, 0.2], [0.2, 0.475, 0.2, 0.], 

    [0.2, 0.525, 0.2, 0.], [0.2, 0.525, 0.0, 0.2], 
    [0.4, 0.525, 0.0, 0.2], [0.2, 0.725, 0.525, 0.], 

    [0.75, 0.25, 0., 0.2], [0.75, 0.55, 0., 0.19],
    [0.6, 0.25, 0.15, 0.], 
    ], 'water': [], 'forest': []},
    "playground": {'building': [[0.5, 0.5, 0.5, 0.]], 'water': [], 'forest': []},
    "water_park": {'building': [[0., 0.5, 0.33, 0.]], "water": [[0.33, 0.5, 0.33, 0.]], "forest": [[0.66, 0.5, 0.33, 0.]]},
}

# %% [markdown]
# # Main

# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # %%
    terrain = make_terrain(db["water_park"], size=100)
    plt.imshow(np.transpose(terrain.basemap, (1,0,2)))

# %%

    lines = [
        [0.66, 0.5, 0.33, 0.], 
    ]
    size = 50
    raster_map = map_raster_map(lines, size)
    plt.imshow(np.flip(raster_map, 0))

# %%
