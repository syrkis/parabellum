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
        args[key] = map_raster_map(lines, size).T
    basemap = jnp.where(args["building"][:,:,None], jnp.tile(building_color, (size, size, 1)), jnp.tile(empty_color, (size,size, 1)))
    basemap = jnp.where(args["water"][:,:,None], jnp.tile(water_color, (size, size, 1)), basemap)
    basemap = jnp.where(args["forest"][:,:,None], jnp.tile(forest_color, (size, size, 1)), basemap)
    args["basemap"] = basemap
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


# %% [raw]
#     lines = [
#         [0.66, 0.5, 0.33, 0.], 
#     ]
#     size = 50
#     raster_map = map_raster_map(lines, size)
#     plt.imshow(np.flip(raster_map, 0))

# %% [markdown]
# # Main

# %%
def compute_astar_map(mask, goal):
    """
    Start from goal on a grid world to compute the shortest path to the goal from any reachable cells 
    """
    n = len(mask)
    neighbors = [(-1, 0),  (0, -1), (1, 0), (0, 1)]  # N, E, S, W like in Parabellum
    current_x = [goal[0]]
    current_y = [goal[1]]
    cost = 0
    directions = np.ones(mask.shape) * -1
    costs = np.ones(mask.shape)*(n**2)
    costs[goal] = cost
    while len(current_x)>0:
        cost += 1
        new_x = np.empty(0, dtype=int)
        new_y = np.empty(0, dtype=int)
        
        for n_id, (i, j) in enumerate(neighbors):
            neighbors_x = np.array(current_x) + i
            neighbors_y = np.array(current_y) + j
            inside_mask = np.where(np.logical_and(np.logical_and(neighbors_x >= 0, neighbors_x < n), np.logical_and(neighbors_y >= 0, neighbors_y < n)))
            idxs = (neighbors_x[inside_mask], neighbors_y[inside_mask])
            valid = np.where(np.logical_and(costs[idxs] == n**2, mask[idxs]))
            valid_idx = inside_mask[0][valid[0]]
            idxs = (neighbors_x[valid_idx], neighbors_y[valid_idx])
            directions[idxs] = n_id
            costs[idxs] = cost
            
            new_x = np.concatenate([new_x, neighbors_x[valid_idx]])
            new_y = np.concatenate([new_y, neighbors_y[valid_idx]])
                    
        current_x, current_y = new_x, new_y
    return np.array(directions, dtype=int), np.array(costs, dtype=int)


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # %%
    terrain = make_terrain(db["F"], size=100)
    plt.imshow(terrain.building)

    # %%
    plt.imshow(terrain.basemap)
