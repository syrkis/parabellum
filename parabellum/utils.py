# %% utils.py


# Imports
from collections import namedtuple

import cartopy.crs as ccrs
import esch
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat
from jax import debug, tree
from PIL import Image

# Twilight colors (used in neurocope)
red = "#EA344A"
blue = "#2B60F6"


# %% Plotting
def gif_fn(scene, seq, scale=4):  # animate positions TODO: remove dead units
    pos = seq.coords.astype(int)
    cord = jnp.concat((jnp.arange(pos.shape[0]).repeat(pos.shape[1])[..., None], pos.reshape(-1, 2)), axis=1).T
    idxs = cord[:, seq.health.flatten().astype(bool) > 0]
    mask = scene.terrain.building  # .at[*jnp.int32(gps.marks.T)].set(1)
    imgs = 1 - np.array(repeat(mask, "... -> a ...", a=len(pos)).at[*idxs].set(1))
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs * 255]  # type: ignore
    imgs[0].save("/Users/nobr/desk/s3/btc2sim/sim.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)


def svg_fn(scene, seq, action):
    size = scene.terrain.building.shape[0]
    dwg = esch.init(size, size)
    esch.grid_fn(np.array(scene.terrain.building).T, dwg, shape="square")
    arr = np.array(rearrange(seq.coords[:, :, ::-1], "time unit coord -> unit coord time"), dtype=np.float32)

    # add unit circles
    fill = [red if t == -1 else blue for t in scene.unit_teams]
    size = jnp.sqrt(scene.unit_types + 1).tolist()
    esch.anim_sims_fn(arr, dwg, fill=fill, size=size)

    # add range circles
    reach = scene.unit_type_reach[scene.unit_types].tolist()
    esch.anim_sims_fn(arr, dwg, size=reach, fill=["none" for _ in range(len(reach))])

    # start_shots =
    # print(tree.map(jnp.shape, action))
    time, unit = jnp.where(action.shoot)
    # debug.breakpoint()
    start_pos = seq.coords[time, unit][:, ::-1]
    end_pos = start_pos + action.coord[time, unit][:, ::-1]
    fill = [red if t == -1 else blue for t in scene.unit_teams[unit]]
    size = jnp.sqrt(scene.unit_type_blast[scene.unit_types[unit]]).tolist()
    esch.anim_shot_fn(start_pos.tolist(), end_pos.tolist(), (time - 1).tolist(), dwg, color=fill, size=size)

    esch.save(dwg, "/Users/nobr/desk/s3/btc2sim/sim.svg")


def svgs_fn(scene, seq):
    size = scene.terrain.building.shape[0]
    side = jnp.sqrt(seq.coords.shape[0]).astype(int).item()
    dwg = esch.init(size, size, side, side, line=True)
    for i in range(side):
        for j in range(side):
            sub_seq = tree.map(lambda x: x[i * side + j], seq)
            group = dwg.g()
            group.translate((size + 1) * i, (size + 1) * j)
            arr = np.array(rearrange(sub_seq.coords[:, :, ::-1], "t unit coord -> unit coord t"), dtype=np.float32)
            esch.grid_fn(np.array(scene.terrain.building).T, dwg, group, shape="square")
            esch.anim_sims_fn(arr, dwg, group)
            dwg.add(group)
    esch.save(dwg, "/Users/nobr/desk/s3/btc2sim/sims.svg")


# Geography stuff
BBox = namedtuple("BBox", ["north", "south", "east", "west"])  # type: ignore


# coordinate function
def to_mercator(bbox: BBox) -> BBox:
    proj = ccrs.Mercator()
    west, south = proj.transform_point(bbox.west, bbox.south, ccrs.PlateCarree())
    east, north = proj.transform_point(bbox.east, bbox.north, ccrs.PlateCarree())
    return BBox(north=north, south=south, east=east, west=west)


def to_platecarree(bbox: BBox) -> BBox:
    proj = ccrs.PlateCarree()
    west, south = proj.transform_point(bbox.west, bbox.south, ccrs.Mercator())

    east, north = proj.transform_point(bbox.east, bbox.north, ccrs.Mercator())
    return BBox(north=north, south=south, east=east, west=west)


def obstacle_mask_fn(limit):
    def aux(i, j):
        xs = jnp.linspace(0, i + 1, i + j + 1)
        ys = jnp.linspace(0, j + 1, i + j + 1)
        cc = jnp.stack((xs, ys)).astype(jnp.int8)
        mask = jnp.zeros((limit, limit)).at[*cc].set(1)
        return mask

    x = jnp.repeat(jnp.arange(limit), limit)
    y = jnp.tile(jnp.arange(limit), limit)
    mask = jnp.stack([aux(*c) for c in jnp.stack((x, y)).T])
    return mask.astype(jnp.int8).reshape(limit, limit, limit, limit)
