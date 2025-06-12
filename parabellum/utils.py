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
from parabellum.types import Config

# Twilight colors (used in neurocope)
red = "#EA344A"
blue = "#2B60F6"


# %% Plotting
def gif_fn(cfg: Config, seq, scale=4):  # animate positions TODO: remove dead units
    pos = seq.coord.astype(int)
    cord = jnp.concat((jnp.arange(pos.shape[0]).repeat(pos.shape[1])[..., None], pos.reshape(-1, 2)), axis=1).T
    idxs = cord[:, seq.health.flatten().astype(bool) > 0]
    imgs = 1 - np.array(repeat(cfg.map, "... -> a ...", a=len(pos)).at[*idxs].set(1))
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs * 255]  # type: ignore
    imgs[0].save("/Users/syrkis/desk/s3/btc2sim/sim.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)


def svg_fn(cfg: Config, seq, action, fps=2):
    dwg = esch.init(cfg.size, cfg.size)
    esch.grid_fn(np.array(cfg.map).T, dwg, shape="square")
    arr = np.array(rearrange(seq.coord[:, :, ::-1], "time unit coord -> unit coord time"), dtype=np.float32)

    # add unit circles
    fill = [red if t == -1 else blue for t in cfg.teams]
    size = jnp.sqrt(cfg.types + 1).tolist()
    esch.anim_sims_fn(arr, dwg, fill=fill, size=size, fps=fps)

    # add range circles
    # reach = scene.unit_type_reach[scene.unit_types].tolist()
    # esch.anim_sims_fn(arr, dwg, size=reach, fill=["none" for _ in range(len(reach))], fps=fps)

    # start_shots =
    # print(tree.map(jnp.shape, action))
    time, unit = jnp.where(action.shoot)
    # debug.breakpoint()
    start_pos = seq.coord[time, unit][:, ::-1]
    end_pos = start_pos + action.coord[time, unit][:, ::-1]
    fill = [red if t == -1 else blue for t in cfg.teams[unit]]
    size = jnp.sqrt(cfg.rules.blast[cfg.types[unit]]).tolist()
    esch.anim_shot_fn(start_pos.tolist(), end_pos.tolist(), (time - 1).tolist(), dwg, color=fill, size=size, fps=fps)

    esch.save(dwg, "/Users/syrkis/desk/s3/btc2sim/sim.svg")


def svgs_fn(cfg: Config, seq):
    side = jnp.sqrt(seq.coord.shape[0]).astype(int).item()
    dwg = esch.init(cfg.size, cfg.size, side, side, line=True)
    for i in range(side):
        for j in range(side):
            sub_seq = tree.map(lambda x: x[i * side + j], seq)
            group = dwg.g()
            group.translate((cfg.size + 1) * i, (cfg.size + 1) * j)
            arr = np.array(rearrange(sub_seq.coord[:, :, ::-1], "t unit coord -> unit coord t"), dtype=np.float32)
            esch.grid_fn(np.array(cfg.map).T, dwg, group, shape="square")
            esch.anim_sims_fn(arr, dwg, group)
            dwg.add(group)
    esch.save(dwg, "/Users/syrkis/desk/s3/btc2sim/sims.svg")


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
    mask = jnp.stack([aux(*c) for c in jnp.stack((x, y)).T])  # type: ignore
    return mask.astype(jnp.int8).reshape(limit, limit, limit, limit)
