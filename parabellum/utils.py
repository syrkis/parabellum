# %% utils.py
#   parabellum ut


# Imports
# import esch
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat
from jax import tree, lax
from jaxtyping import Array
from functools import partial
from PIL import Image
# from parabellum.types import Config

# Twilight colors (used in neurocope)
red = "#EA344A"
blue = "#2B60F6"


# %% Plotting
def quantize_fn(env, state) -> Array:  # for plotting and vision processing # TODO: add unit type color
    img = repeat(1 - env.map, "... -> ... 3") * 255
    col = jnp.where(env.teams[..., None] == 1, jnp.array((10, 90, 230)), jnp.array((230, 10, 90)))
    return img.at[*jnp.int32(state.pos).T].set(col)


def gif_fn(env, seq, fname, scale=4):  # animate positions TODO: remove dead units
    imgs = np.array(lax.map(partial(quantize_fn, env), seq), dtype=np.uint8)
    imgs = [Image.fromarray(e).resize(np.array(e.shape[:2]) * scale, Image.NEAREST) for e in imgs]  # type: ignore
    imgs[0].save(f"/Users/nobr/desk/s3/nebellum/{fname}.gif", save_all=True, append_images=imgs[1:], duration=24, loop=0)


# def svg_fn(cfg, seq, action, fname, targets=None, fps=2, debug=False):
#     # set up and background
#     e = esch.Drawing(h=cfg.size, w=cfg.size, row=1, col=seq.pos.shape[0], debug=debug, pad=10)
#     esch.grid_fn(e, repeat(np.array(cfg.map, dtype=float), f"... -> {seq.pos.shape[0]} ...") * 0.8, shape="square")

#     # loop thorugh teams
#     for i in jnp.unique(cfg.teams):  # c#fg.teams.unique():
#         col = "black" if i == 1 else "none"

#         # loop through types
#         for j in jnp.unique(cfg.types):
#             mask = (cfg.teams == i) & (cfg.types == j)
#             size, blast = float(cfg.rules.r[j]), float(cfg.rules.blast[j])
#             subset = np.array(rearrange(seq.pos, "a b c d -> a c d b"), dtype=float)[:, mask]
#             # print(tree.map(jnp.shape, action), mask.shape)
#             sub_action = tree.map(lambda x: x[:, :, mask], action)
#             # print(tree.map(jnp.shape, sub_action))
#             esch.sims_fn(e, subset, action=sub_action, fps=fps, col=col, stroke="black", size=size, blast=blast)

#             if debug:
#                 sight, reach = float(cfg.rules.sight[j]), float(cfg.rules.reach[j])
#                 esch.sims_fn(e, subset, action=None, col="none", fps=fps, size=reach, stroke="grey")
#                 esch.sims_fn(e, subset, action=None, col="none", fps=fps, size=sight, stroke="yellow")

#     if targets is not None:
#         pos = np.array(repeat(targets, f"... -> {seq.pos.shape[0]} ..."))
#         arr = np.ones(pos.shape[:-1])
#         esch.mesh_fn(e, pos, arr, shape="square", col="purple")

#     # save
#     e.dwg.saveas(fname)
