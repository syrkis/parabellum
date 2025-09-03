# %% utils.py
#   parabellum ut


# Imports
# import esch
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat
from jax import tree
from PIL import Image
# from parabellum.types import Config

# Twilight colors (used in neurocope)
red = "#EA344A"
blue = "#2B60F6"


# %% Plotting
def gif_fn(cfg, seq, scale=4):  # animate positions TODO: remove dead units
    pos = seq.pos.astype(int)
    cord = jnp.concat((jnp.arange(pos.shape[0]).repeat(pos.shape[1])[..., None], pos.reshape(-1, 2)), axis=1).T
    idxs = cord[:, seq.hp.flatten().astype(bool) > 0]
    imgs = 1 - np.array(repeat(cfg.map, "... -> a ...", a=len(pos)).at[*idxs].set(1))
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs * 255]  # type: ignore
    imgs[0].save("/Users/nobr/desk/s3/btc2sim/sims.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)


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
