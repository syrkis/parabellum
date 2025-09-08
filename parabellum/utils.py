# %% utils.py
#   parabellum ut


# Imports
# import esch
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat
from jax import tree, lax, vmap
from jaxtyping import Array
from functools import partial
from PIL import Image
# from parabellum.types import Config

# Twilight colors (used in neurocope)
red = "#EA344A"
blue = "#2B60F6"


def quantize_fn(env, state) -> Array:  # for plotting and vision processing # TODO: add unit type color
    hue = jnp.where(env.teams[..., None] == 0, jnp.array((0, 0, 255)), jnp.array((255, 0, 0)))
    col = jnp.int32(hue * (((env.types / (env.types.max() + 1)) * 0.5 + 0.5) * (state.hp > 0))[..., None])
    return (repeat(1 - env.map, "... -> ... 3") * 255).at[*jnp.int32(state.pos).T].set(col)


def gifs_fn(env, seqs):  # small multipls gif
    imgs = np.array(vmap(vmap(partial(quantize_fn, env)))(seqs), dtype=np.uint8)
    imgs = rearrange(imgs, "(a b) t s1 s2 col -> t (a s1) (b s2) col", a=int(env.cfg.r**0.5), b=int(env.cfg.r**0.5))
    return [Image.fromarray(e).resize(np.array(e.shape[:2]) * 256 // env.cfg.size, Image.NEAREST) for e in imgs]  # type: ignore


def gif_fn(env, seq, bg=True):  # animate positions TODO: remove dead units
    imgs = np.array(vmap(partial(quantize_fn, env))(seq), dtype=np.uint8)
    imgs = jnp.array([Image.fromarray(e).resize((1000, 1000), Image.NEAREST) for e in imgs])
    imgs = jnp.where(imgs == 255, env.raw, imgs)
    imgs = [Image.fromarray(e) for e in np.array(imgs)]  # type: ignore
    return imgs


# def gif_fn(env, seq, fname, scale=4):  # animate positions TODO: remove dead units
# imgs = np.array(lax.map(partial(quantize_fn, env), seq), dtype=np.uint8)
# imgs = [Image.fromarray(e).resize(np.array(e.shape[:2]) * scale, Image.NEAREST) for e in imgs]  # type: ignore
# imgs[0].save(f"/Users/nobr/desk/s3/nebellum/{fname}.gif", save_all=True, append_images=imgs[1:], duration=24, loop=0)
