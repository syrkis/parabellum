# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
import jax.numpy as jnp
import numpy as np
from jax import random, lax, debug
from PIL import Image
import parabellum as pb
from einops import repeat, rearrange
from omegaconf import DictConfig
from jax_tqdm import scan_tqdm
import esch  # rwrfs


# %% Setup #################################################################
loc = dict(place="Tietgenkollegiet, Copenhagen, Denmark", size=100)
red = dict(plane=1, soldier=1)
blue = dict(plane=1, soldier=1)
cfg = DictConfig(dict(steps=100, knn=4, blue=blue, red=red) | loc)

# Access values using dot notation
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
rng, key = random.split(random.PRNGKey(0))


# %% Functions ###############################################################
def action_fn(rng):
    coord = random.normal(rng, (env.num_units, 2))
    shoot = random.bernoulli(rng, 0.5, shape=(env.num_units,))
    return pb.types.Action(coord=coord, shoot=shoot)


@scan_tqdm(cfg.steps)
def step(state, inputs):
    idx, rng = inputs
    action = action_fn(rng)
    obs, state = env.step(rng, scene, state, action)
    # debug.breakpoint()
    return state, state


def gif_fn(scene, seq, scale=2):  # animate positions TODO: remove dead units
    pos = seq.coords.astype(int)
    cord = jnp.concat((jnp.arange(pos.shape[0]).repeat(pos.shape[1])[..., None], pos.reshape(-1, 2)), axis=1).T
    idxs = cord[:, seq.health.flatten().astype(bool) > 0]
    imgs = np.array(repeat(scene.terrain.building, "... -> a ...", a=cfg.steps).at[*idxs].set(1)).astype(np.uint8) * 255
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs]  # type: ignore
    imgs[0].save("output.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)


def svg_fn(scene, seq):
    dwg = esch.init(100, 100)
    esch.grid_fn(np.array(scene.terrain.building).T, dwg)
    arr = np.array(rearrange(seq.coords, "time unit coord -> unit coord time"), dtype=np.float32)
    esch.anim_sims_fn(arr, dwg)
    esch.save(dwg, "test.svg")


# %% Main #####################################################################
obs, state = env.reset(key, scene)
rngs = random.split(rng, cfg.steps)
state, seq = lax.scan(step, state, (jnp.arange(cfg.steps), rngs))
svg_fn(scene, seq)
