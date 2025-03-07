# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
from flax.core import scan
import jax.numpy as jnp
import numpy as np
from jax import random, lax
from PIL import Image
import parabellum as pb
from omegaconf import OmegaConf
from jax_tqdm import scan_tqdm


# %% Setup #################################################################
cfg = OmegaConf.load("conf.yaml")
rng, key = random.split(random.PRNGKey(0))
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)


# %% Functions ###############################################################
def action_fn(rng):
    coord = random.normal(rng, (env.num_units, 2))
    kinds = random.bernoulli(rng, 0.5, shape=(env.num_units,))
    return pb.types.Action(coord=coord, kinds=kinds)


@scan_tqdm(200)
def step(state, inputs):
    idx, rng = inputs
    action = action_fn(rng)
    obs, state = env.step(rng, scene, state, action)
    return state, state


def anim(seq, scale=8, width=10):  # animate positions
    idxs = jnp.concat((jnp.arange(seq.shape[0]).repeat(seq.shape[1])[..., None], seq.reshape(-1, 2)), axis=1).T
    imgs = np.array(jnp.zeros((seq.shape[0], width, width)).at[*idxs].set(255)).astype(np.uint8)  # setting color
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs]  # type: ignore
    imgs[0].save("output.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)


# %% Main #####################################################################
obs, state = env.reset(key, scene)
rngs = random.split(rng, 200)
state, seq = lax.scan(step, state, (jnp.arange(200), rngs))
anim(seq.unit_position.astype(int), width=env.cfg.size)
#
