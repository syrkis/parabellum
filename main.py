# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
import jax.numpy as jnp
import numpy as np
from jax import random, lax
from PIL import Image
import parabellum as pb
from einops import repeat
from omegaconf import OmegaConf
from jax_tqdm import scan_tqdm


# %% Setup #################################################################
n_steps = 100
cfg = OmegaConf.load("conf.yaml")
rng, key = random.split(random.PRNGKey(0))
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)


# %% Functions ###############################################################
def action_fn(rng):
    coord = random.normal(rng, (env.num_units, 2))
    kinds = random.bernoulli(rng, 0.1, shape=(env.num_units,))
    return pb.types.Action(coord=coord, kinds=kinds)


@scan_tqdm(n_steps)
def step(state, inputs):
    idx, rng = inputs
    action = action_fn(rng)
    obs, state = env.step(rng, scene, state, action)
    return state, state


def anim(scene, seq, scale=4):  # animate positions
    idxs = jnp.concat((jnp.arange(seq.shape[0]).repeat(seq.shape[1])[..., None], seq.reshape(-1, 2)), axis=1).T
    imgs = np.array(repeat(scene.terrain.building, "... -> a ...", a=n_steps).at[*idxs].set(1)).astype(np.uint8) * 255
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs]  # type: ignore
    imgs[0].save("output.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)


# %% Main #####################################################################
obs, state = env.reset(key, scene)
rngs = random.split(rng, n_steps)
state, seq = lax.scan(step, state, (jnp.arange(n_steps), rngs))
anim(scene, seq.unit_position.astype(int))
