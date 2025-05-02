# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
import jax.numpy as jnp
import numpy as np
from jax import random, lax
import jax
from PIL import Image
import parabellum as pb
from einops import repeat
from omegaconf import OmegaConf
from jax_tqdm import scan_tqdm

jax.config.update("jax_debug_nans", True)


# %% Setup #################################################################
n_steps = 100
rng, key = random.split(random.PRNGKey(0))
cfg = OmegaConf.load("conf.yaml")
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)


# %% Functions ###############################################################
def action_fn(rng):
    coord = random.normal(rng, (env.num_units, 2))
    shoot = random.bernoulli(rng, 0.5, shape=(env.num_units,))
    return pb.types.Action(coord=coord, shoot=shoot)


@scan_tqdm(n_steps)
def step(state, inputs):
    idx, rng = inputs
    action = action_fn(rng)
    obs, state = env.step(rng, scene, state, action)
    return state, state


def anim(scene, seq, scale=2):  # animate positions TODO: remove dead units
    pos = seq.coords.astype(int)
    cord = jnp.concat((jnp.arange(pos.shape[0]).repeat(pos.shape[1])[..., None], pos.reshape(-1, 2)), axis=1).T
    idxs = cord[:, seq.health.flatten().astype(bool) > 0]
    imgs = np.array(repeat(scene.terrain.building, "... -> a ...", a=n_steps).at[*idxs].set(1)).astype(np.uint8) * 255
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs]  # type: ignore
    imgs[0].save("output.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)


# %% Main #####################################################################
obs, state = env.reset(key, scene)
rngs = random.split(rng, n_steps)
state, seq = lax.scan(step, state, (jnp.arange(n_steps), rngs))
anim(scene, seq, scale=8)


print(jnp.isnan(seq.coords).any())
