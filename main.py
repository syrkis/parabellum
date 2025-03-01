# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
import jax.numpy as jnp
import numpy as np
from jax import random, lax
from PIL import Image
from einops import repeat
import parabellum as pb
from purejaxrl import ppo

# %% Config #################################################################
env = pb.env.Env(cfg=(cfg := pb.env.Conf()))
rng, key = random.split(random.PRNGKey(0))


# %% Functions ###############################################################
def step(state, rng):
    moving = random.normal(rng, (env.scene.num_agents, 2))
    action = pb.env.Action(health=None, moving=moving)
    obs, state = env.step(rng, state, action)
    return state, state


def anim(env, seq, scale=8, width=10):  # animate positions
    idxs = jnp.concat((jnp.arange(seq.shape[0]).repeat(seq.shape[1])[..., None], seq.reshape(-1, 2)), axis=1).T  #
    imgs = np.array(repeat(env.scene.terrain.building, "... -> t ...", t=seq.shape[0]).at[*idxs].set(0.5)) * 255.0
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs]  # type: ignore
    imgs[0].save("output.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)


# %% Main #####################################################################
obs, state = env.reset(key)
rngs = random.split(rng, 200)
print(dir(ppo))
# state, seq = lax.scan(step, state, rngs)
# anim(env, seq.unit_position.astype(int), width=env.cfg.size, scale=8)
#
