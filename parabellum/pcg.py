# pcg.py
#   procedural content generation
# by: Noah Syrkis

# %% Imports
from jax import random, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial


# %% Functions
seed = 0
n = 100
rng = random.PRNGKey(seed)
Y = random.uniform(rng, (n,))


def g(t):
    return (1 - jnp.cos(jnp.pi * t)) / 2


def lerp(a, b, t):
    t -= jnp.floor(t)  # the fractional part of t
    return (1 - t) * a + t * b


def cerp(a, b, t):
    t -= jnp.floor(t)  # the fractional part of t
    return g(1 - t) * a + g(t) * b


def body_fn(x):
    i = jnp.floor(x).astype(jnp.uint8)
    return cerp(Y[i], Y[i + 1], x)


@partial(vmap, in_axes=(None, 0, None))
def noise_fn(y, t, n):
    return y[t % n]


@vmap
def perlin_fn(t):
    return noise_fn(Y, t * jnp.arange(n * 3), n)


xs = jnp.linspace(0, 1, 1000)
noise = perlin_fn(2 ** jnp.arange(3)).sum(0)

fig, ax = plt.subplots(figsize=(20, 4), dpi=100)
ax.set_ylim(0, 1)
ax.plot(noise / noise.max())
