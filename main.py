# %% main.py
#    parabellum main
# by: Noah Syrkis

# Imports ###################################################################
from jax import random, lax, vmap, profiler
import parabellum as pb
from jaxtyping import Array
from functools import partial
from typing import Tuple
from parabellum.types import Action, State, Config


# %% Functions ###############################################################
# @profiler.annotate_function
def action_fn(cfg: Config, rng: Array) -> Action:
    pos = random.normal(rng, (cfg.length, 2)) * cfg.rules.reach[cfg.types][..., None]
    kind = random.randint(rng, (cfg.length,), minval=0, maxval=3)
    return Action(pos=pos, kind=kind)


# @profiler.annotate_function
def step_fn(env: pb.env.Env, cfg: Config, state: State, rng: Array) -> Tuple[State, Tuple[State, Action]]:
    action = action_fn(cfg, rng)
    obs, state = env.step(rng, state, action)
    return state, (state, action)


# @profiler.annotate_function
def traj_fn(env, cfg, state, rng) -> Tuple[State, Tuple[State, Action]]:
    step = partial(step_fn, env, cfg)
    rngs = random.split(rng, cfg.steps)
    return lax.scan(step, state, rngs)


# %% Main #####################################################################
cfg = pb.types.Config()
env = pb.env.Env(cfg=cfg)
init_key, traj_key = random.split(random.PRNGKey(0), (2, cfg.sims))


with profiler.trace("./profiler_logs", create_perfetto_trace=True):
    obs, state = vmap(partial(env.reset, cfg))(init_key)
    state, seq = vmap(partial(traj_fn, env, cfg))(state, traj_key)


print("Profiling complete! Starting TensorBoard...")
print("Run: tensorboard --logdir=profiler_logs")
