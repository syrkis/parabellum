# parabellum

Parabellum is an ultra-scalable, high-performance warfare simulation engine.
It is based on JaxMARL's SMAX environment, but has been heavily modified to
support a wide range of new features and improvements.

## Installation

Install through PyPI:

```bash
pip install parabellum
```

## Usage

```python
import parabellum as pb  # import the library
from jax import random  # random for stochasticity

env = pb.Parabellum()  # create the environment

rng, key = random.split(random.PRNGKey(0))  # create a random key
obs, state = env.reset(rng)  # get initial observation and state

state_sequence = []  # store the state sequence for later use (visualization, etc.)

for _ in range(1000):  # run the simulation for 1000 steps
    rng, rng_act, key_step = random.split(key)  # split the random key
    key_act = random.split(rng_act, len(obs.keys()))  # split the random key for each agent

    actions = {a: env.action_space(a).sample(k) for a, k in zip(obs.keys(), key_act)}  # sample actions
    state_sequence.append((key_act, state, actions))  # store the state sequence

    obs, reward, done, state = env.step(key_step, action, state)  # perform a step


vis = pb.Visualizer(env, state_sequence)  # create a visualizer
vis.animate()  # save the animation to a file
```

## TODO

- [ ] Parallel pygame vis
- [ ] Color for health?
- [ ] Add the ability to see ongoing game.
- [ ] Bug test friendly fire.
- [ ] Start sim from arbitrary state.
- [ ] Save when the episode ends in some state/obs variable
- [ ] Look for the source of the bug when using more Allies than Enemies
- [ ] Y inversed axis for parabellum visualization
- [ ] Units see through obstacles? 
