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
import parabellum as pb
from jax import random

# create the environment
env = pb.Parabellum()

# initiate stochasticity
rng = random.PRNGKey(0)
rng, key = random.split(rng)

# initialize the environment state
obs, state = env.reset(key)
state_sequence = []

for _ in range(1000):

    # manage stochasticity
    rng, rng_act, key_step = random.split(key)
    key_act = random.split(rng_act, len(obs.keys()))

    # sample actions and append to state sequence
    actions = {a: env.action_space(a).sample(k) for a, k in zip(obs.keys(), key_act)}
    state_sequence.append((key_act, state, actions))

    # step the environment
    obs, reward, done, state = env.step(key_step, action, state)


# save visualization of the state sequence
vis = pb.Visualizer(env, state_sequence)
vis.animate()
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
