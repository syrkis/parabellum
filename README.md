# parabellum

Parabellum is an ultra-scalable, high-performance warfare simulation engine.
It is based on JaxMARL's SMAX environment, but has been heavily modified to
support a wide range of new features and improvements.

## Installation

`parabellum` is python3.11 based. Install it through PyPI:

```bash
pip install parabellum
```

## Usage

Parabellum is designed to be used in conjunction with JAX, a high-performance
numerical computing library. Here is a simple example of how to use Parabellum
to simulate a game with 10 agents and 10 enemies, each taking random actions.

```python
import parabellum as pb
from jax import random

# create the environment
kwargs = dict(map_width=256, map_height=256, num_agents=10, num_enemies=10)
env = pb.Parabellum(**kwargs)

# initiate stochasticity
rng = random.PRNGKey(0)
rng, key = random.split(rng)

# initialize the environment state
obs, state = env.reset(key)
state_sequence = []

for _ in range(1000):

    # manage stochasticity
    rng, rng_act, key_step = random.split(key)
    key_act = random.split(rng_act, len(env.agents))

    # sample actions and append to state sequence
    act = {a: env.action_space(a).sample(k)
        for a, k in zip(env.agents, key_act)}

    # step the environment
    state_sequence.append((key_act, state, act))
    obs, state, reward, done, info = env.step(key_step, act, state)


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
