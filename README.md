# parabellum

Parabellum is an ultra-scalable, high-performance warfare simulation engine, developped with funding from Armasuisse.
It is based on JaxMARL's SMAX environment, but has been heavily modified to
support a range of new features, including:
- Obstacles
- Rasterized maps
- Blast radii
- Friendly fire
- Pygame visualization
- Pygame interactive


## Installation

Parabellum is written in Python 3.11 and can be installed using pip:

```bash
pip install parabellum
```

## Usage

Parabellum is designed to be used in conjunction with JAX, a high-performance
numerical computing library. Here is a simple example of how to use Parabellum
to simulate a game with 10 agents and 10 enemies, each taking random actions:

```python
import parabellum as pb
from jax import random

# define the scenario
place = "Thun, Swizerland"
terrain = pb.terrain_fn(place, 1000)
scenario = pb.make_scenario(place, terrain, 10, 10)
env = pb.Parabellum(scenario)  # <- Parabellum is the central class of parabellum

# initiate stochasticity
rng, key = random.split(random.PRNGKey(seed := 0))
obs, state = env.reset(key)
state_sequence = []

for _ in range(n_steps := 100):

    # manage stochasticity
    rng, rng_act, key_step = random.split(key)
    key_act = random.split(rng_act, len(env.agents))

    # sample random actions
    act = {a: env.action_space(a).sample(k)
        for a, k in zip(env.agents, key_act)}

    # store and step
    state_sequence.append((key_act, state, act))
    obs, state, reward, done, info = env.step(key_step, act, state)


# save visualization of the state sequence
vis = pb.Visualizer(env, state_sequence)  # <- Visualizer is a nice to have class
vis.animate()
```

## TODO

- [x] Parallel pygame vis
    - [ ] Parallel bullet renderings
    - [ ] Combine parallell plots into one (maybe out of parabellum scope)
- [ ] Add skin to visualizer.
- [x] Add the ability to see ongoing game.
- [ ] Bug test friendly fire.
- [x] Start sim from arbitrary state.
- [ ] Save when the episode ends in some state/obs variable
- [ ] Look for the source of the bug when using more Allies than Enemies
- [x] Y inversed axis for parabellum visualization
- [ ] Units see through obstacles? 
