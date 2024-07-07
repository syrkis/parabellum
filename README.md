# Parabellum

Ultra-scalable JaxMARL based warfare simulation engine developed with Armasuisse funding.

[![Documentation Status](https://readthedocs.org/projects/parabellum/badge/?version=latest)](https://parabellum.readthedocs.io/en/latest/?badge=latest)

## Features

- Obstacles and terrain integration
- Rasterized maps
- Blast radii simulation
- Friendly fire mechanics
- Pygame visualization
- JAX-based parallelization

## Install

```bash
pip install parabellum
```

## Quick Start

```python
import parabellum as pb
from jax import random

terrain = pb.terrain_fn("Thun, Switzerland", 1000)
scenario = pb.make_scenario("Thun", terrain, 10, 10)
env = pb.Parabellum(scenario)

rng, key = random.split(random.PRNGKey(0))
obs, state = env.reset(key)

# Simulation loop
for _ in range(100):
    rng, rng_act, key_step = random.split(key)
    key_act = random.split(rng_act, len(env.agents))
    act = {a: env.action_space(a).sample(k) for a, k in zip(env.agents, key_act)}
    obs, state, reward, done, info = env.step(key_step, act, state)

# Visualize
vis = pb.Visualizer(env, state_sequence)
vis.animate()
```

## Documentation

Full documentation: [parabellum.readthedocs.io](https://parabellum.readthedocs.io)

## Team

- Noah Syrkis
- Timothée Anne
- Supervisor: Sebastian Risi

## License

MIT