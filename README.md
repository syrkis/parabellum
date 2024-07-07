# Parabellum

## Ultra-Scalable Warfare Simulation Engine

Parabellum is an advanced, high-performance warfare simulation engine developed with funding from Armasuisse. Built upon JaxMARL's SMAX environment, Parabellum has been extensively modified to support a wide range of features crucial for realistic military simulations.

Version: 1.0.0

## Key Features

- Obstacles and terrain integration
- Rasterized maps for detailed environments
- Realistic blast radii simulation
- Friendly fire mechanics
- Pygame visualization for real-time monitoring
- Interactive Pygame interface for user engagement

## Purpose and Applications

Parabellum is designed for researchers, military strategists, and policy analysts who require sophisticated tools for modeling complex military scenarios. Its ultra-scalable architecture allows for simulations ranging from small-unit tactics to large-scale operations, providing valuable insights for training, strategy development, and policy analysis.

## Prerequisites

- Python 3.11 or higher
- JAX (version 0.3.25 or higher)
- PyGame (version 2.1.0 or higher)

## Installation

Install Parabellum using pip:

```bash
pip install parabellum
```

## Usage

Here's a basic example demonstrating how to use Parabellum to simulate a scenario with 10 agents and 10 enemies, each taking random actions:

```python
import parabellum as pb
from jax import random

# Define the scenario
terrain = pb.terrain_fn(place="Thun, Switzerland", size=1000)
scenario = pb.make_scenario(place, terrain, allies=10, enemies=10)
env = pb.Parabellum(scenario)  # Parabellum is the central class

# Initialize stochasticity
rng, key = random.split(random.PRNGKey(seed=0))
obs, state = env.reset(key)

state_sequence = []
for _ in range(n_steps := 100):
    # Manage stochasticity
    rng, rng_act, key_step = random.split(key)
    key_act = random.split(rng_act, len(env.agents))
    
    # Sample random actions
    act = {a: env.action_space(a).sample(k)
           for a, k in zip(env.agents, key_act)}
    
    # Store and step
    state_sequence.append((key_act, state, act))
    obs, state, reward, done, info = env.step(key_step, act, state)

# Visualize the state sequence
vis = pb.Visualizer(env, state_sequence)
vis.animate()
```

This script will generate an animation of the simulation, showcasing agent movements and interactions within the defined terrain.

## Advanced Usage

For more complex scenarios, custom terrain generation, or integration with AI models, please refer to our [detailed documentation](https://parabellum.readthedocs.io).

## Contributing

We welcome contributions to Parabellum! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to submit issues, feature requests, and code changes.

## License

Parabellum is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Ethical Considerations

As a warfare simulation tool, Parabellum has potential dual-use applications. We encourage users to consider the ethical implications of their research and to use this software responsibly. Parabellum is intended for defensive and strategic analysis purposes only.

## Acknowledgments

- Armasuisse for their funding and support
- The JaxMARL team for their foundational work on the SMAX environment
- Contributors and beta testers who have helped shape and improve Parabellum

## Performance

Parabellum leverages JAX's XLA compilation for significant performance gains. In benchmarks, it has demonstrated the ability to simulate scenarios with up to 10,000 agents in real-time on standard hardware, outperforming similar tools by an order of magnitude.

## Roadmap

- Parallel Pygame visualization enhancements
- Improved friendly fire mechanics
- Enhanced obstacle interaction for units
- Integration with machine learning frameworks for AI-driven simulations

For a full list of planned features and known issues, please see our [GitHub Issues](https://github.com/syrkis/parabellum/issues) page.

## Support

For bug reports, feature requests, or general questions, please open an issue on our [GitHub repository](https://github.com/syrkis/parabellum/issues).

For more detailed discussions or collaborations, contact the lead developer at noah@syrkis.com.

---

Parabellum: Because in preparations for peace, one must be ready for war.