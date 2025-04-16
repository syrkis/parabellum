#import "@preview/touying:0.6.1": *
#import "@preview/fletcher:0.5.7" as fletcher: diagram, node, edge
#import "@local/lilka:0.0.0": *
#show: lilka

#let title = "PARABELLUM"
#let subtitle = "A Combat Simulation Environment"
#let info = (author: "Noah Syrkis", date: datetime.today(), title: title)
#show: slides.with(config-info(..info), config-common(handout: false))
#metadata((title: title, slug: "parabellum-presentation"))

// Title slide
#title-slide()

// Introduction slide
= What is Parabellum?

Parabellum is a JAX-based combat simulation environment designed for:

- Multi-agent reinforcement learning
- Tactical simulations
- Combat system modeling
- Testing AI-driven decision making

#slide[
  - *Geography and Terrain*: Realistic environmental modeling
  - *Unit Types*: Different capabilities and parameters
  - *Combat Mechanics*: Damage, range, line of sight
  - *Multi-agent Framework*: Teams of allied and enemy units
  - *JAX Optimization*: Fast, differentiable simulation components
]

// Technical Details
= Technical Implementation

#slide[
  *Core Technologies:*
  - JAX for differentiable programming and XLA
  - Equinox for filtering JIT compilation
  - OpenStreetMap for terrain data
][
  *Key Features:*
  - Vectorized operations
  - Efficient state representation
  - Obstacle and visibility handling
  - Modular component design
]

// Simulation Flow
= Simulation Flow


#figure(
  fletcher.diagram(
    spacing: 4em,
    node-stroke: 2pt,
    // node-fill: gradient.radial(
    //   // blue.lighten(80%),
    //   blue,
    //   center: (30%, 20%),
    //   radius: 80%,
    // ),
    node((0, 0), [Initialize], radius: 2.5em),
    edge(label: "reset", "->"),
    node((1, 0), [Observe], radius: 2.5em),
    edge(label: "obs_fn", "->"),
    node((2, 0), [Act], radius: 2.5em),
    edge(label: "step_fn", "->"),
    node((3, 0), [Update], radius: 2.5em),
    edge(label: "Done?", "->", bend: -40deg),
    node((4, 0), [End], radius: 2.5em),
    // edge((3, 0), (1, 0), "Continue", "->", bend: 40deg),
  ),
)

// Reinforcement Learning
= Reinforcement Learning Applications

- *Policy Learning*: Training agents to make tactical decisions
- *Multi-agent Coordination*: Team-based strategy development
- *Adversarial Learning*: Red vs. blue team competitions
- *Scenario Generation*: Creating varied testing environments

// Example Use Cases
= Example Use Cases

- Military tactical simulations
- Game AI development
- Strategic planning systems
- Multi-agent coordination research
- Emergent behavior studies

// Future Directions
= Future Directions

- Enhanced terrain generation
- More sophisticated unit types and abilities
- Integration with larger simulation frameworks
- Performance optimizations
- Extended visualization capabilities

// Demo & Conclusion
= Conclusion

#text(size: 1.3em)[Thank you!]

- Repository: github.com/noahsyrkis/parabellum
- Documentation: parabellum.readthedocs.io
// - Contact: noah.syrkis@example.com

*Questions?*
