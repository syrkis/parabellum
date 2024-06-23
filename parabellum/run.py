# run.py
#   parabellum run functions
# by: Noah Syrkis

# imports
import jax.numpy as jnp
import jax


# functions
def run_fn():
    from parabellum import Parabellum, scenarios

    scenario = scenarios["default"]
    env = Parabellum(scenario=scenario, map_width=32, map_height=32)


if __name__ == "__main__":
    run_fn()
