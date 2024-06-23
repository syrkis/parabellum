# run.py
#   parabellum run functions for running interactive environments
# by: Noah Syrkis

# imports
import jax.numpy as jnp
import jax
import time
import pygame


def plot_frame(env, screen, state):
    positions = state.unit_positions / env.map_width * 640
    for position in positions:
        pygame.draw.circle(screen, (255, 0, 0), position.tolist(), 5)


# functions
def run_fn():
    from parabellum import Parabellum, scenarios

    scenario = scenarios["default"]
    env = Parabellum(scenario=scenario, map_width=32, map_height=32)
    rng, key = jax.random.split(jax.random.PRNGKey(0))
    obs, state = env.reset(key)
    pygame.init()

    screen = pygame.display.set_mode((640, 640))

    for i in range(10):
        # take random actions and show the environment
        actions = {a: env.action_space(a).sample(rng) for a in env.agents}
        obs, state, _, _, _ = env.step(rng, state, actions)
        plot_frame(env, screen, state)
        pygame.display.flip()
        pygame.time.wait(1000)
        time.sleep(0.1)

    # exit loop
    pygame.quit()


if __name__ == "__main__":
    run_fn()
