# run.py
#   parabellum run game live
# by: Noah Syrkis

# Noah Syrkis
import pygame
from jax import random
from functools import partial
import darkdetect
import jax.numpy as jnp
from chex import dataclass
import jaxmarl
from typing import Tuple, List, Dict, Optional
import parabellum as pb


# constants
fg = (255, 255, 255) if darkdetect.isDark() else (0, 0, 0)
bg = (0, 0, 0) if darkdetect.isDark() else (255, 255, 255)


# types
State = jaxmarl.environments.smax.smax_env.State
Obs = Reward = Done = Action = Dict[str, jnp.ndarray]
StateSeq = List[Tuple[jnp.ndarray, State, Action]]


@dataclass
class Control:
    running: bool = True
    paused: bool = False
    click: Optional[Tuple[int, int]] = None


@dataclass
class Game:
    clock: pygame.time.Clock
    state: State
    obs: Dict
    state_seq: StateSeq
    control: Control
    env: pb.Environment
    rng: random.PRNGKey


def handle_event(event, control_state):
    """Handle pygame events."""
    if event.type == pygame.QUIT:
        control_state.running = False
    if event.type == pygame.MOUSEBUTTONDOWN:
        pos = pygame.mouse.get_pos()
        control_state.click = pos
    if event.type == pygame.MOUSEBUTTONUP:
        control_state.click = None
    if event.type == pygame.KEYDOWN:  # any key press pauses
        control_state.paused = not control_state.paused
    return control_state


def control_fn(game):
    """Handle pygame events."""
    for event in pygame.event.get():
        game.control = handle_event(event, game.control)
    return game


def render_fn(screen, game):
    """Render the game."""
    if len(game.state_seq) < 3:
        return game
    for rng, state, action in env.expand_state_seq(game.state_seq[-2:])[-8:]:
        screen.fill(bg)
        if game.control.click is not None:
            pygame.draw.circle(screen, "red", game.control.click, 10)
        unit_positions = state.unit_positions
        for pos in unit_positions:
            pos = (pos / env.map_width * 800).tolist()
            pygame.draw.circle(screen, fg, pos, 5)
        pygame.display.flip()
        game.clock.tick(24)  # limits FPS to 24
    return game


def step_fn(game):
    """Step in parabellum."""
    rng, act_rng, step_key = random.split(game.rng, 3)
    act_key = random.split(act_rng, env.num_agents)
    action = {
        a: env.action_space(a).sample(act_key[i]) for i, a in enumerate(env.agents)
    }
    state_seq_entry = (step_key, game.state, action)
    # append state_seq_entry to state_seq
    game.state_seq.append(state_seq_entry)
    obs, state, reward, done, info = env.step(step_key, game.state, action)
    game.state = state
    game.obs = obs
    game.rng = rng
    return game


# state
if __name__ == "__main__":
    env = pb.Parabellum(pb.scenarios["default"])
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    render = partial(render_fn, screen)
    rng, key = random.split(random.PRNGKey(0))
    obs, state = env.reset(key)
    kwargs = dict(
        control=Control(),
        env=env,
        rng=rng,
        state_seq=[],  # [(key, state, action)]
        clock=pygame.time.Clock(),
        state=state,
        obs=obs,
    )
    game = Game(**kwargs)

    while game.control.running:
        game = control_fn(game)
        game = game if game.control.paused else step_fn(game)
        game = game if game.control.paused else render(game)

    pygame.quit()
