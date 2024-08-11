"""
Visualizer for the Parabellum environment
"""

# Standard library imports
from typing import Optional, Tuple
import cv2

# JAX and JAX-related imports
import jax
from chex import dataclass
from jax import vmap, Array
import jax.numpy as jnp
from jaxmarl.viz.visualizer import SMAXVisualizer

# Third-party imports
import numpy as np
import pygame

# Local imports
import parabellum as pb


# skin dataclass
@dataclass
class Skin:
    # basemap: Array  # basemap of buildings
    maskmap: Array  # maskmap of buildings
    bg: Tuple[int, int, int] = (255, 255, 255)
    fg: Tuple[int, int, int] = (0, 0, 0)
    ally: Tuple[int, int, int] = (0, 255, 0)
    enemy: Tuple[int, int, int] = (255, 0, 0)
    pad: int = 100
    size: int = 1000  # excluding padding
    fps: int = 24
    vis_size: int = 1000  # size of the map in Vis (exluding padding)
    scale: Optional[float] = None


class Visualizer(SMAXVisualizer):
    def __init__(self, env: pb.Environment, state_seq, skin: Skin, reward_seq=None):
        super(Visualizer, self).__init__(env, state_seq, reward_seq)

        # self.bullet_seq = vmap(partial(bullet_fn, self.env))(self.state_seq)
        self.action_seq = [action for _, _, action in state_seq]  # bcs SMAX bug
        self.state_seq = state_seq
        self.image = image_fn(skin)
        self.skin = skin
        self.skin.scale = self.skin.size / env.map_width  # assumes square map
        self.env = env

    def animate(self, save_fname: Optional[str] = "output/parabellum", view=None):
        expanded_state_seq, expanded_action_seq = expand_fn(
            self.env, self.state_seq, self.action_seq
        )
        state_seq_seq, action_seq_seq = unbatch_fn(
            expanded_state_seq, expanded_action_seq
        )
        for idx, (state_seq, action_seq) in enumerate(
            zip(state_seq_seq, action_seq_seq)
        ):
            animate_fn(
                self.env,
                self.skin,
                self.image,
                state_seq,
                action_seq,
                f"{save_fname}_{idx}.mp4",
            )


# functions
def animate_fn(env, skin, image, state_seq, action_seq, save_fname):
    pygame.init()
    frames = []
    for idx, (state_tup, action) in enumerate(zip(state_seq, action_seq)):
        frames += [frame_fn(env, skin, image, state_tup[1], action, idx)]
    # use cv2 to write frames to video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(
        save_fname,
        fourcc,
        skin.fps,
        (skin.size + skin.pad * 2, skin.size + skin.pad * 2),
    )
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    pygame.quit()


def init_frame(
    env, skin, image, state: pb.State, action: Array, idx: int
) -> pygame.Surface:
    dims = (skin.size + skin.pad * 2, skin.size + skin.pad * 2)
    frame = pygame.Surface(dims, pygame.SRCALPHA | pygame.HWSURFACE)
    return frame


def transform_frame(env, skin, frame):
    # frame = np.rot90(pygame.surfarray.pixels3d(frame).swapaxes(0, 1), 2)
    frame = np.flip(pygame.surfarray.pixels3d(frame).swapaxes(0, 1), 0)
    return frame


def frame_fn(env, skin, image, state: pb.State, action: Array, idx: int) -> np.ndarray:
    """Create a frame"""
    frame = init_frame(env, skin, image, state, action, idx)

    pipeline = [render_background, render_agents, render_action, render_bullet]
    for fn in pipeline:
        frame = fn(env, skin, image, frame, state, action)

    return transform_frame(env, skin, frame)


def render_background(env, skin, image, frame, state, action):
    coords = (skin.pad - 5, skin.pad - 5, skin.size + 10, skin.size + 10)
    frame.fill(skin.bg)
    frame.blit(image, coords)
    pygame.draw.rect(frame, skin.fg, coords, 3)
    return frame


def render_action(env, skin, image, frame, state, action):
    return frame


def render_bullet(env, skin, image, frame, state, action):
    return frame


def render_agents(env, skin, image, frame, state, action):
    units = state.unit_positions, state.unit_teams, state.unit_types, state.unit_health
    for idx, (pos, team, kind, health) in enumerate(zip(*units)):
        pos = tuple((pos * skin.scale).astype(int) + skin.pad)
        # draw the agent
        if health > 0:
            unit_size = env.unit_type_radiuses[kind]
            radius = float(jnp.ceil((unit_size * skin.scale)).astype(int) + 1)
            pygame.draw.circle(frame, skin.fg, pos, radius, 1)
            pygame.draw.circle(frame, skin.bg, pos, radius + 1, 1)
    return frame


def text_fn(text):
    """rotate text upside down because of pygame issue"""
    return pygame.transform.rotate(text, 180)


def image_fn(skin: Skin):  # TODO:
    """Create an image for background (basemap or maskmap)"""
    motif = cv2.resize(
        np.array(skin.maskmap.T),
        (skin.size, skin.size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint8)
    motif = (motif > 0).astype(np.uint8)
    image = np.zeros((skin.size, skin.size, 3), dtype=np.uint8) + skin.bg
    image[motif == 1] = skin.fg
    image = pygame.surfarray.make_surface(image)
    image = pygame.transform.scale(image, (skin.size, skin.size))
    return image


def unbatch_fn(state_seq, action_seq):
    """state seq is a list of tuples of (step_key, state, actions)."""
    if is_multi_run(state_seq):
        n_envs = state_seq[0][1].unit_positions.shape[0]
        state_seq_seq = [jax.tree_map(lambda x: x[i], state_seq) for i in range(n_envs)]
        action_seq_seq = [
            jax.tree_map(lambda x: x[i], action_seq) for i in range(n_envs)
        ]
    else:
        state_seq_seq = [state_seq]
        action_seq_seq = [action_seq]
    return state_seq_seq, action_seq_seq


def expand_fn(env, state_seq, action_seq):
    """Expand the state sequence"""
    fn = env.expand_state_seq
    state_seq = vmap(fn)(state_seq) if is_multi_run(state_seq) else fn(state_seq)
    action_seq = [
        action_seq[i // env.world_steps_per_env_step] for i in range(len(state_seq))
    ]
    return state_seq, action_seq


def is_multi_run(state_seq):
    return state_seq[0][1].unit_positions.ndim > 2
