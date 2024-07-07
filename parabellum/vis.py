"""
Visualizer for the Parabellum environment
"""

from tqdm import tqdm
import jax.numpy as jnp
import jax
from jax import vmap
from jax import tree_util
from functools import partial
import darkdetect
import numpy as np
import pygame
import matplotlib.pyplot as plt
import os
from moviepy.editor import ImageSequenceClip
from typing import Optional
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.viz.visualizer import SMAXVisualizer

# default dict
from collections import defaultdict


# constants
action_to_symbol = {0: "↑", 1: "→", 2: "↓", 3: "←", 4: "Ø"}


def small_multiples():
    # make video of small multiples based on all videos in output
    video_files = [f"output/parabellum_{i}.mp4" for i in range(4)]
    # load mp4 videos and make a grid
    clips = [ImageSequenceClip.load(filename) for filename in video_files]
    print(len(clips))


def text_fn(text):
    """rotate text upside down because of pygame issue"""
    return pygame.transform.rotate(text, 180)


class Visualizer(SMAXVisualizer):
    def __init__(self, env: MultiAgentEnv, state_seq, reward_seq=None, skin=None):
        self.fig, self.ax = None, None
        super().__init__(env, state_seq, reward_seq)
        # clear the figure made by SMAXVisualizer
        plt.close()
        self.bg = (0, 0, 0) if darkdetect.isDark() else (255, 255, 255)
        self.fg = (255, 255, 255) if darkdetect.isDark() else (0, 0, 0)
        self.pad = 75
        # TODO: make sure it's always a 1024x1024 image
        self.width = 1000
        self.s = self.width + self.pad + self.pad
        self.scale = self.width / env.map_width
        self.action_seq = [action for _, _, action in state_seq]  # bcs SMAX bug
        # self.bullet_seq = vmap(partial(bullet_fn, self.env))(self.state_seq)

    def animate(self, save_fname: str = "output/parabellum.mp4"):
        multi_dim = self.state_seq[0][1].unit_positions.ndim > 2
        if multi_dim:
            n_envs = self.state_seq[0][1].unit_positions.shape[0]
            if not self.have_expanded:
                state_seqs = vmap(self.env.expand_state_seq)(self.state_seq)
                self.have_expanded = True
            for i in range(n_envs):
                state_seq = jax.tree_map(lambda x: x[i], state_seqs)
                action_seq = jax.tree_map(lambda x: x[i], self.action_seq)
                self.animate_one(
                    state_seq, action_seq, save_fname.replace(".mp4", f"_{i}.mp4")
                )
        else:
            state_seq = self.env.expand_state_seq(self.state_seq)
            self.animate_one(state_seq, self.action_seq, save_fname)

    def animate_one(self, state_seq, action_seq, save_fname):
        frames = []  # frames for the video
        pygame.init()  # initialize pygame
        terrain = np.array(self.env.terrain_raster.T)
        rgb_array = np.zeros((terrain.shape[0], terrain.shape[1], 3), dtype=np.uint8)
        if darkdetect.isLight():
            rgb_array += 255
        rgb_array[terrain == 1] = self.fg
        mask_surface = pygame.surfarray.make_surface(rgb_array)
        mask_surface = pygame.transform.scale(mask_surface, (self.width, self.width))

        for idx, (_, state, _) in tqdm(enumerate(state_seq), total=len(self.state_seq)):
            action = action_seq[idx // self.env.world_steps_per_env_step]
            screen = pygame.Surface(
                (self.s, self.s), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            screen.fill(self.bg)  # fill the screen with the background color
            screen.blit(
                mask_surface,
                (self.pad, self.pad, self.width, self.width),
            )
            # add env.scenario.place to the title (in top padding)
            font = pygame.font.SysFont("Fira Code", 18)
            width = self.env.map_width
            title = f"{width}x{width}m in {self.env.scenario.place}"
            text = text_fn(font.render(title, True, self.fg))
            # center the text
            screen.blit(text, (self.s // 2 - text.get_width() // 2, self.pad // 4))
            # draw edge around terrain
            pygame.draw.rect(
                screen,
                self.fg,
                (
                    self.pad - 2,
                    self.pad - 2,
                    self.width + 4,
                    self.width + 4,
                ),
                2,
            )

            self.render_agents(screen, state)  # render the agents
            self.render_action(screen, action)
            # self.render_obstacles(screen)  # render the obstacles

            # bullets
            """ if idx < len(self.bullet_seq) * 8:
                bullets = self.bullet_seq[idx // 8]
                self.render_bullets(screen, bullets, idx % 8) """

            # rotate the screen and append to frames
            pixels = pygame.surfarray.pixels3d(screen).swapaxes(0, 1)
            # rotate the screen 180 degrees (transpose and flip)
            pixels = np.rot90(pixels, 2)  # pygame starts in bottom left
            frames.append(pixels)
        # save the images
        clip = ImageSequenceClip(frames, fps=48)
        clip.write_videofile(save_fname, fps=48)
        clip.write_gif(save_fname.replace(".mp4", ".gif"), fps=24)
        pygame.quit()

    def render_agents(self, screen, state):
        time_tuple = zip(
            state.unit_positions,
            state.unit_teams,
            state.unit_types,
            state.unit_health,
        )
        for idx, (pos, team, kind, hp) in enumerate(time_tuple):
            face_col = self.fg if int(team.item()) == 0 else self.bg
            pos = tuple(((pos * self.scale) + self.pad).tolist())
            # draw the agent
            if hp > 0:
                hp_frac = hp / self.env.unit_type_health[kind]
                unit_size = self.env.unit_type_radiuses[kind]
                radius = jnp.ceil((unit_size * self.scale * hp_frac)).astype(int) + 1
                pygame.draw.circle(screen, face_col, pos, radius)
                pygame.draw.circle(screen, self.fg, pos, radius, 1)

            # draw the sight range
            # sight_range = self.env.unit_type_sight_ranges[kind] * self.scale
            # pygame.draw.circle(screen, self.fg, pos, sight_range.astype(int), 2)

            # draw attack range
            # attack_range = self.env.unit_type_attack_ranges[kind] * self.scale
            # pygame.draw.circle(screen, self.fg, pos, attack_range.astype(int), 2)
            # work out which agents are being shot

    def render_action(self, screen, action):
        if self.env.action_type != "discrete":
            return

        def coord_fn(idx, n, team, text):
            text_adj = text.get_width() / 2
            is_ally = team == "ally"
            return (
                # vertically centered so that n / 2 is above and below the center
                self.pad + self.width + self.pad / 2 - text_adj
                if is_ally
                else self.pad / 2 - text_adj,
                self.s / 2 - (n / 2) * self.s / 20 + idx * self.s / 20,
            )

        for team, number in [("ally", 0), ("enemy", 1)]:
            for idx in range(self.env.num_allies):
                symb = action_to_symbol.get(
                    action[f"{team}_{idx}"].astype(int).item(), "Ø"
                )
                font = pygame.font.SysFont(
                    "Fira Code", jnp.sqrt(self.s).astype(int).item()
                )
                text = text_fn(font.render(symb, True, self.fg))
                coord = coord_fn(idx, self.env.num_allies, team, text)
                screen.blit(text, coord)

    def render_obstacles(self, screen):
        for c, d in zip(self.env.obstacle_coords, self.env.obstacle_deltas):
            d = tuple(((c + d) * self.scale).tolist())
            c = tuple((c * self.scale).tolist())
            pygame.draw.line(screen, self.fg, c, d, 5)

    def render_bullets(self, screen, bullets, jdx):
        jdx += 1
        ally_bullets, enemy_bullets = bullets
        for source, target in ally_bullets:
            position = source + (target - source) * jdx / 8
            position *= self.scale
            pygame.draw.circle(screen, self.fg, tuple(position.tolist()), 3)
        for source, target in enemy_bullets:
            position = source + (target - source) * jdx / 8
            position *= self.scale
            pygame.draw.circle(screen, self.fg, tuple(position.tolist()), 3)


# functions
# bullet functions
def dist_fn(env, pos):  # computing the distances between all ally and enemy agents
    delta = pos[None, :, :] - pos[:, None, :]
    dist = jnp.sqrt((delta**2).sum(axis=2))
    dist = dist[: env.num_allies, env.num_allies :]
    return {"ally": dist, "enemy": dist.T}


def range_fn(env, dists, ranges):  # computing what targets are in range
    ally_range = dists["ally"] < ranges[: env.num_allies][:, None]
    enemy_range = dists["enemy"] < ranges[env.num_allies :][:, None]
    return {"ally": ally_range, "enemy": enemy_range}


def target_fn(acts, in_range, team):  # computing the one hot valid targets
    t_acts = jnp.stack([v for k, v in acts.items() if k.startswith(team)]).T
    t_targets = jnp.where(t_acts > 4, -1, t_acts - 5)  # first 5 are move actions
    t_attacks = jnp.eye(in_range[team].shape[1] + 1)[t_targets][:, :-1]
    return t_attacks * in_range[team]  # one hot valid targets


def attack_fn(env, state_seq):  # one hot attack list
    attacks = []
    for _, state, acts in state_seq:
        dists = dist_fn(env, state.unit_positions)
        ranges = env.unit_type_attack_ranges[state.unit_types]
        in_range = range_fn(env, dists, ranges)
        target = partial(target_fn, acts, in_range)
        attack = {"ally": target("ally"), "enemy": target("enemy")}
        attacks.append(attack)
    return attacks


def bullet_fn(env, states):
    bullet_seq = []
    attack_seq = attack_fn(env, states)

    def aux_fn(team):
        bullets = jnp.stack(jnp.where(one_hot[team] == 1)).T
        # bullets = bullets.at[:, 2 if team == "ally" else 1].add(env.num_allies)
        return bullets

    state_zip = zip(states[:-1], states[1:])
    for i, ((_, state, _), (_, n_state, _)) in enumerate(state_zip):
        one_hot = attack_seq[i]
        ally_bullets, enemy_bullets = aux_fn("ally"), aux_fn("enemy")

        ally_bullets_source = state.unit_positions[ally_bullets[:, 0]]
        enemy_bullets_target = n_state.unit_positions[enemy_bullets[:, 1]]

        enemy_bullets_source = state.unit_positions[
            enemy_bullets[:, 0] + env.num_allies
        ]
        ally_bullets_target = n_state.unit_positions[
            ally_bullets[:, 1] + env.num_allies
        ]

        ally_bullets = jnp.stack((ally_bullets_source, ally_bullets_target), axis=1)
        enemy_bullets = jnp.stack((enemy_bullets_source, enemy_bullets_target), axis=1)

        bullet_seq.append((ally_bullets, enemy_bullets))
    return bullet_seq


# test the visualizer
if __name__ == "__main__":
    from jax import random, numpy as jnp
    from parabellum import Parabellum, scenarios

    # small_multiples()  # testing small multiples (not working yet)
    # exit()

    n_envs = 2
    env = Parabellum(scenarios["default"], action_type="discrete")
    rng, reset_rng = random.split(random.PRNGKey(0))
    reset_key = random.split(reset_rng, n_envs)
    obs, state = vmap(env.reset)(reset_key)
    state_seq = []

    for i in range(100):
        rng, act_rng, step_rng = random.split(rng, 3)
        act_key = random.split(act_rng, (len(env.agents), n_envs))
        act = {
            a: vmap(env.action_space(a).sample)(act_key[i])
            for i, a in enumerate(env.agents)
        }
        step_key = random.split(step_rng, n_envs)
        state_seq.append((step_key, state, act))
        obs, state, reward, done, infos = vmap(env.step)(step_key, state, act)

    vis = Visualizer(env, state_seq)
    vis.animate()
