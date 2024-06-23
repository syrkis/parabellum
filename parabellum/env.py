"""Parabellum environment based on SMAX"""

import jax.numpy as jnp
import jax
import numpy as np
from jax import random
from jax import jit
from flax.struct import dataclass
import chex
from jax import vmap
from jaxmarl.environments.smax.smax_env import State, SMAX
from typing import Tuple, Dict
from functools import partial


@dataclass
class Scenario:
    """Parabellum scenario"""

    obstacle_coords: chex.Array  # TODO: use map instead of obstacles
    obstacle_deltas: chex.Array

    unit_types: chex.Array
    num_allies: int
    num_enemies: int

    smacv2_position_generation: bool = False
    smacv2_unit_type_generation: bool = False


# default scenario
scenarios = {
    "default": Scenario(
        jnp.array([[6, 10], [26, 10]]) * 8,
        jnp.array([[0, 12], [0, 1]]) * 8,
        jnp.array([[6, 10], [26, 10]]) * 8,
        jnp.array([[0, 12], [0, 1]]) * 8,
        jnp.zeros((19,), dtype=jnp.uint8),
        9,
        10,
    )
}


class Parabellum(SMAX):
    def __init__(self, scenario: Scenario, **kwargs):
        super(Parabellum, self).__init__(**kwargs)
        self.obstacle_coords = scenario.obstacle_coords
        self.obstacle_deltas = scenario.obstacle_deltas
        self.unit_type_attack_blasts = jnp.zeros((19,), dtype=jnp.float32)
        self.max_steps = 200
        self._push_units_away = lambda x: x  # overwrite push units

    def _our_push_units_away(
        self, pos, unit_types, firmness: float = 1.0
    ):  # copy of SMAX._push_units_away but used without state and called inside _world_step to allow more obstacles constraints
        delta_matrix = pos[:, None] - pos[None, :]
        dist_matrix = (
            jnp.linalg.norm(delta_matrix, axis=-1)
            + jnp.identity(self.num_agents)
            + 1e-6
        )
        radius_matrix = (
            self.unit_type_radiuses[unit_types][:, None]
            + self.unit_type_radiuses[unit_types][None, :]
        )
        overlap_term = jax.nn.relu(radius_matrix / dist_matrix - 1.0)
        unit_positions = (
            pos
            + firmness * jnp.sum(delta_matrix * overlap_term[:, :, None], axis=1) / 2
        )
        return unit_positions

    @partial(jax.jit, static_argnums=(0,))  # replace the _world_step method
    def _world_step(  # modified version of JaxMARL's SMAX _world_step
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Tuple[chex.Array, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        @partial(jax.vmap, in_axes=(None, None, 0, 0))
        def inter_fn(pos, new_pos, obs, obs_end):
            d1 = jnp.cross(obs - pos, new_pos - pos)
            d2 = jnp.cross(obs_end - pos, new_pos - pos)
            d3 = jnp.cross(pos - obs, obs_end - obs)
            d4 = jnp.cross(new_pos - obs, obs_end - obs)
            return (d1 * d2 <= 0) & (d3 * d4 <= 0)

        def update_position(idx, vec):
            # Compute the movements slightly strangely.
            # The velocities below are for diagonal directions
            # because these are easier to encode as actions than the four
            # diagonal directions. Then rotate the velocity 45
            # degrees anticlockwise to compute the movement.
            pos = state.unit_positions[idx]
            new_pos = (
                pos
                + vec
                * self.unit_type_velocities[state.unit_types[idx]]
                * self.time_per_step
            )
            # avoid going out of bounds
            new_pos = jnp.maximum(
                jnp.minimum(new_pos, jnp.array([self.map_width, self.map_height])),
                jnp.zeros((2,)),
            )

            #######################################################################
            ############################################ avoid going into obstacles
            obs = self.obstacle_coords
            obs_end = obs + self.obstacle_deltas
            inters = jnp.any(inter_fn(pos, new_pos, obs, obs_end))
            new_pos = jnp.where(inters, pos, new_pos)

            #######################################################################
            #######################################################################

            return new_pos

        #######################################################################
        ######################################### units close enough to get hit

        def bystander_fn(attacked_idx):
            idxs = jnp.zeros((self.num_agents,))
            idxs *= (
                jnp.linalg.norm(
                    state.unit_positions - state.unit_positions[attacked_idx], axis=-1
                )
                < self.unit_type_attack_blasts[state.unit_types[attacked_idx]]
            )
            return idxs

            #######################################################################
            #######################################################################

        def update_agent_health(idx, action, key):  # TODO: add attack blasts
            # for team 1, their attack actions are labelled in
            # reverse order because that is the order they are
            # observed in
            attacked_idx = jax.lax.cond(
                idx < self.num_allies,
                lambda: action + self.num_allies - self.num_movement_actions,
                lambda: self.num_allies - 1 - (action - self.num_movement_actions),
            )
            # deal with no-op attack actions (i.e. agents that are moving instead)
            attacked_idx = jax.lax.select(
                action < self.num_movement_actions, idx, attacked_idx
            )

            attack_valid = (
                (
                    jnp.linalg.norm(
                        state.unit_positions[idx] - state.unit_positions[attacked_idx]
                    )
                    < self.unit_type_attack_ranges[state.unit_types[idx]]
                )
                & state.unit_alive[idx]
                & state.unit_alive[attacked_idx]
            )
            attack_valid = attack_valid & (idx != attacked_idx)
            attack_valid = attack_valid & (state.unit_weapon_cooldowns[idx] <= 0.0)
            health_diff = jax.lax.select(
                attack_valid,
                -self.unit_type_attacks[state.unit_types[idx]],
                0.0,
            )
            # design choice based on the pysc2 randomness details.
            # See https://github.com/deepmind/pysc2/blob/master/docs/environment.md#determinism-and-randomness

            #########################################################
            ############################### Add bystander health diff

            bystander_idxs = bystander_fn(attacked_idx)  # TODO: use
            bystander_valid = (
                jnp.where(attack_valid, bystander_idxs, jnp.zeros((self.num_agents,)))
                .astype(jnp.bool_)
                .astype(jnp.float32)
            )
            bystander_health_diff = (
                bystander_valid * -self.unit_type_attacks[state.unit_types[idx]]
            )

            #########################################################
            #########################################################

            cooldown_deviation = jax.random.uniform(
                key, minval=-self.time_per_step, maxval=2 * self.time_per_step
            )
            cooldown = (
                self.unit_type_weapon_cooldowns[state.unit_types[idx]]
                + cooldown_deviation
            )
            cooldown_diff = jax.lax.select(
                attack_valid,
                # subtract the current cooldown because we are
                # going to add it back. This way we effectively
                # set the new cooldown to `cooldown`
                cooldown - state.unit_weapon_cooldowns[idx],
                -self.time_per_step,
            )
            return (
                health_diff,
                attacked_idx,
                cooldown_diff,
                (bystander_health_diff, bystander_idxs),
            )

        def perform_agent_action(idx, action, key):
            movement_action, attack_action = action
            new_pos = update_position(idx, movement_action)
            health_diff, attacked_idxes, cooldown_diff, (bystander) = (
                update_agent_health(idx, attack_action, key)
            )

            return new_pos, (health_diff, attacked_idxes), cooldown_diff, bystander

        keys = jax.random.split(key, num=self.num_agents)
        pos, (health_diff, attacked_idxes), cooldown_diff, bystander = jax.vmap(
            perform_agent_action
        )(jnp.arange(self.num_agents), actions, keys)

        # units push each other
        new_pos = self._our_push_units_away(pos, state.unit_types)

        # avoid going into obstacles after being pushed

        bondaries_coords = jnp.array(
            [[0, 0], [0, 0], [self.map_width, 0], [0, self.map_height]]
        )
        bondaries_deltas = jnp.array(
            [
                [self.map_width, 0],
                [0, self.map_height],
                [0, self.map_height],
                [self.map_width, 0],
            ]
        )
        obstacle_coords = jnp.concatenate(
            [self.obstacle_coords, bondaries_coords]
        )  # add the map boundaries to the obstacles to avoid
        obstacle_deltas = jnp.concatenate(
            [self.obstacle_deltas, bondaries_deltas]
        )  # add the map boundaries to the obstacles to avoid
        obst_start = obstacle_coords
        obst_end = obst_start + obstacle_deltas

        def check_obstacles(pos, new_pos, obst_start, obst_end):
            inters = jnp.any(inter_fn(pos, new_pos, obst_start, obst_end))
            return jnp.where(inters, pos, new_pos)

        pos = jax.vmap(check_obstacles, in_axes=(0, 0, None, None))(
            pos, new_pos, obst_start, obst_end
        )

        # Multiple enemies can attack the same unit.
        # We have `(health_diff, attacked_idx)` pairs.
        # `jax.lax.scatter_add` aggregates these exactly
        # in the way we want -- duplicate idxes will have their
        # health differences added together. However, it is a
        # super thin wrapper around the XLA scatter operation,
        # which has this bonkers syntax and requires this dnums
        # parameter. The usage here was inferred from a test:
        # https://github.com/google/jax/blob/main/tests/lax_test.py#L2296
        dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )
        unit_health = jnp.maximum(
            jax.lax.scatter_add(
                state.unit_health,
                jnp.expand_dims(attacked_idxes, 1),
                health_diff,
                dnums,
            ),
            0.0,
        )

        #########################################################
        ############################ subtracting bystander health

        _, bystander_health_diff = bystander
        unit_health -= bystander_health_diff.sum(axis=0)  # might be axis=1

        #########################################################
        #########################################################

        unit_weapon_cooldowns = state.unit_weapon_cooldowns + cooldown_diff
        state = state.replace(
            unit_health=unit_health,
            unit_positions=pos,
            unit_weapon_cooldowns=unit_weapon_cooldowns,
        )
        return state


if __name__ == "__main__":
    n_envs = 4
    kwargs = dict(map_width=64, map_height=64)
    env = Parabellum(scenarios["default"], **kwargs)
    rng, reset_rng = random.split(random.PRNGKey(0))
    reset_key = random.split(reset_rng, n_envs)
    obs, state = vmap(env.reset)(reset_key)
    state_seq = []

    for i in range(10):
        rng, act_rng, step_rng = random.split(rng, 3)
        act_key = random.split(act_rng, (len(env.agents), n_envs))
        act = {
            a: vmap(env.action_space(a).sample)(act_key[i])
            for i, a in enumerate(env.agents)
        }
        step_key = random.split(step_rng, n_envs)
        state_seq.append((step_key, state, act))
        obs, state, reward, done, infos = vmap(env.step)(step_key, state, act)
