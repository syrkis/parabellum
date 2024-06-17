"""Parabellum environment based on SMAX"""

import jax.numpy as jnp
import jax
import numpy as np
from jax import random
from jax import jit
from flax.struct import dataclass
import chex
from jaxmarl.environments.smax.smax_env import State, SMAX
from typing import Tuple, Dict
from functools import partial


@dataclass
class Scenario:
    """Parabellum scenario"""

    obstacle_coords: chex.Array
    obstacle_deltas: chex.Array

    unit_types: chex.Array
    num_allies: int = 9
    num_enemies: int = 10

    smacv2_position_generation: bool = False
    smacv2_unit_type_generation: bool = False


# default scenario
scenarios = {
    "default": Scenario(
        jnp.array([[6, 10], [26, 10]]) * 8,
        jnp.array([[0, 12], [0, 1]]) * 8,
        jnp.zeros((19,), dtype=jnp.uint8),
    )
}


class Parabellum(SMAX):
    def __init__(
        self,
        scenario: Scenario = scenarios["default"],
        unit_type_attack_blasts=jnp.array([0, 0, 0, 0, 0, 0]) + 8,
        **kwargs,
    ):
        super().__init__(scenario=scenario, **kwargs)
        self.unit_type_attack_blasts = unit_type_attack_blasts
        self.obstacle_coords = scenario.obstacle_coords.astype(jnp.float32)
        self.obstacle_deltas = scenario.obstacle_deltas.astype(jnp.float32)
        self.max_steps = 200
        # overwrite supers _world_step method

    @partial(jax.jit, static_argnums=(0,))  # replace the _world_step method
    def _world_step(  # modified version of JaxMARL's SMAX _world_step
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Tuple[chex.Array, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
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

            @partial(jax.vmap, in_axes=(None, None, 0, 0))
            def inter_fn(pos, new_pos, obs, obs_end):
                d1 = jnp.cross(obs - pos, new_pos - pos)
                d2 = jnp.cross(obs_end - pos, new_pos - pos)
                d3 = jnp.cross(pos - obs, obs_end - obs)
                d4 = jnp.cross(new_pos - obs, obs_end - obs)
                return (d1 * d2 < 0) & (d3 * d4 < 0)

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
    env = Parabellum(map_width=256, map_height=256)
    rng, key = random.split(random.PRNGKey(0))
    obs, state = env.reset(key)
    state_seq = []
    for step in range(100):
        rng, key = random.split(rng)
        key_act = random.split(key, len(env.agents))
        actions = {
            agent: jax.random.randint(key_act[i], (), 0, 5)
            for i, agent in enumerate(env.agents)
        }
        _, state, _, _, _ = env.step(key, state, actions)
        state_seq.append((obs, state, actions))
