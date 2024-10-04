"""Parabellum environment based on SMAX"""

import jax.numpy as jnp
import jax
from jax import random, Array, vmap, jit
from flax.struct import dataclass
import chex
from jaxmarl.environments.smax.smax_env import SMAX

from math import ceil

from typing import Tuple, Dict, cast
from functools import partial
from parabellum import tps, geo, terrain_db


@dataclass
class Scenario:
    """Parabellum scenario"""

    place: str
    terrain: tps.Terrain
    unit_starting_sectors: jnp.ndarray  # must be of size (num_units, 4) where sectors[i] = (x, y, width, height) of the ith unit's spawning sector (in % of the real map)
    unit_types: chex.Array
    num_allies: int
    num_enemies: int

    smacv2_position_generation: bool = False
    smacv2_unit_type_generation: bool = False


@dataclass
class State:
    # terrain: Array
    unit_positions: Array  # fsfds
    unit_alive: Array
    unit_teams: Array
    unit_health: Array
    unit_types: Array
    unit_weapon_cooldowns: Array
    prev_movement_actions: Array
    prev_attack_actions: Array
    time: int
    terminal: bool


def make_scenario(
    place,
    size,
    unit_starting_sectors,
    allies_type,
    n_allies,
    enemies_type,
    n_enemies,
):
    if place in terrain_db.db:
        terrain = terrain_db.make_terrain(terrain_db.db[place], size)
    else:
        terrain = geo.geography_fn(place, size)
    if type(unit_starting_sectors) == list:
        default_sector = [
            0,
            0,
            size,
            size,
        ]  # Noah feel confident that this is right. This means 50% chance. Sorry timothee if you end up here later. my bad bro.
        correct_unit_starting_sectors = []
        for i in range(n_allies + n_enemies):
            selected_sector = None
            for unit_ids, sector in unit_starting_sectors:
                if i in unit_ids:
                    selected_sector = sector
            if selected_sector is None:
                selected_sector = default_sector
            correct_unit_starting_sectors.append(selected_sector)
        unit_starting_sectors = correct_unit_starting_sectors
    if type(allies_type) == int:
        allies = [allies_type] * n_allies
    else:
        assert len(allies_type) == n_allies
        allies = allies_type

    if type(enemies_type) == int:
        enemies = [enemies_type] * n_enemies
    else:
        assert len(enemies_type) == n_enemies
        enemies = enemies_type
    unit_types = jnp.array(allies + enemies, dtype=jnp.uint8)
    return Scenario(
        place,
        terrain,
        unit_starting_sectors,  # type: ignore
        unit_types,
        n_allies,
        n_enemies,
    )


def scenario_fn(place):
    # scenario function for Noah, cos the one above is confusing
    terrain = geo.geography_fn(place)
    num_allies = 10
    num_enemies = 10
    unit_types = jnp.array([0] * num_allies + [1] * num_enemies, dtype=jnp.uint8)
    # start units in default sectors
    unit_starting_sectors = jnp.array([[0, 0, 1, 1]] * (num_allies + num_enemies))
    return Scenario(
        place=place,
        terrain=terrain,
        unit_starting_sectors=unit_starting_sectors,
        unit_types=unit_types,
        num_allies=num_allies,
        num_enemies=num_enemies,
    )


def spawn_fn(rng: jnp.ndarray, units_spawning_sectors):
    """Spawns n agents on a map."""
    spawn_positions = []
    for sector in units_spawning_sectors:
        rng, key_start, key_noise = random.split(rng, 3)
        noise = 0.25 + random.uniform(key_noise, (2,)) * 0.5
        idx = random.choice(key_start, sector[0].shape[0])
        coord = jnp.array([sector[0][idx], sector[1][idx]])
        spawn_positions.append(coord + noise)
    return jnp.array(spawn_positions, dtype=jnp.float32)


def sectors_fn(sectors: jnp.ndarray, invalid_spawn_areas: jnp.ndarray):
    """
    sectors must be of size (num_units, 4) where sectors[i] = (x, y, width, height) of the ith unit's spawning sector (in % of the real map)
    """
    width, height = invalid_spawn_areas.shape
    spawning_sectors = []
    for sector in sectors:
        coordx, coordy = (
            jnp.array(sector[0] * width, dtype=jnp.int32),
            jnp.array(sector[1] * height, dtype=jnp.int32),
        )
        sector = (
            invalid_spawn_areas[
                coordx : coordx + ceil(sector[2] * width),
                coordy : coordy + ceil(sector[3] * height),
            ]
            == 0
        )
        valid = jnp.nonzero(sector)
        if valid[0].shape[0] == 0:
            raise ValueError(f"Sector {sector} only contains invalid spawn areas.")
        spawning_sectors.append(
            jnp.array(valid) + jnp.array([coordx, coordy]).reshape((2, -1))
        )
    return spawning_sectors


class Environment(SMAX):
    def __init__(self, scenario: Scenario, **kwargs):
        map_height, map_width = scenario.terrain.building.shape
        args = dict(scenario=scenario, map_height=map_height, map_width=map_width)
        if "unit_type_pushable" in kwargs:
            self.unit_type_pushable = kwargs["unit_type_pushable"]
            del kwargs["unit_type_pushable"]
        else:
            self.unit_type_pushable = jnp.array([1, 1, 0, 0, 0, 1])
        if "reset_when_done" in kwargs:
            self.reset_when_done = kwargs["reset_when_done"]
            del kwargs["reset_when_done"]
        else:
            self.reset_when_done = True
        super(Environment, self).__init__(**args, walls_cause_death=False, **kwargs)
        self.terrain = scenario.terrain
        self.unit_starting_sectors = scenario.unit_starting_sectors
        # self.unit_type_names = ["tinker", "tailor", "soldier", "spy"]
        # self.unit_type_health = jnp.array([100, 100, 100, 100], dtype=jnp.float32)
        # self.unit_type_damage = jnp.array([10, 10, 10, 10], dtype=jnp.float32)
        self.scenario = scenario
        self.unit_type_velocities = (
            jnp.array([3.15, 2.25, 4.13, 3.15, 4.13, 3.15]) / 2.5
            if "unit_type_velocities" not in kwargs
            else kwargs["unit_type_velocities"]
        )
        self.unit_type_attack_blasts = jnp.zeros((3,), dtype=jnp.float32)  # TODO: add
        self.max_steps = 200
        self._push_units_away = lambda state, firmness=1: state  # overwrite push units
        self.spawning_sectors = sectors_fn(
            self.unit_starting_sectors,
            scenario.terrain.building + scenario.terrain.water,
        )
        self.resolution = (
            jnp.array(jnp.max(self.unit_type_sight_ranges), dtype=jnp.int32) * 2
        )
        self.t = jnp.tile(jnp.linspace(0, 1, self.resolution), (2, 1))

    def reset(self, rng: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:  # type: ignore
        """Environment-specific reset."""
        unit_positions = spawn_fn(rng, self.spawning_sectors)
        unit_teams = jnp.zeros((self.num_agents,))
        unit_teams = unit_teams.at[self.num_allies :].set(1)
        unit_weapon_cooldowns = jnp.zeros((self.num_agents,))
        # default behaviour spawn all marines
        unit_types = cast(Array, self.scenario.unit_types)
        unit_health = self.unit_type_health[unit_types]
        state = State(
            unit_positions=unit_positions,
            unit_alive=jnp.ones((self.num_agents,), dtype=jnp.bool_),
            unit_teams=unit_teams,
            unit_health=unit_health,
            unit_types=unit_types,
            prev_movement_actions=jnp.zeros((self.num_agents, 2)),
            prev_attack_actions=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            time=0,
            terminal=False,
            unit_weapon_cooldowns=unit_weapon_cooldowns,
            # terrain=self.terrain,
        )
        state = self._push_units_away(state)  # type: ignore  could be slow
        obs = self.get_obs(state)
        # remove world_state from obs
        world_state = self.get_world_state(state)
        obs["world_state"] = jax.lax.stop_gradient(world_state)
        return obs, state

        # def step_env(self, rng, state: State, action: Array):  # type: ignore
        # obs, state, rewards, dones, infos = super().step_env(rng, state, action)
        # delete world_state from obs
        # obs.pop("world_state")
        # if not self.reset_when_done:
        # for key in dones.keys():
        # dones[key] = False
        # return obs, state, rewards, dones, infos

    def get_obs_unit_list(self, state: State) -> Dict[str, chex.Array]:  # type: ignore
        """Applies observation function to state."""

        def get_features(i, j):
            """Get features of unit j as seen from unit i"""
            # Can just keep them symmetrical for now.
            # j here means 'the jth unit that is not i'
            # The observation is such that allies are always first
            # so for units in the second team we count in reverse.
            j = jax.lax.cond(
                i < self.num_allies, lambda: j, lambda: self.num_agents - j - 1
            )
            offset = jax.lax.cond(i < self.num_allies, lambda: 1, lambda: -1)
            j_idx = jax.lax.cond(
                ((j < i) & (i < self.num_allies)) | ((j > i) & (i >= self.num_allies)),
                lambda: j,
                lambda: j + offset,
            )
            empty_features = jnp.zeros(shape=(len(self.unit_features),))
            features = self._observe_features(state, i, j_idx)
            visible = (
                jnp.linalg.norm(state.unit_positions[j_idx] - state.unit_positions[i])
                < self.unit_type_sight_ranges[state.unit_types[i]]
            )
            return jax.lax.cond(
                visible
                & state.unit_alive[i]
                & state.unit_alive[j_idx]
                & self.has_line_of_sight(
                    state.unit_positions[j_idx],
                    state.unit_positions[i],
                    self.terrain.building + self.terrain.forest,
                ),
                lambda: features,
                lambda: empty_features,
            )

        get_all_features_for_unit = jax.vmap(get_features, in_axes=(None, 0))
        get_all_features = jax.vmap(get_all_features_for_unit, in_axes=(0, None))
        other_unit_obs = get_all_features(
            jnp.arange(self.num_agents), jnp.arange(self.num_agents - 1)
        )
        other_unit_obs = other_unit_obs.reshape((self.num_agents, -1))
        get_all_self_features = jax.vmap(self._get_own_features, in_axes=(None, 0))
        own_unit_obs = get_all_self_features(state, jnp.arange(self.num_agents))
        obs = jnp.concatenate([other_unit_obs, own_unit_obs], axis=-1)
        return {agent: obs[self.agent_ids[agent]] for agent in self.agents}

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
        return jnp.where(
            self.unit_type_pushable[unit_types][:, None], unit_positions, pos
        )

    def has_line_of_sight(self, source, target, raster_input):
        # suppose that the target is in sight_range of source, otherwise the line of sight might miss some cells
        cells = jnp.array(
            source[:, jnp.newaxis] * self.t + (1 - self.t) * target[:, jnp.newaxis],
            dtype=jnp.int32,
        )
        mask = jnp.zeros(raster_input.shape).at[cells[0, :], cells[1, :]].set(1)
        flag = ~jnp.any(jnp.logical_and(mask, raster_input))
        return flag

    @partial(jax.jit, static_argnums=(0,))  # replace the _world_step method
    def _world_step(  # modified version of JaxMARL's SMAX _world_step
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Tuple[chex.Array, chex.Array],
    ) -> State:
        def raster_crossing(pos, new_pos, mask: jnp.ndarray):
            pos, new_pos = pos.astype(jnp.int32), new_pos.astype(jnp.int32)
            minimum = jnp.minimum(pos, new_pos)
            maximum = jnp.maximum(pos, new_pos)
            mask = jnp.where(jnp.arange(mask.shape[0]) >= minimum[0], mask.T, 0).T
            mask = jnp.where(jnp.arange(mask.shape[0]) <= maximum[0], mask.T, 0).T
            mask = jnp.where(jnp.arange(mask.shape[1]) >= minimum[1], mask, 0)
            mask = jnp.where(jnp.arange(mask.shape[1]) <= maximum[1], mask, 0)
            return jnp.any(mask)

        def update_position(idx, vec):
            # Compute the movements slightly strangely.
            # The velocities below are for diagonal directions
            # because these are easier to encode as actions than the four
            # diagonal directions. Then rotate the velocity 45
            # degrees anticlockwise to compute the movement.
            pos = cast(Array, state.unit_positions[idx])
            new_pos = (
                pos
                + vec
                * self.unit_type_velocities[state.unit_types[idx]]
                * self.time_per_step
            )
            # avoid going out of bounds
            new_pos = jnp.maximum(
                jnp.minimum(
                    new_pos, jnp.array([self.map_width - 1, self.map_height - 1])
                ),
                jnp.zeros((2,)),
            )

            #######################################################################
            ############################################ avoid going into obstacles
            clash = raster_crossing(
                pos, new_pos, self.terrain.building + self.terrain.water
            )
            new_pos = jnp.where(clash, pos, new_pos)

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
            attacked_idx = cast(int, attacked_idx)  # Cast to int
            # deal with no-op attack actions (i.e. agents that are moving instead)
            attacked_idx = jax.lax.select(
                action < self.num_movement_actions, idx, attacked_idx
            )
            distance = jnp.linalg.norm(
                state.unit_positions[idx] - state.unit_positions[attacked_idx]
            )
            attack_valid = (
                (distance <= self.unit_type_attack_ranges[state.unit_types[idx]])
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
            health_diff = jnp.where(
                state.unit_types[idx] == 1,
                health_diff
                * distance
                / self.unit_type_attack_ranges[state.unit_types[idx]],
                health_diff,
            )
            # design choice based on the pysc2 randomness details.
            # See https://github.com/deepmind/pysc2/blob/master/docs/environment.md#determinism-and-randomness

            #########################################################
            ############################### Add bystander health diff

            # bystander_idxs = bystander_fn(attacked_idx)  # TODO: use
            # bystander_valid = (
            #     jnp.where(attack_valid, bystander_idxs, jnp.zeros((self.num_agents,)))
            #     .astype(jnp.bool_)  # type: ignore
            #     .astype(jnp.float32)
            # )
            # bystander_health_diff = (
            #     bystander_valid * -self.unit_type_attacks[state.unit_types[idx]]
            # )

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
                # (bystander_health_diff, bystander_idxs),
            )

        def perform_agent_action(idx, action, key):
            movement_action, attack_action = action
            new_pos = update_position(idx, movement_action)
            health_diff, attacked_idxes, cooldown_diff = update_agent_health(
                idx, attack_action, key
            )

            return new_pos, (health_diff, attacked_idxes), cooldown_diff

        keys = jax.random.split(key, num=self.num_agents)
        pos, (health_diff, attacked_idxes), cooldown_diff = jax.vmap(
            perform_agent_action
        )(jnp.arange(self.num_agents), actions, keys)

        # units push each other
        new_pos = self._our_push_units_away(pos, state.unit_types)
        clash = jax.vmap(raster_crossing, in_axes=(0, 0, None))(
            pos, new_pos, self.terrain.building + self.terrain.water
        )
        pos = jax.vmap(jnp.where)(clash, pos, new_pos)
        # avoid going out of bounds
        pos = jnp.maximum(
            jnp.minimum(pos, jnp.array([self.map_width - 1, self.map_height - 1])),  # type: ignore
            jnp.zeros((2,)),
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

        # _, bystander_health_diff = bystander
        # unit_health -= bystander_health_diff.sum(axis=0)  # might be axis=1

        #########################################################
        #########################################################

        unit_weapon_cooldowns = state.unit_weapon_cooldowns + cooldown_diff
        # replace unit health, unit positions and unit weapon cooldowns
        state = state.replace(  # type: ignore
            unit_health=unit_health,
            unit_positions=pos,
            unit_weapon_cooldowns=unit_weapon_cooldowns,
        )
        return state


if __name__ == "__main__":
    n_envs = 4

    n_allies = 10
    scenario_kwargs = {
        "allies_type": 0,
        "n_allies": n_allies,
        "enemies_type": 0,
        "n_enemies": n_allies,
        "place": "Vesterbro, Copenhagen, Denmark",
        "size": 100,
        "unit_starting_sectors": [
            ([i for i in range(n_allies)], [0.0, 0.45, 0.1, 0.1]),
            ([n_allies + i for i in range(n_allies)], [0.8, 0.5, 0.1, 0.1]),
        ],
    }
    scenario = make_scenario(**scenario_kwargs)
    env = Environment(scenario)
    rng, reset_rng = random.split(random.PRNGKey(0))
    reset_key = random.split(reset_rng, n_envs)
    obs, state = vmap(env.reset)(reset_key)
    state_seq = []

    import time

    step = vmap(jit(env.step))
    tic = time.time()
    for i in range(10):
        rng, act_rng, step_rng = random.split(rng, 3)
        act_key = random.split(act_rng, (len(env.agents), n_envs))
        act = {
            a: vmap(env.action_space(a).sample)(act_key[i])
            for i, a in enumerate(env.agents)
        }
        step_key = random.split(step_rng, n_envs)
        state_seq.append((step_key, state, act))
        obs, state, reward, done, infos = step(step_key, state, act)
        tic = time.time()
