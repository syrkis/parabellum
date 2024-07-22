# gun.py
#   parabellum bullet rendering assosciated functions
# by: Noah Syrkis

# imports
from functools import partial
import jax.numpy as jnp


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
