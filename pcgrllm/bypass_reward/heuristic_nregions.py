import jax.numpy as jnp
from functools import partial
from jax import jit
import random

def compute_reward(prev_array, prev_stats, curr_array, curr_stats) -> float:
    '''
    prev_array: jnp.array
    prev_stats: dict (str, jnp.array)
    curr_array: jnp.array
    curr_stats: dict (str, jnp.array)
    '''

    curr_diameter = curr_stats['DIAMETER']
    prev_diameter = prev_stats['DIAMETER']
    curr_n_regions = curr_stats['N_REGIONS']
    prev_n_regions = prev_stats['N_REGIONS']

    prev_stats = jnp.array([prev_diameter, prev_n_regions])
    curr_stats = jnp.array([curr_diameter, curr_n_regions])

    stat_trgs = jnp.array([jnp.inf, 1])
    ctrl_threshes = jnp.array([0, 0])

    prev_loss = jnp.abs(stat_trgs - prev_stats)
    prev_loss = jnp.clip(prev_loss - ctrl_threshes, 0)
    loss = jnp.abs(stat_trgs - curr_stats)
    loss = jnp.clip(loss - ctrl_threshes, 0)
    reward = prev_loss - loss
    reward = jnp.where(stat_trgs == jnp.inf, curr_stats - prev_stats, reward)
    reward = jnp.where(stat_trgs == -jnp.inf, prev_stats - curr_stats, reward)
    reward = jnp.sum(reward)

    return reward