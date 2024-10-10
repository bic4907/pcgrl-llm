import jax.numpy as jnp
from functools import partial
from jax import jit
import random

def compute_reward(array: jnp.array, stats) -> float:
    reward = 0

    # Insight 1: Reward based on the difference between the diameter and a target value
    diameter = stats['DIAMETER']
    reward += diameter

    # Insight 2: Reward based on the number of regions
    n_regions = jnp.abs(stats['N_REGIONS'] - 1)
    reward += n_regions

    return reward