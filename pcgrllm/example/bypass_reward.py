import jax.numpy as jnp
from functools import partial
from jax import jit
import random

def compute_reward(array: jnp.array, stats) -> float:
    reward = 0

    # Insight 1: Reward based on the difference between the diameter and a target value
    diameter = stats['DIAMETER']
    target_diameter = 5
    reward += abs(diameter - target_diameter)

    # Insight 2: Reward based on the number of regions
    n_regions = stats['N_REGIONS']
    target_regions = 10
    reward += abs(n_regions - target_regions)

    return reward