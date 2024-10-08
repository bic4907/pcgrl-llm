import jax.numpy as jnp
import random

def compute_reward(array: jnp.array, stats: dict) -> float:
    reward = 0

    diameter = stats['DIAMETER']

    reward += abs(diameter - 5)

    return reward