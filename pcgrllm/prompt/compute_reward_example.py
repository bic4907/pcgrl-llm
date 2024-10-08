import jax.numpy as jnp
from functools import partial
from jax import jit
import random

def compute_reward(array: jnp.array, stats) -> float:
    reward = 0

    diameter = stats['DIAMETER']

    reward += abs(diameter - 5)

    return reward