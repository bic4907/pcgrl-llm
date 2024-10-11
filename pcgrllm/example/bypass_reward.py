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
    reward = 0.0


    return reward