import jax.numpy as jnp
from functools import partial
from jax import jit
import random

class generate_reward:
    def __init__(self, array_shape: jnp.array, target_character: str, stats_keys: dict):
        self.array_shape = array_shape
        self.target_character = target_character
        self.stats_keys = stats_keys

    def compute_reward(self, array: jnp.array, stats) -> float:
        reward = 0

        # start of code
        diameter = stats['DIAMETER']

        reward += abs(diameter - 5)
        # end of  code

        return reward