import jax.numpy as jnp

def compute_reward(array: jnp.array, stats) -> float:
    reward = 0

    diameter = stats['DIAMETER']

    reward += abs(diameter - 5)

    return reward