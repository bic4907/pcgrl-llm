import jax.numpy as jnp
import random

def compute_reward(state) -> float:

    # return float(random.sample(range(-10, 10), 1)[0])
    #normal distribution
    return random.gauss(0, 1)