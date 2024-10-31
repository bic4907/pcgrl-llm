import jax.numpy as jnp

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    reward = 0.0

    # Define target counts for the scenario
    target_spiders = 3
    target_doors = 1

    # Count the number of spiders and doors in the current array
    curr_spiders = jnp.sum(curr_array == 6)
    curr_doors = jnp.sum(curr_array == 8)

    # Count the number of spiders and doors in the previous array
    prev_spiders = jnp.sum(prev_array == 6)
    prev_doors = jnp.sum(prev_array == 8)

    # Calculate the difference in spider count and apply delta reward
    spider_diff = abs(prev_spiders - target_spiders) - abs(curr_spiders - target_spiders)
    reward += spider_diff * 0.5

    # Calculate the difference in door count and apply delta reward
    door_diff = abs(prev_doors - target_doors) - abs(curr_doors - target_doors)
    reward += door_diff * 1.0

    return reward