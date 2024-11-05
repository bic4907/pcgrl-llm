import jax.numpy as jnp

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "M" pattern
    height, width = curr_array.shape
    target_m = jnp.ones((height, width)) * 2  # Start with all walls

    # Create the "M" pattern

    for i in range(height):
        target_m = target_m.at[i, 0].set(1)  # Left vertical line
        target_m = target_m.at[i, width - 1].set(1)  # Right vertical line
        if i <= height:
            target_m = target_m.at[i, i].set(1)  # Left diagonal
            target_m = target_m.at[i, width - 1 - i].set(1)  # Right diagonal

    # Calculate the reward based on similarity to the target "M"
    match = jnp.sum(curr_array == target_m)
    total_cells = height * width
    reward = match / total_cells

    return reward
