import jax.numpy as jnp

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "N" pattern
    height, width = curr_array.shape
    target_n = jnp.ones((height, width)) * 2  # Start with all walls

    # Create the "N" pattern
    for i in range(height):
        target_n = target_n.at[i, 0].set(1)  # Left vertical line
        target_n = target_n.at[i, width - 1].set(1)  # Right vertical line
        target_n = target_n.at[i, i].set(1)  # Diagonal from bottom-left to top-right

    # Calculate the reward based on similarity to the target "N"
    match = jnp.sum(curr_array == target_n)
    total_cells = height * width
    reward = match / total_cells

    return reward