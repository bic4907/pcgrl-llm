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
        target_n = target_n.at[i, i].set(1)  # Diagonal from top-left to bottom-right

    # Calculate similarity of prev_array and curr_array to the target "N"
    match_prev = jnp.sum(prev_array == target_n)
    match_curr = jnp.sum(curr_array == target_n)

    # Calculate delta reward
    total_cells = height * width
    reward_prev = match_prev / total_cells
    reward_curr = match_curr / total_cells
    delta_reward = reward_curr - reward_prev

    return delta_reward

