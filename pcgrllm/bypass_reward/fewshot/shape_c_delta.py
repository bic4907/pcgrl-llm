import jax.numpy as jnp

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "C" pattern
    height, width = curr_array.shape
    target_c = jnp.ones((height, width)) * 2  # Start with all walls

    # Create the "C" pattern
    for i in range(height):
        target_c = target_c.at[i, 0].set(1)  # Left vertical line
    for j in range(width):
        target_c = target_c.at[0, j].set(1)  # Top horizontal line
        target_c = target_c.at[height - 1, j].set(1)  # Bottom horizontal line

    # Calculate similarity of prev_array and curr_array to the target "C"
    match_prev = jnp.sum(prev_array == target_c)
    match_curr = jnp.sum(curr_array == target_c)

    # Calculate delta reward
    total_cells = height * width
    reward_prev = match_prev / total_cells
    reward_curr = match_curr / total_cells
    delta_reward = reward_curr - reward_prev

    return delta_reward


