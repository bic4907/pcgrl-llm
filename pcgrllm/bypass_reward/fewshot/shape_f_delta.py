import jax.numpy as jnp

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "F" pattern
    height, width = curr_array.shape
    target_f = jnp.ones((height, width)) * 2  # Start with all walls

    # Create the "F" pattern
    for i in range(height):
        target_f = target_f.at[i, 0].set(1)  # Left vertical line
    for j in range(width):
        target_f = target_f.at[0, j].set(1)  # Top horizontal line
    for j in range(width // 2):
        target_f = target_f.at[height // 2, j].set(1)  # Middle horizontal line

    # Calculate similarity of prev_array and curr_array to the target "F"
    match_prev = jnp.sum(prev_array == target_f)
    match_curr = jnp.sum(curr_array == target_f)

    # Calculate delta reward
    total_cells = height * width
    reward_prev = match_prev / total_cells
    reward_curr = match_curr / total_cells
    delta_reward = reward_curr - reward_prev

    return delta_reward


