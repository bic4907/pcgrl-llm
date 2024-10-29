import jax.numpy as jnp

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "M" pattern
    height, width = curr_array.shape
    target_m = jnp.ones((height, width)) * 2  # Start with all walls

    # Create the "M" pattern
    mid_point = width // 2
    for i in range(height):
        target_m = target_m.at[i, 0].set(1)  # Left vertical line
        target_m = target_m.at[i, width - 1].set(1)  # Right vertical line
        if i <= height // 2:
            target_m = target_m.at[i, i].set(1)  # Left diagonal
            target_m = target_m.at[i, width - 1 - i].set(1)  # Right diagonal

    # Calculate similarity of prev_array and curr_array to the target "M"
    match_prev = jnp.sum(prev_array == target_m)
    match_curr = jnp.sum(curr_array == target_m)

    # Calculate delta reward
    total_cells = height * width
    reward_prev = match_prev / total_cells
    reward_curr = match_curr / total_cells
    delta_reward = reward_curr - reward_prev

    return delta_reward

