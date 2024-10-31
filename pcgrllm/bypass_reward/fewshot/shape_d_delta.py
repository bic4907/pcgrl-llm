import jax.numpy as jnp

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "D" pattern
    height, width = curr_array.shape
    target_d = jnp.ones((height, width)) * 2  # Start with all walls

    # Create the "D" pattern
    for i in range(height):
        target_d = target_d.at[i, 0].set(1)  # Left vertical line
        if i == 0 or i == height - 1:  # Top and bottom horizontal lines
            for j in range(1, width - 1):
                target_d = target_d.at[i, j].set(1)
        else:  # Curved right side
            target_d = target_d.at[i, width - 1].set(1)

    # Calculate similarity of prev_array and curr_array to the target "D"
    match_prev = jnp.sum(prev_array == target_d)
    match_curr = jnp.sum(curr_array == target_d)

    # Calculate delta reward
    total_cells = height * width
    reward_prev = match_prev / total_cells
    reward_curr = match_curr / total_cells
    delta_reward = reward_curr - reward_prev

    return delta_reward


