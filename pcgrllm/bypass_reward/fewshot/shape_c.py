import jax.numpy as jnp

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "C" pattern
    height, width = curr_array.shape
    target_c = jnp.ones((height, width)) * 2  # Start with all walls

    # Create the "C" pattern
    for i in range(height):
        if i == 0 or i == height - 1:  # Top and bottom horizontal lines
            for j in range(width - 1):  # Leave the rightmost column empty
                target_c = target_c.at[i, j].set(1)
        else:  # Left vertical line
            target_c = target_c.at[i, 0].set(1)

    # Calculate the reward based on similarity to the target "C"
    match = jnp.sum(curr_array == target_c)
    total_cells = height * width
    reward = match / total_cells

    return reward