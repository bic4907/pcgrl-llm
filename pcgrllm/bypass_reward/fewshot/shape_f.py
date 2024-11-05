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

    # Add the top and middle horizontal lines
    for j in range(width):
        target_f = target_f.at[1, j].set(1)  # Top horizontal line
        if j < width // 4:  # Middle horizontal line (halfway)
            target_f = target_f.at[height // 2, j].set(1)

    # Calculate the reward based on similarity to the target "F"
    match = jnp.sum(curr_array == target_f)
    total_cells = height * width
    reward = match / total_cells

    return reward