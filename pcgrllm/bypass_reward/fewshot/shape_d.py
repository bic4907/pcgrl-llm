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

        # Top and bottom horizontal lines
        if i == 0 or i == height - 1:
            for j in range(width // 2):
                target_d = target_d.at[i, j].set(1)

        # Right curve (semi-circle)
        if 0 < i < height - 1:
            target_d = target_d.at[i, width - 1].set(1)
            if i < height // 2:  # Upper curve
                target_d = target_d.at[i, width - i - 1].set(1)
            elif i > height // 2:  # Lower curve
                target_d = target_d.at[i, i].set(1)

    # Calculate the reward based on similarity to the target "D"
    match = jnp.sum(curr_array == target_d)
    total_cells = height * width
    reward = match / total_cells

    return reward