import jax.numpy as jnp
from jax import jit



def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "M" pattern with thickness 4
    height, width = curr_array.shape
    target_m = jnp.ones((height, width)) * 2  # Start with all walls

    # Create the "M" pattern with thickness of 4
    mid_point = width // 2
    thickness = 4  # Thickness of the "M" strokes

    for i in range(height):
        # Left vertical line
        for t in range(thickness):
            if 0 <= i < height:
                target_m = target_m.at[i, t].set(1)  # Set left thickness

        # Right vertical line
        for t in range(thickness):
            if 0 <= i < height:
                target_m = target_m.at[i, width - 1 - t].set(1)  # Set right thickness

        # Left diagonal (only up to halfway down the height)
        if i <= height // 2:
            for t in range(thickness):
                if 0 <= i + t < height and 0 <= i < width:
                    target_m = target_m.at[i + t, i].set(1)  # Set thickness for left diagonal

        # Right diagonal (only up to halfway down the height)
        if i <= height // 2:
            for t in range(thickness):
                if 0 <= i + t < height and 0 <= width - 1 - i < width:
                    target_m = target_m.at[i + t, width - 1 - i].set(1)  # Set thickness for right diagonal

    # Calculate similarity of prev_array and curr_array to the target "M"
    match_prev = jnp.sum(prev_array == target_m)
    match_curr = jnp.sum(curr_array == target_m)

    # Calculate delta reward
    total_cells = height * width
    reward_prev = match_prev / total_cells
    reward_curr = match_curr / total_cells
    delta_reward = reward_curr - reward_prev

    return delta_reward
