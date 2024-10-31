import jax.numpy as jnp
from jax import jit

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "F" pattern with thickness 4
    height, width = curr_array.shape
    target_f = jnp.ones((height, width)) * 2  # Start with all walls

    # Create the "F" pattern with thickness of 4
    thickness = 4  # Thickness of the "F" strokes

    for i in range(height):
        # Left vertical line
        for t in range(thickness):
            if 0 <= i < height:
                target_f = target_f.at[i, t].set(1)  # Set left thickness

        # Top horizontal line
        if 0 <= i < thickness:
            for j in range(width):
                target_f = target_f.at[i, j].set(1)  # Set top horizontal line

        # Middle horizontal line (around one-third of the height)
        if height // 3 <= i < height // 3 + thickness:
            for j in range(width // 2):
                target_f = target_f.at[i, j].set(1)  # Set middle horizontal line up to half of the width

    # Calculate similarity of prev_array and curr_array to the target "F"
    match_prev = jnp.sum(prev_array == target_f)
    match_curr = jnp.sum(curr_array == target_f)

    # Calculate delta reward
    total_cells = height * width
    reward_prev = match_prev / total_cells
    reward_curr = match_curr / total_cells
    delta_reward = reward_curr - reward_prev

    return delta_reward
