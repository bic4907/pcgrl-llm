import jax.numpy as jnp
from jax import jit

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "C" pattern with thickness 4
    height, width = curr_array.shape
    target_c = jnp.ones((height, width)) * 2  # Start with all walls

    # Create the "C" pattern with thickness of 4
    thickness = 4  # Thickness of the "C" strokes

    for i in range(height):
        # Left vertical line
        for t in range(thickness):
            if 0 <= i < height:
                target_c = target_c.at[i, t].set(1)  # Set left thickness

        # Top horizontal line
        if 0 <= i < thickness:
            for j in range(width - thickness):  # Extend almost to the right edge
                target_c = target_c.at[i, j].set(1)  # Set top horizontal line

        # Bottom horizontal line
        if height - thickness <= i < height:
            for j in range(width - thickness):  # Extend almost to the right edge
                target_c = target_c.at[i, j].set(1)  # Set bottom horizontal line

    # Calculate similarity of prev_array and curr_array to the target "C"
    match_prev = jnp.sum(prev_array == target_c)
    match_curr = jnp.sum(curr_array == target_c)

    # Calculate delta reward
    total_cells = height * width
    reward_prev = match_prev / total_cells
    reward_curr = match_curr / total_cells
    delta_reward = reward_curr - reward_prev

    return delta_reward
