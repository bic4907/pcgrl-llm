import jax.numpy as jnp
from jax import jit

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "N" pattern with thickness 4
    height, width = curr_array.shape
    target_n = jnp.ones((height, width)) * 2  # Start with all walls

    # Create the "N" pattern with thickness of 4
    thickness = 4  # Thickness of the "N" strokes

    for i in range(height):
        # Left vertical line
        for t in range(thickness):
            if 0 <= i < height:
                target_n = target_n.at[i, t].set(1)  # Set left thickness

        # Right vertical line
        for t in range(thickness):
            if 0 <= i < height:
                target_n = target_n.at[i, width - 1 - t].set(1)  # Set right thickness

        # Diagonal line connecting left to right (from top-left to bottom-right)
        if i + thickness < height:
            for t in range(thickness):
                if 0 <= i < height and 0 <= i + t < width:
                    target_n = target_n.at[i + t, i + t].set(1)  # Set thickness for diagonal

    # Calculate similarity of prev_array and curr_array to the target "N"
    match_prev = jnp.sum(prev_array == target_n)
    match_curr = jnp.sum(curr_array == target_n)

    # Calculate delta reward
    total_cells = height * width
    reward_prev = match_prev / total_cells
    reward_curr = match_curr / total_cells
    delta_reward = reward_curr - reward_prev

    return delta_reward
