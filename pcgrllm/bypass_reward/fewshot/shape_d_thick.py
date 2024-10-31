import jax.numpy as jnp
from jax import jit

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "D" pattern with thickness 4
    height, width = curr_array.shape
    target_d = jnp.ones((height, width)) * 2  # Start with all walls

    # Create the "D" pattern with thickness of 4
    thickness = 4  # Thickness of the "D" strokes
    curve_width = width - thickness  # Width of the curved part of "D"

    for i in range(height):
        # Left vertical line
        for t in range(thickness):
            if 0 <= i < height:
                target_d = target_d.at[i, t].set(1)  # Set left thickness

        # Top horizontal line for the curved part
        if 0 <= i < thickness:
            for j in range(thickness, curve_width):
                target_d = target_d.at[i, j].set(1)  # Set top horizontal line for the curve

        # Bottom horizontal line for the curved part
        if height - thickness <= i < height:
            for j in range(thickness, curve_width):
                target_d = target_d.at[i, j].set(1)  # Set bottom horizontal line for the curve

        # Right vertical curve line, gradually descending to form a rounded shape
        if thickness <= i < height - thickness:
            for t in range(thickness):
                target_d = target_d.at[i, curve_width + t - 1].set(1)  # Set right curved thickness

    # Calculate similarity of prev_array and curr_array to the target "D"
    match_prev = jnp.sum(prev_array == target_d)
    match_curr = jnp.sum(curr_array == target_d)

    # Calculate delta reward
    total_cells = height * width
    reward_prev = match_prev / total_cells
    reward_curr = match_curr / total_cells
    delta_reward = reward_curr - reward_prev

    return delta_reward
