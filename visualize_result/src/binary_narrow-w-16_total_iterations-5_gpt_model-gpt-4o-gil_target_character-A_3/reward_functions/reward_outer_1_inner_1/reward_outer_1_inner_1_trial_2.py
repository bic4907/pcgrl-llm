import jax.numpy as jnp

def compute_reward(prev_array, prev_stats, curr_array, curr_stats) -> float:
    reward = 0.0

    # Define the target letter "A" pattern
    height, width = curr_array.shape
    mid = width // 2

    # Create a target pattern for "A"
    target_pattern = jnp.zeros((height, width), dtype=jnp.int32)

    # Create the two diagonal lines of "A"
    for i in range(height):
        if i < height // 2:
            target_pattern = target_pattern.at[i, mid - i].set(1)
            target_pattern = target_pattern.at[i, mid + i].set(1)
        else:
            target_pattern = target_pattern.at[i, mid - (height // 2 - 1)].set(1)
            target_pattern = target_pattern.at[i, mid + (height // 2 - 1)].set(1)

    # Create the horizontal line of "A"
    for j in range(mid - (height // 2 - 1), mid + (height // 2)):
        target_pattern = target_pattern.at[height // 2, j].set(1)

    # Calculate the difference between the current array and the target pattern
    difference = jnp.sum(jnp.abs(curr_array - target_pattern))

    # Calculate the reward based on how close the current array is to the target pattern
    max_difference = height * width * 2  # Maximum possible difference
    reward = (max_difference - difference) / max_difference

    return reward