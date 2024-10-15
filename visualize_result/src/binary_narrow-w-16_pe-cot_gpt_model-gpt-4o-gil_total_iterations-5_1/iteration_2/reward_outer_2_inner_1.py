import jax.numpy as jnp

def compute_reward(prev_array, prev_stats, curr_array, curr_stats) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "A" pattern
    target_pattern = jnp.array([
        [0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1]
    ])

    # Get the dimensions of the current array
    height, width = curr_array.shape

    # Calculate the center position to place the target pattern
    center_y = height // 2 - target_pattern.shape[0] // 2
    center_x = width // 2 - target_pattern.shape[1] // 2

    # Extract the sub-array from the current array where the target pattern should be
    sub_array = curr_array[center_y:center_y + target_pattern.shape[0], center_x:center_x + target_pattern.shape[1]]

    # Calculate the match score between the sub-array and the target pattern
    match_score = jnp.sum(sub_array == target_pattern)

    # Normalize the match score by the total number of tiles in the target pattern
    max_score = target_pattern.size
    normalized_score = match_score / max_score

    # Penalize for any deviation from the target pattern
    deviation_penalty = jnp.sum(sub_array != target_pattern) * 0.1

    # Set the reward as the normalized score minus the deviation penalty
    reward = normalized_score - deviation_penalty

    # Ensure reward is non-negative
    reward = jnp.maximum(reward, 0.0)

    return reward