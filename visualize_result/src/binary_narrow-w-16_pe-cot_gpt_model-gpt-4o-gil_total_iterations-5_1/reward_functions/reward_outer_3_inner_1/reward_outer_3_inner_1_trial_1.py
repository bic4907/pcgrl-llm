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

    # Initialize maximum match score
    max_match_score = 0.0

    # Sliding window approach to find the best match for the target pattern
    for y in range(height - target_pattern.shape[0] + 1):
        for x in range(width - target_pattern.shape[1] + 1):
            # Extract the sub-array from the current array
            sub_array = curr_array[y:y + target_pattern.shape[0], x:x + target_pattern.shape[1]]

            # Calculate the match score between the sub-array and the target pattern
            match_score = jnp.sum(sub_array == target_pattern)

            # Update the maximum match score
            max_match_score = jnp.maximum(max_match_score, match_score)

    # Normalize the match score by the total number of tiles in the target pattern
    max_score = target_pattern.size
    normalized_score = max_match_score / max_score

    # Increase penalty for deviation from the target pattern
    deviation_penalty = (max_score - max_match_score) * 0.2

    # Set the reward as the normalized score minus the deviation penalty
    reward = normalized_score - deviation_penalty

    # Ensure reward is non-negative
    reward = jnp.maximum(reward, 0.0)

    return reward