import jax.numpy as jnp

def compute_reward(prev_array, prev_stats, curr_array, curr_stats) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "B" pattern
    target_pattern = jnp.array([
        [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

    # Calculate the similarity between the current array and the target pattern
    similarity = jnp.sum(curr_array == target_pattern)

    # Normalize the similarity score by the total number of tiles
    max_similarity = target_pattern.size
    normalized_similarity = similarity / max_similarity

    # Assign the normalized similarity as the reward
    reward += normalized_similarity

    return reward