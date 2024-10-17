import jax.numpy as jnp

def compute_reward(prev_array, prev_stats, curr_array, curr_stats) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter "B" pattern
    target_pattern = jnp.array([
        [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

    # Calculate the match between the current array and the target pattern
    match = jnp.sum(curr_array == target_pattern)

    # Normalize the match to get a reward value between 0 and 1
    max_possible_match = target_pattern.size
    reward = match / max_possible_match

    # Adjust the weight of the pattern matching component
    reward *= 2.0  # Increase the influence of pattern matching

    # Penalize for noise and artifacts
    noise_penalty = jnp.sum((curr_array != 1) & (curr_array != 2))
    reward -= noise_penalty * 0.01  # Increase penalty for noise

    # Revise the uniformity penalty to encourage larger letters
    uniformity_penalty = jnp.sum(curr_array == 1) / curr_array.size
    reward -= uniformity_penalty * 0.1  # Increase penalty for uniformity

    # Introduce a reward component for the size of the letter
    size_reward = jnp.sum(curr_array == 2) / curr_array.size
    reward += size_reward * 0.2  # Increase reward for size

    # Strong penalty for lack of structure
    no_structure_penalty = jnp.where(jnp.sum(curr_array == 2) == 0, 0.5, 0.0)
    reward -= no_structure_penalty

    # Ensure reward is within bounds
    reward = jnp.clip(reward, 0.0, 1.0)

    return reward