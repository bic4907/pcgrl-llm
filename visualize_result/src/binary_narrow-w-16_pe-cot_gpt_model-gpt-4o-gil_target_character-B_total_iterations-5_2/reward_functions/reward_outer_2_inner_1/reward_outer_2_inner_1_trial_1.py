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

    # Penalize for noise and artifacts
    noise_penalty = jnp.sum((curr_array != 1) & (curr_array != 2))
    reward -= noise_penalty * 0.01

    # Reward for maintaining symmetry and consistency
    symmetry_reward = jnp.sum(curr_array[:, :8] == jnp.flip(curr_array[:, 8:], axis=1))
    reward += symmetry_reward / max_possible_match

    # Ensure reward is within bounds
    reward = jnp.clip(reward, 0.0, 1.0)

    return reward