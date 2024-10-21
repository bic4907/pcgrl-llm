import jax.numpy as jnp

def compute_reward(prev_array, prev_stats, curr_array, curr_stats) -> float:
    reward = 0.0

    # Define the target letter "A" pattern
    target_letter = jnp.array([
        [0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1]
    ])

    # Scale the target letter to fit the current array size
    scale_factor = curr_array.shape[0] // target_letter.shape[0]
    scaled_target = jnp.kron(target_letter, jnp.ones((scale_factor, scale_factor)))

    # Ensure the scaled target fits within the current array
    scaled_target = scaled_target[:curr_array.shape[0], :curr_array.shape[1]]

    # Calculate the difference between the current array and the target pattern
    difference = jnp.abs(curr_array - scaled_target)

    # Calculate the reward based on how closely the current array matches the target pattern
    match_score = jnp.sum(difference == 0)
    total_tiles = curr_array.size

    # Reward is based on the proportion of tiles that match the target pattern
    reward = match_score / total_tiles

    return reward