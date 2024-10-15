import jax.numpy as jnp

def compute_reward(prev_array, prev_stats, curr_array, curr_stats) -> float:
    reward = 0.0

    # Define the target pattern for the letter 'A'
    target_pattern = jnp.array([
        [0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1]
    ])

    # Extract the central part of the current array to compare with the target pattern
    curr_central = curr_array[5:11, 5:11]

    # Calculate shape matching score
    shape_match_score = jnp.sum(curr_central == target_pattern)

    # Calculate symmetry score
    left_half = curr_array[:, :8]
    right_half = curr_array[:, 8:]
    symmetry_score = jnp.sum(left_half == jnp.flip(right_half, axis=1))

    # Calculate central peak score
    central_peak_score = jnp.sum(curr_array[0:3, 7:9] == 2)

    # Calculate legs score
    legs_score = jnp.sum(curr_array[3:6, [6, 10]] == 2)

    # Combine scores into a total reward
    reward = shape_match_score + symmetry_score + central_peak_score + legs_score

    # Normalize the reward
    reward = jnp.tanh(reward / 100.0)

    return reward