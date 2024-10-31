import jax.numpy as jnp

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    # Initialize reward
    reward = 0.0

    # Define the target letter 'D' shape
    height, width = curr_array.shape
    target_shape = jnp.zeros((height, width), dtype=jnp.int32)

    # Create the 'D' shape
    # Vertical line on the left
    target_shape = target_shape.at[:, 0].set(1)
    # Horizontal line on the top and bottom
    target_shape = target_shape.at[0, :].set(1)
    target_shape = target_shape.at[height-1, :].set(1)
    # Curved part of 'D' on the right
    for i in range(1, height-1):
        if i < height // 2:
            target_shape = target_shape.at[i, width-1].set(1)
        else:
            target_shape = target_shape.at[i, width-2].set(1)

    # Calculate the reward based on the similarity to the target shape
    similarity = jnp.sum(curr_array == target_shape)
    prev_similarity = jnp.sum(prev_array == target_shape)

    max_similarity = jnp.sum(target_shape == 1)

    # Normalize the reward
    reward_curr = similarity / max_similarity
    reward_prev = prev_similarity / max_similarity

    reward = reward_curr - reward_prev

    return reward