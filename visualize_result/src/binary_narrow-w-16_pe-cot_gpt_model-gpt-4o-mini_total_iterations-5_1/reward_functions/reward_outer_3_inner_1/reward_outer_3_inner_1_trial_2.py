import jax.numpy as jnp

def compute_reward(prev_array, prev_stats, curr_array, curr_stats) -> float:
    reward = 0.0
    
    # Define the target letter 'A' shape in a 16x16 grid
    target_shape = jnp.array([
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    # Calculate the number of matching tiles with the target shape
    matching_tiles = jnp.sum(curr_array == target_shape)
    
    # Calculate the total number of tiles in the target shape
    total_target_tiles = jnp.sum(target_shape == 1)
    
    # Reward is proportional to the number of matching tiles
    reward += matching_tiles * 0.1  # Adjust the scaling factor as needed
    
    # Introduce a penalty for non-matching values (e.g., tile number 2)
    non_matching_tiles = jnp.sum(curr_array == 2)
    reward -= non_matching_tiles * 0.05  # Adjust the penalty factor as needed
    
    # Reward for larger representations of the letter "A"
    size_factor = jnp.sum(curr_array == 1) / curr_array.size
    reward += jnp.where(size_factor > 0.5, 1.0, 0.0)  # Bonus for significant representation
    
    # Encourage variation in the representation of the letter "A"
    unique_tiles = jnp.unique(curr_array)
    reward += jnp.where(unique_tiles.size > 2, 0.5, 0.0)  # Reward for diversity in representation
    
    return reward