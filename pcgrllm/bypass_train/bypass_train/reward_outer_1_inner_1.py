import jax.numpy as jnp

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    reward = 0.0
    
    # Define the target letter "A" shape in a 16x16 grid
    target_shape = jnp.array([
        [0, 0, 2, 2, 2, 2, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0],
        [0, 2, 2, 2, 2, 2, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 2, 2, 2, 0, 0],
    ])
    
    # Scale the target shape to fit the 16x16 level
    target_shape_scaled = jnp.pad(target_shape, ((0, 8), (0, 8)), mode='constant', constant_values=1)
    
    # Calculate the number of matching tiles
    matches = jnp.sum((curr_array == 1) & (target_shape_scaled == 1))
    
    # Reward based on the number of matching tiles
    reward += matches * 0.1  # Each match gives a small reward
    
    return reward