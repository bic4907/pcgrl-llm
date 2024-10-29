


import jax.numpy as jnp



import jax.numpy as jnp
import jax



import jax.numpy as jnp
import jax

def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    reward = 0.0

    # Define tile numbers for readability
    EMPTY = 1
    WALL = 2
    SPIDER = 3
    PLAYER = 4
    KEY = 5
    DOOR = 6

    # Calculate the number of each tile type in the current array
    num_empty_tiles = jnp.sum(curr_array == EMPTY)
    num_wall_tiles = jnp.sum(curr_array == WALL)
    num_spiders = jnp.sum(curr_array == SPIDER)
    num_players = jnp.sum(curr_array == PLAYER)
    num_keys = jnp.sum(curr_array == KEY)
    num_doors = jnp.sum(curr_array == DOOR)

    # Reward for having a room with 3 spiders
    reward += jax.lax.cond(num_spiders == 3, lambda: 10.0, lambda: -5.0)

    # Reward for having exactly one player
    reward += jax.lax.cond(num_players == 1, lambda: 5.0, lambda: -5.0)

    # Reward for having exactly one key
    reward += jax.lax.cond(num_keys == 1, lambda: 5.0, lambda: -5.0)

    # Reward for having exactly one door
    reward += jax.lax.cond(num_doors == 1, lambda: 5.0, lambda: -5.0)

    # Penalty for having too many WALL tiles, which might indicate a small letter
    reward -= num_wall_tiles * 0.05

    # Ensure the entire level is filled with valid tiles
    total_tiles = curr_array.size
    valid_tiles = num_empty_tiles + num_wall_tiles + num_spiders + num_players + num_keys + num_doors
    reward += jax.lax.cond(valid_tiles == total_tiles, lambda: 0.0, lambda: -10.0)

    return reward



# Test function for a 16x16 array
def test_compute_reward():
    # Define a 16x16 array with the "M" pattern
    curr_array_m = jnp.array([
        [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1],
        [1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1],
        [1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1],
        [1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 3, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 3, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 2, 3, 2, 1, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
        [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
    ])

    # Define a 16x16 array without the "M" pattern
    curr_array_non_m = jnp.ones((16, 16)) * 2  # All cells set to 2 (walls)

    # Compute rewards
    reward_m = compute_reward(None, None, curr_array_m, None)
    reward_non_m = compute_reward(None, None, curr_array_non_m, None)

    print(f"Reward for array with 'M' pattern: {reward_m}")
    print(f"Reward for array without 'M' pattern: {reward_non_m}")

# Run the test
test_compute_reward()