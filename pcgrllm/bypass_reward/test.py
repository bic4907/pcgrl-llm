import jax
import jax.numpy as jnp
from jax import jit


@jit
def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    reward = 0.0

    # Define constants
    EMPTY = 1
    WALL = 2
    ROOM_SIZE = 3
    PATH_LENGTH = 8

    # Function to count rooms of a specific size
    def count_rooms(array, room_size):
        room_count = 0
        for i in range(array.shape[0] - room_size + 1):
            for j in range(array.shape[1] - room_size + 1):
                sub_array = array[i:i+room_size, j:j+room_size]
                room_count = jax.lax.cond(
                    jnp.all(sub_array == WALL),
                    lambda x: x + 1,
                    lambda x: x,
                    room_count
                )
        return room_count

    # Function to check for a path of a specific length
    def has_path(array, path_length):
        path_count = 0
        for i in range(array.shape[0]):
            for j in range(array.shape[1] - path_length + 1):
                sub_array = array[i, j:j+path_length]
                path_count = jax.lax.cond(
                    jnp.all(sub_array == EMPTY),
                    lambda x: x + 1,
                    lambda x: x,
                    path_count
                )
        return path_count > 0

    # Calculate the number of rooms in the current and previous arrays
    curr_room_count = count_rooms(curr_array, ROOM_SIZE)
    prev_room_count = count_rooms(prev_array, ROOM_SIZE)

    # Calculate the presence of a path in the current and previous arrays
    curr_has_path = has_path(curr_array, PATH_LENGTH)
    prev_has_path = has_path(prev_array, PATH_LENGTH)

    # Reward for the number of rooms
    room_diff = curr_room_count - prev_room_count
    reward += room_diff * 1.0

    # Reward for the presence of a path
    path_diff = int(curr_has_path) - int(prev_has_path)
    reward += path_diff * 1.0

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
    reward_m = compute_reward(curr_array_m, None, curr_array_m, None)
    reward_non_m = compute_reward(None, None, curr_array_non_m, None)

    print(f"Reward for array with 'M' pattern: {reward_m}")
    print(f"Reward for array without 'M' pattern: {reward_non_m}")

# Run the test
test_compute_reward()