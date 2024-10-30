import jax.numpy as jnp
import jax
from jax import jit


@jit
def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    reward = 0.0

    EMPTY = 1
    WALL = 2
    ROOM_SIZE = 3
    PATH_LENGTH = 8

    # Optimized room counting using vectorized operations
    def count_rooms(array, room_size):
        room_count = 0
        for i in range(array.shape[0] - room_size + 1):
            for j in range(array.shape[1] - room_size + 1):
                sub_array = array[i:i+room_size, j:j+room_size]
                room_count += jnp.all(sub_array == WALL)
        return room_count

    # Optimized path check with vectorized operations
    def has_path(array, path_length):
        flattened = array.flatten()
        # Count consecutive EMPTY sequences in the flattened array
        consecutive_empty = (flattened == EMPTY).astype(int)
        max_consecutive_empty = jnp.max(jnp.convolve(consecutive_empty, jnp.ones(path_length), mode='valid'))
        return max_consecutive_empty >= path_length



    prev_room_count = count_rooms(prev_array, ROOM_SIZE)
    curr_room_count = count_rooms(curr_array, ROOM_SIZE)
    

    # prev_has_path = has_path(prev_array, PATH_LENGTH)
    # curr_has_path = has_path(curr_array, PATH_LENGTH)

    # room_count_diff = curr_room_count - prev_room_count
    # path_diff = curr_has_path.astype(int) - prev_has_path.astype(int)

    # reward += room_count_diff * 0.5
    # reward += path_diff * 1.0

    return reward