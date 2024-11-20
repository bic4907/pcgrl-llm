import jax.numpy as jnp
from enum import IntEnum

import numpy as np


class Tiles(IntEnum):
    BORDER = 0
    EMPTY = 1


def idx_dict_to_arr(d):
    """Convert dictionary to array, where dictionary has form (index: value)."""
    return np.array([d[i] for i in range(len(d))])



def get_available_tile_mapping(tile_enum, unavailable_tiles):
    """
    Generate a mapping for available tiles, excluding unavailable tiles.

    Args:
        tile_enum (dict): A dictionary mapping tile names to their indices.
        tile_probs (dict): A dictionary mapping tile names to their probabilities.
        unavailable_tiles (set): A set of tile names that are unavailable.

    Returns:
        available_tile_enum (dict): A dictionary mapping available tile names to their new indices.
        available_tile_probs (dict): A dictionary mapping available tile names to their probabilities.
        index_mapping (dict): A dictionary mapping old indices to new indices (for remapping purposes).
    """

    max_index = len(tile_enum)  # Total number of tiles
    index_mapping = jnp.full(max_index, -1, dtype=jnp.int32)  # Initialize with -1

    current_index = 0

    for tile in tile_enum:
        if tile.value == 0:
            continue
        if tile.value not in unavailable_tiles:
            index_mapping = index_mapping.at[current_index].set(tile.value)
            current_index += 1

    return index_mapping