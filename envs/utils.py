import cv2
import jax
import jax.numpy as jnp
from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt  # For generating color palettes
from PIL.Image import Image


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



def generate_color_palette(num_colors, seed=0):
    """
    Generate a shuffled color palette with distinct colors using JAX.

    Args:
        num_colors (int): Number of colors to generate.
        seed (int): Random seed for shuffling.

    Returns:
        list: A shuffled list of RGBA tuples.
    """
    # Generate a color palette using matplotlib
    palette = plt.cm.get_cmap('tab20', num_colors)  # Use a distinct colormap
    colors = jnp.array([(int(r * 255), int(g * 255), int(b * 255), 128) for r, g, b, _ in palette.colors])

    # Shuffle the colors using JAX
    key = jax.random.PRNGKey(seed)
    shuffled_indices = jax.random.permutation(key, num_colors)
    shuffled_colors = colors[shuffled_indices]

    # Convert JAX array back to a list of tuples
    return shuffled_colors


def create_rgba_circle(tile_size, thickness=2, color=(255, 255, 255, 128), alpha=1.0):
    """
    Create an RGBA circle with transparency using OpenCV and convert it to a PIL Image.

    Args:
        tile_size (int): The size of the square image (width and height in pixels).
        thickness (int): The thickness of the circle's outline.
        color (tuple): The RGBA color of the circle (R, G, B, A).

    Returns:
        PIL.Image.Image: The circle as a PIL Image.
    """
    color = list(color)
    color[3] = int(color[3] * alpha)  # Adjust the alpha value

    # Create a blank RGBA image
    circle_image = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)

    # Draw the circle with the specified RGBA color
    cv2.circle(
        circle_image,
        (tile_size // 2, tile_size // 2),  # Center of the circle
        tile_size // 2 - thickness // 2,  # Radius of the circle
        color,  # RGBA color (B, G, R, A in OpenCV)
        thickness
    )

    # Convert BGRA to RGBA (OpenCV uses BGRA by default)
    circle_image = cv2.cvtColor(circle_image, cv2.COLOR_BGRA2RGBA)

    # Convert the NumPy array to a PIL Image
    circle = Image.fromarray(circle_image)

    return circle