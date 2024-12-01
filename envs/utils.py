from functools import partial

import cv2
import jax
import jax.numpy as jnp
from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt  # For generating color palettes
from PIL import Image
from jax import jit


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


partial(jit, static_argnums=(0, 1))
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
    colors = jnp.array([(int(r * 255), int(g * 255), int(b * 255), 255) for r, g, b, _ in palette.colors])

    # Shuffle the colors using JAX
    key = jax.random.PRNGKey(seed)
    shuffled_indices = jax.random.permutation(key, num_colors)
    shuffled_colors = colors[shuffled_indices]

    # Convert JAX array back to a list of tuples
    return jnp.array(shuffled_colors, dtype=jnp.uint8)

@partial(jit, static_argnums=(0, 1))
def generate_offset_palette(num_offsets, range_x=(-5, 5), range_y=(-5, 5), seed=0):
    """
    Generate a shuffled offset palette with random (x, y) values using JAX.

    Args:
        num_offsets (int): Number of offsets to generate.
        range_x (tuple): Range for x values (min, max).
        range_y (tuple): Range for y values (min, max).
        seed (int): Random seed for shuffling.

    Returns:
        list: A shuffled list of (x, y) offset tuples.
    """
    # Generate random offsets
    key = jax.random.PRNGKey(seed)
    key_x, key_y = jax.random.split(key)

    x_offsets = jax.random.randint(key_x, shape=(num_offsets,), minval=range_x[0], maxval=range_x[1] + 1)
    y_offsets = jax.random.randint(key_y, shape=(num_offsets,), minval=range_y[0], maxval=range_y[1] + 1)

    offsets = jnp.stack([x_offsets, y_offsets], axis=1)

    # Shuffle the offsets
    shuffled_indices = jax.random.permutation(key, num_offsets)
    shuffled_offsets = offsets[shuffled_indices]

    # Convert JAX array back to a list of tuples
    return jnp.array([jnp.array(offset) for offset in shuffled_offsets], dtype=jnp.int32)


def draw_circle_with_jax(image, center, radius, color, thickness=1):
    """
    Draws a circle on an RGBA image using JAX.

    Parameters:
    - tile_size: Size of the square image.
    - center: (x, y) coordinates of the circle's center.
    - radius: Radius of the circle.
    - color: RGBA color (array of 4 values in range 0-255).
    - thickness: Thickness of the circle's border. Use -1 for filled circle.

    Returns:
    - image: RGBA image with the circle drawn.
    """
    # Create a blank RGBA image

    # Generate grid of coordinates
    height, width, _ = image.shape  # Get the canvas size

    # Generate grid of coordinates
    y, x = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing='ij')

    # Compute distance of each pixel from the circle's center
    distance_from_center = jnp.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Create a mask for the circle's border
    if thickness > 0:
        inner_radius = radius - thickness
        circle_mask = (distance_from_center <= radius) & (distance_from_center >= inner_radius)
    else:
        # For filled circle
        circle_mask = distance_from_center <= radius

    # Apply the color to the circle mask
    for i in range(4):  # Iterate over RGBA channels
        image = image.at[:, :, i].set(jnp.where(circle_mask, color[i], image[:, :, i]))

    return image

partial(jit, static_argnums=(0, 1, 2, 3))
def create_rgba_circle(tile_size, thickness=2, color=jnp.array([255, 255, 255, 128]),
                       alpha=0.7,
                       return_image=True):
    """
    Create an RGBA circle with transparency using OpenCV and convert it to a PIL Image.

    Args:
        tile_size (int): The size of the square image (width and height in pixels).
        thickness (int): The thickness of the circle's outline.
        color (tuple): The RGBA color of the circle (R, G, B, A).

    Returns:
        PIL.Image.Image: The circle as a PIL Image.
    """


    # set the index 3 item to by multiplied by alpha
    # multiply the alpha value by the alpha value
    color.at[3].set(color[3] * alpha // 1)
    # Create a blank RGBA image
    circle = jnp.zeros((tile_size, tile_size, 4), dtype=np.uint8)

    # Draw the circle with the specified RGBA color


    circle = draw_circle_with_jax(circle, (tile_size // 2, tile_size // 2), tile_size // 6 - thickness // 6, color, thickness)


    if return_image:
        circle = Image.fromarray(circle)

    return circle