import cv2
import jax.numpy as jnp
from os.path import join, dirname
import numpy as np
from PIL import Image

from envs.pathfinding import check_event
from envs.probs.dungeon3 import Dungeon3Tiles, Dungeon3Problem
from envs.utils import generate_color_palette, create_rgba_circle

image_dir = join(dirname(__file__), '..', 'envs', 'probs', 'tile_ims')

import jax.numpy as jnp


def render_level(array: np.array, tile_size: int = 16, return_numpy: bool = False):
    """
    Render a level array into an image using tile images.

    Args:
        array (np.array): The level array to render.
        tile_size (int): The size of each tile in pixels.
        return_numpy (bool): Whether to return the image as a numpy array.

    Returns:
        PIL.Image.Image or np.array: The rendered image as a PIL image or numpy array.
    """
    tiles = {
        1: Image.open(join(image_dir, 'empty.png')),
        2: Image.open(join(image_dir, 'solid.png')),
        3: Image.open(join(image_dir, 'player.png')),
        4: Image.open(join(image_dir, 'bat.png')),
        5: Image.open(join(image_dir, 'scorpion.png')),
        6: Image.open(join(image_dir, 'spider.png')),
        7: Image.open(join(image_dir, 'key.png')),
        8: Image.open(join(image_dir, 'door.png')),
    }

    # Generate a color palette for the circles
    num_solutions = 10  # Adjust based on the expected number of solutions
    color_palette = generate_color_palette(num_solutions)

    # Create a blank image with the appropriate size
    img = Image.new('RGB', (array.shape[1] * tile_size, array.shape[0] * tile_size), color='white')

    # Paste tiles onto the image
    for y, row in enumerate(array):
        for x, tile in enumerate(row):
            val = int(tile)
            resized_tile = tiles[val].resize((tile_size, tile_size))
            img.paste(resized_tile, (x * tile_size, y * tile_size))

    level = jnp.array(array)
    passable_tiles = Dungeon3Problem.passable_tiles
    p_xy = jnp.argwhere(level == Dungeon3Tiles.PLAYER, size=1, fill_value=-1)[0]
    d_xy = jnp.argwhere(level == Dungeon3Tiles.DOOR, size=1, fill_value=-1)[0]

    # Assign a color to each solution
    for solution_idx, key in enumerate(jnp.argwhere(level == Dungeon3Tiles.KEY)):
        color = color_palette[solution_idx % len(color_palette)]
        circle = create_rgba_circle(tile_size, color=color, alpha=0.7)

        cnt, solutions = check_event(env_map=level,
                                     passable_tiles=passable_tiles,
                                     src=p_xy,
                                     key=key,
                                     trg=d_xy)
        if solutions is not None:
            for solution in solutions[0]:
                for point in solution:
                    point = list(point)
                    y, x = point

                    if x == -1 or y == -1:
                        continue

                    # Draw the circle on the image
                    img.paste(circle, (x * tile_size, y * tile_size), circle)

    if return_numpy:
        return np.array(img)
    else:
        img.show()
