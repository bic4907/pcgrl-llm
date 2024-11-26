import jax.numpy as jnp
from os.path import join, dirname
import numpy as np
from PIL import Image

image_dir = join(dirname(__file__), '..', 'envs', 'probs', 'tile_ims')

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

    # Create a blank image with the appropriate size
    img = Image.new('RGB', (array.shape[1] * tile_size, array.shape[0] * tile_size), color='white')

    # Paste tiles onto the image
    for y, row in enumerate(array):
        for x, tile in enumerate(row):
            val = int(tile)
            resized_tile = tiles[val].resize((tile_size, tile_size))
            img.paste(resized_tile, (x * tile_size, y * tile_size))

    if return_numpy:
        return np.array(img)
    else:
        img.show()

if __name__ == '__main__':
    arr = jnp.ones((16, 16), dtype=jnp.int32)
    arr = arr.at[5:11, 5:11].set(2)
    render_level(arr, tile_size=32)  # Example usage with a custom tile size
