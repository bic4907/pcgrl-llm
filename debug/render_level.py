

import jax.numpy as jnp
from os.path import join, dirname
import numpy as np
from PIL import Image

image_dir = join(dirname(__file__), '..', 'envs', 'probs', 'tile_ims')

def render_level(array: np.array):
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

    img = Image.new('RGB', (array.shape[1] * 16, array.shape[0] * 16), color='white')
    for y, row in enumerate(array):
        for x, tile in enumerate(row):
            val = int(tile)
            img.paste(tiles[val], (x * 16, y * 16))
    img.show()


if __name__ == '__main__':
    arr = jnp.ones((16, 16), dtype=jnp.int32)
    arr = arr.at[5: 11, 5: 11].set(2)
    render_level(arr)